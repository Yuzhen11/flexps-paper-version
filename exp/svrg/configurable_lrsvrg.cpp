#include <algorithm>
#include <cmath>
#include <vector>

#include "core/color.hpp"
#include "datastore/datastore.hpp"
#include "datastore/batch_data_sampler.hpp"
#include "datastore/datastore_utils.hpp"
#include "kvstore/kvstore.hpp"
#include "worker/engine.hpp"

#include "lib/load_data.hpp"

#include "exp/svrg/svrg_helper.hpp"

// *Sample setup*
//
// input=hdfs:///ml/webspam
// alpha=0.05
// num_features=16609143
// data_size=350000
// fgd_threads_per_worker=10
// sgd_overall_batch_size=30000
// outer_loop=5
using namespace husky;
using husky::lib::ml::LabeledPointHObj;

void debug_info(const Info& info, const std::string& debug_info, int process_id) {
    if (info.get_cluster_id() == process_id) {
        husky::LOG_I << debug_info;
    } else {
        husky::LOG_I << debug_info;
    }
}
template <typename T>
void debug_kvstore(std::vector<T>& obj, const std::string& debug_info) {
    std::string tmp;
    for (auto i : obj) {
        tmp += std::to_string(i) + "_";
    }

    husky::LOG_I << "Debug kvstore " + debug_info + ": " + tmp;
}

void debug_sparse_vec(std::vector<std::pair<int, float>>& obj, const std::string& debug_info) {
    std::string tmp;
    for (auto i : obj) {
        tmp += "("+ std::to_string(i.first) + "," + std::to_string(i.second)+ ")" + "_";
    }

    husky::LOG_I << "Debug kvstore " + debug_info + ": " + tmp;
}

void test_error(std::vector<float>& rets_w, datastore::DataStore<LabeledPointHObj<float, float, true>>& data_store,
                const std::string& error_info = "") {
    datastore::DataIterator<LabeledPointHObj<float, float, true>> data_iterator(data_store);

    std::string debug_kvstore_w;
    int flag = 1;
    int count = 0;
    float c_count = 0;  /// correct count
    while (data_iterator.has_next()) {
        // get next data
        auto& data = data_iterator.next();

        count = count + 1;
        auto& x = data.x;
        float y = data.y;
        if (y < 0)
            y = 0;

        float pred_y = 0.0;
        for (auto field : x) {
            // calculate predict y
            pred_y += rets_w[field.fea] * field.val;
            if (flag) {
                debug_kvstore_w += "__" + std::to_string(rets_w[field.fea]);
            }
        }
        pred_y = 1. / (1. + exp(-1 * pred_y));

        pred_y = (pred_y > 0.5) ? 1 : 0;
        if (int(pred_y) == int(y)) {
            c_count += 1;
        }
        flag = 0;
    }

    //    husky::LOG_I << "current w: " + debug_kvstore_w;
    husky::LOG_I << error_info + " accuracy is " << std::to_string(c_count / count);
}


std::map<std::string, int> get_my_part_u_range(int num_params, const Info& info) {
    int my_cluster_id = info.get_cluster_id();
    int total_num_threads = info.get_num_workers();

    int average_num_params = num_params / total_num_threads; // an integer division
    int my_part_u_start = my_cluster_id * average_num_params;
    int my_part_u_end = (my_cluster_id + 1  == total_num_threads) ? num_params : my_part_u_start + average_num_params;
    int my_part_u_len = my_part_u_end - my_part_u_start;
    std::map<std::string, int> str2int;
    str2int["start"] = my_part_u_start;
    str2int["length"] = my_part_u_len;
    return str2int;
}

std::vector<std::pair<int, float>> get_my_part_u(int num_params, const Info& info, ml::mlworker2::PSBspWorker<float>& mlworker_u) {
    // the key range is: [cluster_id * avg, min((cluster_id+1)*avg, num_params))
    std::map<std::string, int> range_info = get_my_part_u_range(num_params, info);
    std::vector<constants::Key> my_part_u_keys(range_info["length"]);
    std::iota(my_part_u_keys.begin(), my_part_u_keys.end(), range_info["start"]);

    std::vector<float> my_part_u(range_info["length"]);

    mlworker_u.Pull(my_part_u_keys, &my_part_u); // set consisency control to be flase since u is read-only in SGD stage
    mlworker_u.Push({}, {}); // issue dummy push

    std::vector<std::pair<int, float>> res;
    int start = range_info["start"];
    for (int i=0; i<my_part_u.size(); i++) {
        res.emplace_back(std::make_pair(i+start, my_part_u[i]));
    }
    return res;
}

std::vector<float> read_all_w(int num_params, ml::mlworker2::PSBspWorker<float>& mlworker_w) {
    std::vector<husky::constants::Key> all_keys(num_params);
    std::iota(all_keys.begin(), all_keys.end(), 0);
    // pull from kvstore_w
    std::vector<float> w0;
    mlworker_w.Pull(all_keys, &w0);// disable consistency control since it's read-only
    mlworker_w.Push({}, {});
    return w0;
}

void get_gradient(const std::vector<LabeledPointHObj<float, float, true>*>& m, const std::vector<husky::constants::Key>& batch_keys, const std::vector<float>& w, std::vector<float>* w_updates) {
    if (m.size()==0) return;
	if (w_updates->size()==0) w_updates->resize(batch_keys.size());
    for (auto data : m) {
 		auto& x = data->x;
       	auto y = data->y;
       	if (y<0) y = 0;

       	float pred_y = 0.0;
       	int i = 0;
	   	for(auto field : x) {
       		while (batch_keys[i] < field.fea) i += 1;
       	    pred_y += w[i] * field.val;
       	}
       	pred_y = 1. / (1. + exp(-1 * pred_y));
       	i = 0;
       	for (auto field : x) {
       	    while (batch_keys[i] < field.fea) i += 1;
       	    w_updates->at(i) += field.val * (pred_y - y);
       	}
   	}	
	int batch_size = batch_keys.size();
	for (auto& a : *w_updates) {
        a /= static_cast<float>(batch_size);
	}
}

void get_gradient_diff(const std::vector<LabeledPointHObj<float, float, true>*>& m,
        const std::vector<husky::constants::Key>& batch_keys, const std::vector<float>& w,
        const std::vector<husky::constants::Key>& all_keys, const std::vector<float>& w0,
        std::vector<float>* w_updates) {

    if (m.size()==0) return;
	if (w_updates->size()==0) w_updates->resize(batch_keys.size());
    for (auto data : m) {
 		auto& x = data->x;
       	auto y = data->y;
       	if (y<0) y = 0;

       	float pred_y = 0.0;
        float pred_y0 = 0.0;
       	int i = 0;
	   	for(auto field : x) {
       		while (batch_keys[i] < field.fea) i += 1;
            if (batch_keys[i] != field.fea) LOG_I<<"key not equal to fea";
       	    pred_y += w[i] * field.val;
            pred_y0 += w0[field.fea] * field.val;
       	}
       	pred_y = 1. / (1. + exp(-1 * pred_y));
       	pred_y0 = 1. / (1. + exp(-1 * pred_y0));
       	i = 0;
       	for (auto field : x) {
       	    while (batch_keys[i] < field.fea) i += 1;
       	    w_updates->at(i) += field.val * (pred_y0 - pred_y);
       	}
   	}	
    // get the avg
	int batch_size = batch_keys.size();
	for (auto& a : *w_updates) {
        a /= static_cast<float>(batch_size);
	}
}
 

void update_w(int kv_w, std::vector<std::pair<int, float>>& w_updates, ml::mlworker2::PSBspWorker<float>& mlworker_w) {
}

void minus_my_part_u(const std::vector<husky::constants::Key>& batch_keys, const std::vector<float>& w_updates,
        const std::vector<std::pair<int, float>>& my_part_u, std::vector<std::pair<int, float>>* delta) {
    if (batch_keys.size() == 0) {
        *delta = my_part_u;
        for (auto& a : *delta)
            a.second *= -1;
    }

    if (my_part_u.size() == 0) {
        delta->resize(batch_keys.size());
        for (int i=0; i<batch_keys.size(); i++) {
           (*delta)[i].first = batch_keys[i];
           (*delta)[i].second = w_updates[i];
        }
    }

    int i = 0;
    int j = 0;
    delta->reserve(batch_keys.size() + my_part_u.size());
    while(i<batch_keys.size() || j<my_part_u.size()) {
        int l;
        if (i == batch_keys.size()) {
            l = std::numeric_limits<int>::max(); 
        } else {
            l = batch_keys[i];
        }

        int r;
        if (j == my_part_u.size()) {
            r = std::numeric_limits<int>::max(); 
        } else {
            r = my_part_u[j].first;
        }

        if (l < r) {
            delta->push_back(std::make_pair(l, w_updates[i]));
            i++;
        } else if (r < l) {
            delta->push_back(std::make_pair(r, -1 * my_part_u[j].second));
            j++;
        } else {
            delta->push_back(std::make_pair(l, w_updates[i] - my_part_u[j].second));
            i++;
            j++;
        }
    }
}

        
//// gradient using wm as parameter
//std::vector<float> w_updates;
//get_gradient(mini_batch.get_data_ptrs(), batch_keys, w, &w_updates);

//// gradient using w0 as parameter 
//std::vector<float> w_updates0;
//get_gradient(mini_batch.get_data_ptrs(), all_keys, w0, &w_updates0);

//// gra(w0) - gra(wm)
//for (int i=0; i<batch_keys.size(); i++) {
//    w_updates0[batch_keys[i]] -= w_updates[i];
//}

void SGD_update(datastore::DataStore<LabeledPointHObj<float, float, true>>& data_store, int kv_w, int kv_u,
                float alpha, int num_params, int inner_loop, int batch_size_per_worker, const Info& info,
                ml::mlworker2::PSBspWorker<float>& mlworker_u, ml::mlworker2::PSBspWorker<float>& mlworker_w) {
    
    auto start_time = std::chrono::steady_clock::now();
    datastore::BatchDataSampler<LabeledPointHObj<float, float, true>> mini_batch(data_store, batch_size_per_worker);
    // Parameters need to be stored in kv store:
    // kv_u:1. store the FGD result, passed from FGD stage to SGD stage
    //      2. consistent across different mini-batches in SGD
    // kv_w:1. pull from kvstore once
    //      2. consistent in each small batch
    //      3. synchronized before starting next batch
    //
    //std::vector<std::pair<int, float>> my_part_u = get_my_part_u(num_params, info, mlworker_u);
    std::vector<std::pair<int, float>> my_part_u;
    std::vector<husky::constants::Key> all_keys(num_params);
    std::iota(all_keys.begin(), all_keys.end(), 0);
    std::vector<float> w0 = read_all_w(num_params, mlworker_w);
    
    
    auto start_time_inner = std::chrono::steady_clock::now();
    mini_batch.random_start_point();
    // every inner loop contains one mini batch
    for (int m=0; m<inner_loop; m++) {
        if (m == 1) {
            start_time_inner = std::chrono::steady_clock::now();
        }
        std::vector<husky::constants::Key> batch_keys = mini_batch.prepare_next_batch();
        std::vector<float> w;
        // mlworker_w.Pull(batch_keys, &w);
        // mlworker_w.Push(batch_keys, w);
        auto start_time_a = std::chrono::steady_clock::now();
        mlworker_w.GetKVWorker()->Wait(kv_w, mlworker_w.GetKVWorker()->Pull(kv_w, batch_keys, &w));

        auto start_time_c = std::chrono::steady_clock::now();
        
        // gra(w0) - gra(wm)
        std::vector<float> w_updates0;
        get_gradient_diff(mini_batch.get_data_ptrs(), batch_keys, w, all_keys, w0, &w_updates0);
        
        for (auto& w : w_updates0) {
            w = w * alpha;
        }

        //std::vector<std::pair<int, float>> delta;
        //// gra(w0) - gra(wm) - my_part_u
        //minus_my_part_u(batch_keys, w_updates0, my_part_u, &delta);
        //for (auto& w : delta) {
        //    w.second = w.second * alpha;
        //}

        //std::vector<constants::Key> update_keys;
        //std::vector<float> update_vals(delta.size());
        //for (auto a : delta) {
        //    update_keys.push_back(a.first);
        //    update_vals.push_back(a.second);
        //}

        auto start_time_d = std::chrono::steady_clock::now();
        //mlworker_w.GetKVWorker()->Wait(kv_w, mlworker_w.GetKVWorker()->Push(kv_w, update_keys, update_vals));
        mlworker_w.GetKVWorker()->Wait(kv_w, mlworker_w.GetKVWorker()->Push(kv_w, batch_keys, w_updates0));
        auto start_time_e = std::chrono::steady_clock::now();
        if (info.get_cluster_id() == 0&& (m+1)%1000 == 0)
	        LOG_I<< m <<" batchkeys size:" <<batch_keys.size()<< " total time:" << (std::chrono::duration_cast<std::chrono::milliseconds>(start_time_e - start_time_a)).count();
	    //LOG_I<< "computation time:" << (std::chrono::duration_cast<std::chrono::milliseconds>(start_time_d - start_time_c)).count()<<"";
    }
    auto end_time_inner = std::chrono::steady_clock::now();
	LOG_I<< "avg"<<info.get_current_epoch()<<" time:" << (std::chrono::duration_cast<std::chrono::milliseconds>(end_time_inner - start_time_inner)).count()/float(inner_loop-1)<<"ms";
}

void FGD_update(datastore::DataStore<LabeledPointHObj<float, float, true>>& data_store, int kv_w, int kv_u,
                float alpha, int num_params, const Info& info, size_t data_size,
                ml::mlworker2::PSBspWorker<float>& mlworker_u, ml::mlworker2::PSBspWorker<float>& mlworker_w) {

    // get configurable_task's worker_num vector, worker_num vector describes how many threads each epoch will use
    std::vector<int> worker_num = static_cast<const ConfigurableWorkersTask*>(info.get_task())->get_worker_num();

    int current_epoch = info.get_current_epoch();
    int num_processes = info.get_worker_info().get_num_processes();

    /* Compute u avg full gradient 
     * new_u = clear_delta + new_u
     * 1. the first part of the updates should clear the original u 
     * 2. the second part of the updates should give the new u
     */

    // 1. compute the clear_delta  
    std::vector<husky::constants::Key> all_keys(num_params);
    std::iota(all_keys.begin(), all_keys.end(), 0);
    std::vector<float> rets_u;
    auto FGD_1 = std::chrono::steady_clock::now();
    mlworker_u.Pull(all_keys, &rets_u);
    auto FGD_15 = std::chrono::steady_clock::now();
    std::vector<float> all_w = read_all_w(num_params, mlworker_w);
    auto FGD_2 = std::chrono::steady_clock::now();

    // compute clear_delta to clear kv_u before each FGD to avoid u accumulating
    std::vector<float> clear_delta(num_params);
    for (int i = 0; i < rets_u.size(); i++) {
        // so in order to clear kvstore, each thread should push a part value(-1 * old_value / num_threads) to reset
        // kvstore
        clear_delta[i] = -1 * (rets_u[i] / (worker_num[current_epoch % worker_num.size()] * num_processes));
    }

    // 2. compute the avg gradient using rest_w 
    // position is used to balance the data load
    // when do FGD, we hope data can be loaded balancely
    int position;
    // get all threads in local process
    std::vector<int> tids = info.get_worker_info().get_local_tids();
    for (int k = 0; k < tids.size(); k++) {
        // the position info will be used fo loading data balancely
        if (tids[k] == info.get_global_id()) {
            position = k;
            break;
        }
    }
    datastore::DataLoadBalance<LabeledPointHObj<float, float, true>> data_load_balance(
        data_store, worker_num.at(current_epoch % worker_num.size()), position);

    std::vector<float> delta(num_params);
    // go through all data
    while (data_load_balance.has_next()) {
        auto& data = data_load_balance.next();
        auto& x = data.x;
        float y = data.y;
        float pred_y = 0.0;
        if (y < 0) y = 0;
        for (auto field : x) {
            pred_y += all_w[field.fea] * field.val;
        }
        pred_y = 1. / (1. + exp(-1 * pred_y));

        // calculate GD
        for (auto field : x) {
            // calculate a global delta after going through all data
            // so we need to accumulate all delta
            delta[field.fea] += field.val * (pred_y - y);
        }
    }

    // aggregate count to get accurate avg
    // calculate avg delta to push to kvstore
    for (int i = 0; i < delta.size(); i++) {
        delta[i] = delta[i] / data_size + clear_delta[i];
    }

    auto FGD_3 = std::chrono::steady_clock::now();
    // push kv_u
    mlworker_u.Push(all_keys, delta);
    auto FGD_4 = std::chrono::steady_clock::now();

    if (info.get_cluster_id() == 0) {
        auto pull_time = std::chrono::duration_cast<std::chrono::milliseconds>(FGD_2 - FGD_1).count();
        auto read_all_w = std::chrono::duration_cast<std::chrono::milliseconds>(FGD_2 - FGD_15).count();
        auto computation_time = std::chrono::duration_cast<std::chrono::milliseconds>(FGD_3 - FGD_2).count();
        auto push_time = std::chrono::duration_cast<std::chrono::milliseconds>(FGD_4 - FGD_3).count();
        LOG_I<<"FGD pull time:"<<std::to_string(pull_time);
        LOG_I<<"FGD read_all_w time:"<<std::to_string(read_all_w);
        LOG_I<<"FGD compuation time:"<<std::to_string(computation_time);
        LOG_I<<"FGD push time:"<<std::to_string(push_time);
    }
}

int main(int argc, char** argv) {
    auto start_time0 = std::chrono::steady_clock::now();
    // Set config
    bool rt = init_with_args(argc, argv,
                             {
                                 "alpha",
                                 "data_size",         // Need to know the number of data records for average gradient
                                 "num_features",
                                 "num_load_workers",  // Use this number of workers to load data
                                 "num_train_workers",
                                 "fgd_threads_per_worker",  // The number of worker per process for fgd
                                 "sgd_threads_per_worker",  // The number of worker per process for fgd
                                 "sgd_overall_batch_size",                // Total batch size in sgd stage?
                                 "outer_loop",      // Number of outerloop i.e. (fgd+sgd)
                                 "inner_loop"       // Number of inner_loop i.e. number of minibatches
                             });

    float alpha = std::stof(Context::get_param("alpha"));
    int data_size = std::stoi(Context::get_param("data_size"));
    int num_features = std::stoi(Context::get_param("num_features"));
    int num_params = num_features + 1;  // +1 for intercept

    int num_load_workers = std::stoi(Context::get_param("num_load_workers"));
    int num_train_workers = std::stoi(Context::get_param("num_train_workers"));

    int fgd_threads_per_worker = std::stoi(Context::get_param("fgd_threads_per_worker"));
    int sgd_threads_per_worker = std::stoi(Context::get_param("sgd_threads_per_worker"));
    int sgd_overall_batch_size = std::stoi(Context::get_param("sgd_overall_batch_size"));
    int outer_loop = std::stoi(Context::get_param("outer_loop"));
    int inner_loop = std::stoi(Context::get_param("inner_loop"));

    auto& engine = Engine::Get();
    // start kvstore, should start after mailbox is up
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());
    // create DataStore
    datastore::DataStore<LabeledPointHObj<float, float, true>> data_store(Context::get_worker_info().get_num_local_workers());

    // create and submit load_task
    auto load_task = TaskFactory::Get().CreateTask<Task>();  // 1 epoch, 1 workers
    load_task.set_num_workers(num_load_workers);
    engine.AddTask(std::move(load_task), [&data_store, &num_features](const Info& info) {
        auto local_id = info.get_local_id();
        load_data(Context::get_param("input"), data_store, DataFormat::kLIBSVMFormat, num_features, local_id);
        datastore::DataStoreWrapper<LabeledPointHObj<float, float, true>> data_store_wrapper(data_store);
        husky::LOG_I << YELLOW("datasize: " + std::to_string(data_store_wrapper.get_data_size()));

    });
    auto start_time = std::chrono::steady_clock::now();
    engine.Submit();
    auto end_time = std::chrono::steady_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    if (Context::get_worker_info().get_process_id() == 0)
        husky::LOG_I << YELLOW("Load time: " + std::to_string(load_time) + " ms");

    // create a configurable task and configure the worker_number in FGD stage and SGD stage
    int train_epoch = 2 * outer_loop;
    auto train_task = TaskFactory::Get().CreateTask<ConfigurableWorkersTask>(train_epoch, num_train_workers);

    std::vector<int> worker_num;
    std::vector<std::string> worker_num_type;
    // FGD stage 
    worker_num.push_back(fgd_threads_per_worker);
    worker_num_type.push_back("threads_per_worker");
    // SGD stage 
    worker_num.push_back(sgd_threads_per_worker);
    worker_num_type.push_back("threads_per_worker");
    train_task.set_worker_num(worker_num);
    train_task.set_worker_num_type(*((const std::vector<std::string>*) (&worker_num_type)));

    int kv_w = kvstore::KVStore::Get().CreateKVStore<float>("bsp_add_vector", fgd_threads_per_worker, -1, num_params);  // for bsp server
    int kv_u = kvstore::KVStore::Get().CreateKVStore<float>("bsp_add_vector", sgd_threads_per_worker, -1, num_params);
	int FGD_total_time = 0;
	int SGD_total_time = 0;

    engine.AddTask(std::move(train_task), [&kv_w, &kv_u, &data_store, &alpha, num_params, data_size,
                                           &fgd_threads_per_worker, sgd_threads_per_worker, inner_loop, &sgd_overall_batch_size, &FGD_total_time, &SGD_total_time](const Info& info) {
        auto start_time = std::chrono::steady_clock::now();

        datastore::DataStoreWrapper<LabeledPointHObj<float, float, true>> data_store_wrapper(data_store);

        ml::mlworker2::PSBspWorker<float> mlworker_u(info, *Context::get_zmq_context(), kv_u, num_params);
        ml::mlworker2::PSBspWorker<float> mlworker_w(info, *Context::get_zmq_context(), kv_w, num_params);

        std::string epoch_type;
        int current_epoch = info.get_current_epoch();
        if (current_epoch % 2 == 0) {
            epoch_type = "current_epoch: " + std::to_string(current_epoch) + " FGD";
            if (info.get_cluster_id() == 0) {
                husky::LOG_I << RED("FGD. current_epoch: " + std::to_string(current_epoch));
            }
            // do FGD
            FGD_update(data_store, kv_w, kv_u, alpha, num_params, info, data_size, mlworker_u, mlworker_w);
        } else {
            epoch_type = "current_epoch: " + std::to_string(current_epoch) + " SGD";
            if (info.get_cluster_id() == 0) {
                husky::LOG_I << RED("SGD. current_epoch: " + std::to_string(current_epoch));
            }
            // do SGD
            int total_threads = sgd_threads_per_worker* info.get_worker_info().get_num_processes();
            int batch_size_per_worker = std::ceil(1.0 * std::min(data_size, sgd_overall_batch_size) / total_threads);
            SGD_update(data_store, kv_w, kv_u, alpha, num_params, inner_loop, batch_size_per_worker, info, mlworker_u, mlworker_w);
        }

        auto end_time = std::chrono::steady_clock::now();
        if (info.get_cluster_id() == 0) {
            auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
			if (current_epoch % 2 == 0) {
				FGD_total_time += train_time;
			} else {
				SGD_total_time += train_time;
			}
            husky::LOG_I << YELLOW(epoch_type + " Train time: " + std::to_string(train_time) + " ms");
        }
    });

    // submit train task
    start_time = std::chrono::steady_clock::now();
    engine.Submit();
    end_time = std::chrono::steady_clock::now();
    if (Context::get_worker_info().get_process_id() == 0) {
        auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
		auto end_to_end_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time0).count();
        LOG_I << YELLOW("Total SGD time: " + std::to_string(SGD_total_time) + " ms");
        LOG_I << YELLOW("Total FGD time: " + std::to_string(FGD_total_time) + " ms");
		LOG_I << YELLOW("Total train time: " + std::to_string(SGD_total_time * sgd_threads_per_worker + FGD_total_time * fgd_threads_per_worker) + " ms");
 		LOG_I<<YELLOW("End_to_end time: " + std::to_string(end_to_end_time) + " ms");
    }
    engine.Exit();
    kvstore::KVStore::Get().Stop();
}
