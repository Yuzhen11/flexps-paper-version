#include <vector>

#include "core/color.hpp"
#include "datastore/datastore.hpp"
#include "datastore/datastore_utils.hpp"
#include "kvstore/kvstore.hpp"
#include "worker/engine.hpp"

#include "lib/load_data.hpp"

#include "exp/svrg_helper.hpp"

// *Sample setup*
//
// input=hdfs:///ml/webspam
// alpha=0.05
// num_features=16609143
// data_size=350000
// svrg_exp=3
// svrg_fgd_workers_per_process=10
// sgd_batch_size=30000
// sgd_stage=1
// sgd_continious_epoch=1
// num_sgd_continious_epoch=5
// sgd_exp3_num_threads=10
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

void SGD_update(datastore::DataStore<LabeledPointHObj<float, float, true>>& data_store, int kv_w, int kv_u,
                int kv_w_old, float alpha, int num_params, int num_iters, const Info& info, int num_active_process,
                ml::mlworker2::PSBspWorker<float>& mlworker_u, ml::mlworker2::PSBspWorker<float>& mlworker_w,
                ml::mlworker2::PSBspWorker<float>& mlworker_w_old) {
    // Create a DataSampler for SGD
    datastore::DataLooper<LabeledPointHObj<float, float, true>> data_looper(data_store);

    // init keys
    std::vector<husky::constants::Key> keys;
    for (int i = 0; i < num_params; i++) {
        keys.push_back(i);
    }
    // pull from kvstore_w
    std::vector<float> rets_w;
    mlworker_w.Pull(keys, &rets_w);

    // pull from kvstore_w_old
    std::vector<float> old_rets_w;
    // in SGD, we just pull w_old to update w, so in this step, we let consistency_control false
    mlworker_w_old.Pull(keys, &old_rets_w, false);

    // pull from kvstore_u
    std::vector<float> rets_u;
    // in SGD, we just pull u to update w, so in this step, we let consistency_control false
    mlworker_u.Pull(keys, &rets_u, false);

    std::vector<float> rets_w_delta(num_params);

    // optimization about avoiding dense calculating of w_delta
    // update_counter describes rets_w_delta updating,
    // update_counter_w describes rets_w updating
    std::vector<int> update_counter(num_params, 0);
    std::vector<int> update_counter_w(num_params, 0);
    // record iter
    int iter = 0;
    for (int i = num_iters; i > 0 && data_looper.has_next(); i--) {
        // get next data
        auto& data = data_looper.next();

        // delta and old_delta is sparse
        std::vector<std::pair<int, float>> delta;
        std::vector<std::pair<int, float>> old_delta;

        auto& x = data.x;
        float y = data.y;
        if (y < 0)
            y = 0;
        float pred_y = 0.0;
        float old_pred_y = 0.0;
        // new gradient
        for (auto field : x) {
            // it's a bit tricky, some unprocessed rets_u will be added,
            // some rets_u isn't used by previous data record may be used by this record
            pred_y +=
                (rets_w[field.fea] - alpha * (iter - update_counter_w[field.fea]) * rets_u[field.fea]) * field.val;
            update_counter_w[field.fea] = iter;
        }
        pred_y = 1. / (1. + exp(-1 * pred_y));
        // each data will calculate a delta, once a delta has been calculated, rets_w will be updated
        // so there is no need to accumulate
        for (auto field : x) {
            delta.emplace_back(std::pair<int, float>(field.fea, field.val * (pred_y - y)));
        }

        // old gradient
        for (auto field : x) {
            old_pred_y += old_rets_w[field.fea] * field.val;
        }
        old_pred_y = 1. / (1. + exp(-1 * old_pred_y));
        for (auto field : x) {
            old_delta.emplace_back(std::pair<int, float>(field.fea, field.val * (old_pred_y - y)));
        }

        iter++;
        // optimization: we needn't browsing rets_u which is dense data,
        // in this for-loop, we just need to process some rets_u values whose delta value is not null
        for (int j = 0; j < delta.size(); j++) {
            int idx = delta[j].first;
            // record rets_w_delta
            float w_delta =
                (-1) * alpha * (delta[j].second - old_delta[j].second + (iter - update_counter[idx]) * rets_u[idx]);
            rets_w_delta[idx] += w_delta;
            // update rets_w
            rets_w[idx] += w_delta;
            // record
            update_counter[idx] = iter;
            update_counter_w[idx] = iter;
        }

        if (info.get_cluster_id() == 0 && iter % 1000 == 0) {
            test_error(rets_w, data_store, "iter_" + std::to_string(iter) + "_SGD");
        }
    }

    // to process some rets_u values which have nerver been processed in above loop
    for (int i = 0; i < update_counter.size(); i++) {
        if (update_counter[i] < iter) {
            rets_w_delta[i] += (-1) * alpha * (iter - update_counter[i]) * rets_u[i];
        }
    }

    // num_active_process > 1, means running exp2: SGD is multiple threads, in this case, num_active_process will play a
    // active role to calculate an avg
    if (num_active_process > 1) {
        for (int i = 0; i < rets_w_delta.size(); i++) {
            rets_w_delta[i] /= num_active_process;
        }
    }

    mlworker_w.Push(keys, rets_w_delta);
}

void FGD_update(datastore::DataStore<LabeledPointHObj<float, float, true>>& data_store, int kv_w, int kv_u,
                int kv_w_old, float alpha, int num_params, const Info& info, size_t data_size,
                ml::mlworker2::PSBspWorker<float>& mlworker_u, ml::mlworker2::PSBspWorker<float>& mlworker_w,
                ml::mlworker2::PSBspWorker<float>& mlworker_w_old) {
    // init keys
    std::vector<husky::constants::Key> keys;
    for (int i = 0; i < num_params; i++) {
        keys.push_back(i);
    }

    // pull from kvstore_w
    std::vector<float> rets_w;
    // wait util parameters have been pulled, store into rets_w
    // in FGD, we just pull w to update u, so this step, we let consistency_control false
    mlworker_w.Pull(keys, &rets_w, false);

    std::vector<float> tmp;
    mlworker_w_old.Pull({}, &tmp);

    // get the current_epoch
    int current_epoch = info.get_current_epoch();
    // get configurable_task's worker_num vector, worker_num vector describes how many threads each epoch will use
    std::vector<int> worker_num = static_cast<const ConfigurableWorkersTask*>(info.get_task())->get_worker_num();
    // get all threads in local process
    std::vector<int> tids = info.get_worker_info().get_local_tids();

    // when do FGD, we hope data can be loaded balancely
    // find the pos of local_id
    // position is used for load_data_balance and also used for update kvstore_w_old
    int position;
    for (int k = 0; k < tids.size(); k++) {
        // get the position of current thread
        // the position info will be used fo loading data balancely
        if (tids[k] == info.get_global_id()) {
            position = k;
            break;
        }
    }

    int num_processes = info.get_worker_info().get_num_processes();

    // update kvstore_w_old by kvstore_w, to make kvstore_w_old same with kvstore_w for later SGD step
    // strategy: for performance, each thread of FGD will be responsible for updating a part of kvstore_w_old
    int w_size = rets_w.size();
    // current process responsible part
    int process_w_size = w_size / num_processes;
    // caculate rest
    int rest_process_w_size = w_size - process_w_size * num_processes;
    // current_num_process
    int current_num_process = info.get_worker_info().get_process_id();

    // init range_keys
    std::vector<husky::constants::Key> range_keys;
    std::vector<float> range_values;
    int range_start = 0;
    int range_end = 0;

    int thread_process_w_size;
    int rest_thread_process_w_size;

    if (rest_process_w_size > 0 && current_num_process == num_processes - 1) {  // this process is the last num_process
        thread_process_w_size = (process_w_size + rest_process_w_size) / tids.size();
        rest_thread_process_w_size = process_w_size + rest_process_w_size - thread_process_w_size * tids.size();
    } else {
        // thread_process_w_size means how many keys each thread in the process will be responsible
        thread_process_w_size = process_w_size / tids.size();
        rest_thread_process_w_size = process_w_size - thread_process_w_size * tids.size();
    }

    range_start = process_w_size * current_num_process + position * thread_process_w_size;
    // rest_thread_process_w_size not equals 0, this means the last thread should process all rest w
    if (rest_thread_process_w_size > 0 &&
        position == tids.size() - 1) {  // rest is not 0 && current_thread is the max number
        range_end = range_start + thread_process_w_size + rest_thread_process_w_size;
    } else {
        range_end = range_start + thread_process_w_size;
    }

    // init range_key and value
    for (int i = range_start; i < range_end; i++) {
        range_keys.push_back(i);
        range_values.push_back(rets_w[i]);
    }
    // push kv_w_old
    mlworker_w_old.Push(range_keys, range_values);

    // pull kvstore_u
    std::vector<float> rets_u;
    mlworker_u.Pull(keys, &rets_u);

    std::vector<float> delta(num_params);

    // clear kv_u before each FGD to avoid u accumulating
    std::vector<float> clear_delta(num_params);
    for (int i = 0; i < rets_u.size(); i++) {
        // BSP's updateType is Add to update kvstore,
        // so in order to clear kvstore, each thread should push a part value(-1 * old_value / num_threads) to reset
        // kvstore
        clear_delta[i] = -1 * (rets_u[i] / (worker_num.at(current_epoch % worker_num.size()) * num_processes));
    }

    // push kv_u to clear kv_u
    mlworker_u.Push(keys, clear_delta);

    // Pull kv_u again to ensure Pull/Push/Pull/Push...
    rets_u.clear();
    // kvworker->Wait(kv_u, kvworker->Pull(kv_u, keys, &rets_u, true, true));
    mlworker_u.Pull(keys, &rets_u);

    // dataloadbalance
    datastore::DataLoadBalance<LabeledPointHObj<float, float, true>> data_load_balance(
        data_store, worker_num.at(current_epoch % worker_num.size()), position);

    // go through all data
    int iter = 0;
    while (data_load_balance.has_next()) {
        iter++;
        auto& data = data_load_balance.next();
        auto& x = data.x;
        float y = data.y;
        float pred_y = 0.0;
        if (y < 0)
            y = 0;
        for (auto field : x) {
            pred_y += rets_w[field.fea] * field.val;
        }
        pred_y = 1. / (1. + exp(-1 * pred_y));

        // calculate GD
        for (auto field : x) {
            // calculate a global delta after going through all data
            // so we need to accumulate all delta
            // float tmp = delta[field.fea] + field.val * (pred_y - y);
            // husky::LOG_I << std::to_string(tmp);
            delta[field.fea] += field.val * (pred_y - y);
        }

        if (info.get_cluster_id() == 0 && iter % 10000 == 0) {
            husky::LOG_I << "FGD iter: " + std::to_string(iter);
        }
    }

    // aggregate count to get accurate avg
    // calculate avg delta to push to kvstore
    for (int i = 0; i < delta.size(); i++) {
        delta[i] = delta[i] / data_size;
    }

    // push kv_u
    mlworker_u.Push(keys, delta);
}

int main(int argc, char** argv) {
    // Set config
    bool rt = init_with_args(argc, argv,
                             {
                                 "num_load_workers",  // Use this number of workers to load data
                                 "data_size",         // Need to know the number of data size for average gradient
                                 "svrg_exp",  // svrg exp type in {1,2,3}. TODO: Need to specify what each type means
                                 "svrg_fgd_workers_per_process",  // The number of worker per process for fgd
                                 "sgd_batch_size",                // Total batch size in sgd stage?
                                 "sgd_continious_epoch",          // Number of consecutive sgd stage
                                 "sgd_stage",                     // Just set to 1?
                                 "num_sgd_continious_epoch",      // Number of big stage (fgd+sgd)
                                 "sgd_exp3_num_threads"           // Number of threads in exp3
                             });

    int data_size = std::stoi(Context::get_param("data_size"));
    int num_train_workers = std::stoi(Context::get_param("num_train_workers"));
    int num_load_workers = std::stoi(Context::get_param("num_load_workers"));
    float alpha = std::stof(Context::get_param("alpha"));
    int num_features = std::stoi(Context::get_param("num_features"));
    int num_params = num_features + 1;  // +1 for intercept

    // exp conf
    int svrg_exp = std::stoi(Context::get_param("svrg_exp"));
    int svrg_fgd_workers_per_process = std::stoi(Context::get_param("svrg_fgd_workers_per_process"));

    int sgd_batch_size = std::stoi(Context::get_param("sgd_batch_size"));
    int sgd_continious_epoch = std::stoi(Context::get_param("sgd_continious_epoch"));
    int sgd_stage = std::stoi(Context::get_param("sgd_stage"));
    int num_sgd_continious_epoch = std::stoi(Context::get_param("num_sgd_continious_epoch"));

    int sgd_exp3_num_threads = std::stoi(Context::get_param("sgd_exp3_num_threads"));

    auto& engine = Engine::Get();
    // start the kvstore, should start after mailbox is up
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());

    // create the DataStore
    datastore::DataStore<LabeledPointHObj<float, float, true>> data_store(
        Context::get_worker_info().get_num_local_workers());

    // create load_task
    auto load_task = TaskFactory::Get().CreateTask<HuskyTask>(1, num_load_workers);  // 1 epoch, 1 workers
    engine.AddTask(std::move(load_task), [&data_store, &num_features](const Info& info) {
        auto local_id = info.get_local_id();
        load_data(Context::get_param("input"), data_store, DataFormat::kLIBSVMFormat, num_features, local_id);
        husky::LOG_I << RED("Finished Load Data!");

        datastore::DataStoreWrapper<LabeledPointHObj<float, float, true>> data_store_wrapper(data_store);
        husky::LOG_I << RED("datasize: " + std::to_string(data_store_wrapper.get_data_size()));

    });

    // submit load_task
    auto start_time = std::chrono::steady_clock::now();
    engine.Submit();
    auto end_time = std::chrono::steady_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    if (Context::get_worker_info().get_process_id() == 0)
        husky::LOG_I << YELLOW("Load time: " + std::to_string(load_time) + " ms");

    std::vector<int> worker_num;
    std::vector<std::string> worker_num_type;
    int train_epoch;
    if (svrg_exp == 1) {
        // exp 1: in SGD epoch, run single train thread
        // generate worker_num and worker_num_type based on conf
        // add fgd conf
        worker_num.push_back(svrg_fgd_workers_per_process);
        worker_num_type.push_back("threads_per_worker");
        // add sgd conf
        for (int i = 0; i < sgd_continious_epoch; i++) {
            // in epx1, in sgd epoch, 1 thread is enough
            worker_num.push_back(1);
            worker_num_type.push_back("threads_traverse_cluster");
        }
        // 1 means one fgd, 3 means three sgd
        train_epoch = (1 + sgd_continious_epoch) * num_sgd_continious_epoch;
    } else if (svrg_exp == 2) {
        // exp 2: in SGD epoch, run multiple train threads
        // generate worker_num and worker_num_type based on conf
        // add fgd conf
        worker_num.push_back(svrg_fgd_workers_per_process);
        worker_num_type.push_back("threads_per_worker");
        // add sgd conf
        for (int i = 0; i < sgd_continious_epoch * sgd_stage; i++) {
            // in epx2, in sgd epoch, need same num threads with fgd
            worker_num.push_back(svrg_fgd_workers_per_process);
            worker_num_type.push_back("threads_per_worker");
        }
        train_epoch = (1 + sgd_continious_epoch * sgd_stage) * num_sgd_continious_epoch;
    } else {
        // exp3: in SGD epoch, run multiple train threads, but these multiple threads are in one process, and their
        // worker_num_type is threads_per_cluster
        // generate worker_num and worker_num_type based on conf
        // add fgd conf
        worker_num.push_back(svrg_fgd_workers_per_process);
        worker_num_type.push_back("threads_per_worker");
        // add sgd conf
        for (int i = 0; i < sgd_continious_epoch * sgd_stage; i++) {
            // in epx3, in sgd epoch, need same num threads with fgd
            // get the config about how many threads will needed in the SGD
            worker_num.push_back(sgd_exp3_num_threads);
            worker_num_type.push_back("threads_per_cluster");
        }
        train_epoch = (1 + sgd_continious_epoch * sgd_stage) * num_sgd_continious_epoch;
    }

    auto train_task = TaskFactory::Get().CreateTask<ConfigurableWorkersTask>(train_epoch, num_train_workers);
    train_task.set_worker_num(worker_num);
    train_task.set_worker_num_type(*((const std::vector<std::string>*) (&worker_num_type)));

    std::map<std::string, std::string> hint_w = {
        {husky::constants::kType, husky::constants::kPS},
        {husky::constants::kConsistency, husky::constants::kSSP},
        {husky::constants::kStorageType, husky::constants::kVectorStorage},
        {husky::constants::kStaleness, "1"},
        {husky::constants::kNumWorkers, std::to_string(worker_num.at(1))}  // be careful about this settting
    };
    // when do FGD, total_threads = num_threads_per_worker * num_process
    int total_threads_bsp = worker_num.at(0) * Context::get_worker_info().get_num_processes();
    std::map<std::string, std::string> hint_u = {
        {husky::constants::kType, husky::constants::kPS},
        {husky::constants::kConsistency, husky::constants::kBSP},
        {husky::constants::kStorageType, husky::constants::kVectorStorage},
        {husky::constants::kNumWorkers, std::to_string(total_threads_bsp)}  // be careful about this setting
    };
    std::map<std::string, std::string> hint_w_old = {
        {husky::constants::kType, husky::constants::kPS},
        {husky::constants::kConsistency, husky::constants::kBSP},
        {husky::constants::kStorageType, husky::constants::kVectorStorage},
        {husky::constants::kNumWorkers, std::to_string(total_threads_bsp)},  // be careful about this setting
        {husky::constants::kUpdateType, husky::constants::kAssignUpdate}};

    int kv_w = kvstore::KVStore::Get().CreateKVStore<float>(hint_w, num_params);
    int kv_u = kvstore::KVStore::Get().CreateKVStore<float>(hint_u, num_params);
    // kv_w_old kvstore keeps the old_w to be resued in SGDs. after each FGD, kv_w_old will be updated to keep same with
    // kv_w.
    // for example, FGD SGD SGD SGD SGD
    int kv_w_old = kvstore::KVStore::Get().CreateKVStore<float>(hint_w_old, num_params);

    engine.AddTask(std::move(train_task), [&kv_w, &kv_u, &kv_w_old, &data_store, &alpha, num_params, data_size,
                                           &svrg_exp, &svrg_fgd_workers_per_process, &sgd_stage, &sgd_batch_size,
                                           &sgd_exp3_num_threads](const Info& info) {
        auto start_time = std::chrono::steady_clock::now();

        datastore::DataStoreWrapper<LabeledPointHObj<float, float, true>> data_store_wrapper(data_store);

        int current_epoch = info.get_current_epoch();
        // get configurable_task's worker_num vector, worker_num vector describes how many threads each epoch will use
        std::vector<int> worker_num = static_cast<const ConfigurableWorkersTask*>(info.get_task())->get_worker_num();

        ml::mlworker2::PSBspWorker<float> mlworker_u(info, *Context::get_zmq_context(), kv_u, num_params);
        ml::mlworker2::PSBspWorker<float> mlworker_w(info, *Context::get_zmq_context(), kv_w, num_params);
        ml::mlworker2::PSBspWorker<float> mlworker_w_old(info, *Context::get_zmq_context(), kv_w_old, num_params);

        std::string epoch_type;
        if (current_epoch % worker_num.size() == 0) {
            if (info.get_cluster_id() == 0) {
                husky::LOG_I << RED("FGD. current_epoch: " + std::to_string(current_epoch));
            }
            // do FGD
            epoch_type = "current_epoch: " + std::to_string(current_epoch) + " FGD";
            FGD_update(data_store, kv_w, kv_u, kv_w_old, alpha, num_params, info, data_size, mlworker_u, mlworker_w,
                       mlworker_w_old);
        } else {  // do SGD
            epoch_type = "current_epoch: " + std::to_string(current_epoch) + " SGD";
            if (data_store_wrapper.get_data_size() == 0) {
                return;  // return if there is no data
            } else {
                // alpha = 0.9 * alpha;
                // do SGD
                if (svrg_exp == 1) {
                    husky::LOG_I << RED("SGD exp1. current_epoch: " + std::to_string(current_epoch));
                    // exp1
                    int each_epoch_iter = sgd_batch_size * sgd_stage;
                    SGD_update(data_store, kv_w, kv_u, kv_w_old, alpha, num_params, each_epoch_iter, info, 1,
                               mlworker_u, mlworker_w, mlworker_w_old);
                } else if (svrg_exp == 2) {
                    if (info.get_cluster_id() == 0) {
                        husky::LOG_I << RED("SGD exp2. current_epoch: " + std::to_string(current_epoch));
                    }
                    // exp2
                    int total_threads = svrg_fgd_workers_per_process * info.get_worker_info().get_num_processes();
                    int each_epoch_iter = sgd_batch_size / total_threads;
                    SGD_update(data_store, kv_w, kv_u, kv_w_old, alpha, num_params, each_epoch_iter, info,
                               info.get_worker_info().get_num_processes() * svrg_fgd_workers_per_process, mlworker_u,
                               mlworker_w, mlworker_w_old);
                } else {
                    if (info.get_cluster_id() == 0) {
                        husky::LOG_I << RED("SGD exp3. current_epoch: " + std::to_string(current_epoch));
                    }
                    // exp3
                    int each_epoch_iter = sgd_batch_size / sgd_exp3_num_threads;
                    SGD_update(data_store, kv_w, kv_u, kv_w_old, alpha, num_params, each_epoch_iter, info,
                               sgd_exp3_num_threads, mlworker_u, mlworker_w, mlworker_w_old);
                }
            }
        }
        auto end_time = std::chrono::steady_clock::now();
        if (info.get_cluster_id() == 0) {
            auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            husky::LOG_I << YELLOW(epoch_type + " Train time: " + std::to_string(train_time) + " ms");
        }
    });

    // submit train task
    start_time = std::chrono::steady_clock::now();
    engine.Submit();
    end_time = std::chrono::steady_clock::now();
    if (Context::get_worker_info().get_process_id() == 0) {
        auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        husky::LOG_I << YELLOW("Train time: " + std::to_string(train_time) + " ms");
    }

    engine.Exit();
    kvstore::KVStore::Get().Stop();
}
