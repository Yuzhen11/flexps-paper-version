#pragma once

#include "datastore/datastore.hpp"
#include "datastore/batch_data_sampler.hpp"
#include "datastore/datastore_utils.hpp"
#include "kvstore/kvstore.hpp"
#include "lib/load_data.hpp"
#include "kvstore/kvstore.hpp"
#include "ml/model/model.hpp"
#include "ml/shared/shared_state.hpp"
#include "ml/mlworker/mlworker.hpp"

using namespace husky;
using husky::lib::ml::LabeledPointHObj;

/*
 */
namespace ml {
namespace mlworker2 {

template<typename Val>
class PSBspModel {
   public:
    PSBspModel(int model_id, int num_params, int num_local_threads) :
        model_id_(model_id), num_params_(num_params), num_local_threads_(num_local_threads),
        params_(num_params), process_cache_keys_(num_params) {}

    void Push(const std::vector<size_t>& keys, const std::vector<Val>& vals, int local_id, bool is_leader, bool enable_cc = true) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            for (size_t i = 0; i < keys.size(); ++ i) {
                params_[keys[i]] += vals[i];
            }
        }
        {
            std::unique_lock<std::mutex> lock(push_mtx_);
            push_num_ += 1;
            if (push_num_ == num_local_threads_) {
                push_cv_.notify_all();
            } else {
                // block until push_num_ == num_local_threads_
                push_cv_.wait(lock, [this]() {
                    return push_num_ == num_local_threads_;
                });
            }
        }
        {
            std::unique_lock<std::mutex> lock(push_mtx_);
            push_num2_ += 1;
            if (push_num2_ == num_local_threads_) {
                push_num_ = 0;
                push_num2_ = 0;
            }
        }
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id);
        int ts;
        if (is_leader == true) {
            // leader sends out all keys
            std::vector<size_t> push_keys;
            std::vector<Val> push_vals;
            for (int i = 0; i < params_.size(); ++ i) {
                if (params_[i] != 0) {
                    push_keys.push_back(i);
                    push_vals.push_back(params_[i]);
                }
            }
            ts = kvworker->Push(model_id_, push_keys, push_vals, true, true, enable_cc);
            std::fill(params_.begin(), params_.end(), 0);
        } else {
            std::vector<Val> tmp;
            ts = kvworker->Push(model_id_, {}, tmp, true, true, enable_cc);
        }
        kvworker->Wait(model_id_, ts);
    }
    void Pull(const std::vector<husky::constants::Key>& keys, std::vector<Val>* vals, int local_id, bool enable_cc = true) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            for (auto k : keys) {
                process_cache_keys_[k] = 1;
            }
        }
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id);
        {
            std::unique_lock<std::mutex> lock(pull_mtx_);
            pull_num_ += 1;
            if (pull_num_ == num_local_threads_) {
                pull_keys_.clear();
                pull_vals_.clear();
                // pull !
                for (int i = 0; i < process_cache_keys_.size(); ++ i) {
                    if (process_cache_keys_[i] == 1) {
                        pull_keys_.push_back(i);
                    }
                }
                kvworker->Wait(model_id_, kvworker->Pull(model_id_, pull_keys_, &pull_vals_, true, true, enable_cc));
                pull_cv_.notify_all();
            } else {
                std::vector<Val> tmp;
                kvworker->Pull(model_id_, {}, &tmp, true, true, enable_cc);
                pull_cv_.wait(lock, [this]() {
                    return pull_num_ == num_local_threads_;
                });
            }
        }
        vals->clear();
        int i = 0;
        for (auto k : keys) {
            while (i != pull_keys_.size() && pull_keys_[i] != k) i += 1;
            assert(i != pull_keys_.size());
            vals->push_back(pull_vals_[i]);
        }
        assert(keys.size() == vals->size());
        {
            std::lock_guard<std::mutex> lock(pull_mtx_);
            pull_num2_ += 1;
            if (pull_num2_ == num_local_threads_) {  // reset
                pull_num_ = 0;
                pull_num2_ = 0;
                std::fill(process_cache_keys_.begin(), process_cache_keys_.end(), 0);
            }
        }
    }
   private:
    int num_local_threads_;
    int model_id_;
    int num_params_;

    std::mutex mtx_;
    std::vector<Val> params_;
    std::vector<int> process_cache_keys_;
    std::vector<size_t> pull_keys_;
    std::vector<Val> pull_vals_;

    int push_num_ = 0;
    int pull_num_ = 0;
    int push_num2_ = 0;
    int pull_num2_ = 0;
    std::mutex push_mtx_;
    std::mutex pull_mtx_;
    std::condition_variable push_cv_;
    std::condition_variable pull_cv_;
};

/*
 * PSBspWorker
 * Provide simple process-level cache for PSWorker in BSP mode
 * Only vector-version shared state is provided
 */
template<typename Val>
class PSBspWorker {
    struct PSState {
        PSState(int model_id, int num_params, int num_workers):
        model_(model_id, num_params, num_workers) {}
        PSBspModel<Val> model_;
    };
   public:
    PSBspWorker() = delete;
    PSBspWorker(const PSBspWorker&) = delete;
    PSBspWorker& operator=(const PSBspWorker&) = delete;
    PSBspWorker(PSBspWorker&&) = delete;
    PSBspWorker& operator=(PSBspWorker&&) = delete;

    PSBspWorker(const husky::Info& info, zmq::context_t& context, int kv_store, int num_params)
        : shared_state_(info.get_task_id() + kv_store, info.is_leader(), info.get_num_local_workers(), context),
        info_(info),
        model_id_(kv_store) {

        if (info.is_leader()) {
            PSState* state = new PSState(model_id_, num_params, info.get_num_local_workers());
            // 1. Init
            shared_state_.Init(state);
        }
        // 2. Sync
        shared_state_.SyncState();
        // set kvworker
        local_id_ = info.get_local_id();
        kvworker_ = kvstore::KVStore::Get().get_kvworker(local_id_);
        kvworker_->Wait(model_id_, kvworker_->InitForConsistencyControl(model_id_, info.get_num_workers()));
    }
    ~PSBspWorker() {
        shared_state_.Barrier();
        if (info_.get_local_tids().at(0) == info_.get_global_id()) {
            delete shared_state_.Get();
        }
    }

    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<Val>& vals) {
        assert(push_count_ + 1 == pull_count_);
        push_count_ += 1;
        shared_state_.Get()->model_.Push(keys, vals, local_id_, info_.is_leader());
    }
    
    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<Val>* vals) {
        assert(push_count_ == pull_count_);
        pull_count_ += 1;
        shared_state_.Get()->model_.Pull(keys, vals, local_id_); 
    }

    kvstore::KVWorker* GetKVWorker() {
        return kvworker_;
    }
   private:
    int model_id_;
    kvstore::KVWorker* kvworker_ = nullptr;
    int local_id_;
    // Shared Model
    SharedState<PSState> shared_state_;
    const husky::Info& info_;

    // Just to restrict the usage of the Push/Pull APIs,
    // The correct usage should be Pull, Push, Pull, Push...
    int push_count_ = 0;
    int pull_count_ = 0;
    int ts_ = -1;
    bool send_all_ = true;
};

}
}

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


// Helper functions for SVRG
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
        const std::vector<husky::constants::Key>& batch_keys,
        const std::vector<husky::constants::Key>& all_keys,
        const std::vector<float>& w,
        const std::vector<float>& w0, std::vector<float>* w_updates) {

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

void init_ptrs(std::vector<std::vector<float>>& params, std::vector<std::vector<float>*>& ptrs) {
    ptrs.resize(params.size());
    for (int i=0; i<params.size();i++){
        ptrs[i] = &params[i];
    }
}

void flattenTo(const std::vector<std::vector<float>>& left, std::vector<float>& right) {
    for (int i=0; i<left.size(); i++) {
        right.insert(right.end(), left[i].begin(), left[i].end());
    }
} 

void compressTo(const std::vector<float>& left, std::vector<std::vector<float>>& right, int chunk_size) {
    right.resize((left.size() + chunk_size - 1)/chunk_size);
    for (int i=0; i< left.size(); i++) {
        int r = i/chunk_size;
        int c = i%chunk_size;
        right[r].resize(chunk_size);
        right[r][c] = left[i];
    }
} 

// TODO: consider use Pull({}, {}) to optimize the dummy Pull/Push
void PullDummyChunk(std::unique_ptr<ml::mlworker::GenericMLWorker<float>>& mlworker, int chunk_size) {
    // dummy pull to ensure clock
    std::vector<std::vector<float>> params(1);
    std::vector<std::vector<float>*> ptrs; 
    init_ptrs(params, ptrs);
    mlworker->PullChunks({0}, ptrs);
}

void PushDummyChunk(std::unique_ptr<ml::mlworker::GenericMLWorker<float>>& mlworker, int chunk_size) {
    // dummy push to ensure clock
    std::vector<float> empty(chunk_size);
    std::vector<std::vector<float>> params;
    compressTo(empty, params, chunk_size);
    std::vector<std::vector<float>*> ptrs;
    init_ptrs(params, ptrs);
    mlworker->PushChunks({0}, ptrs);
}

void PullChunksAndFlatten(std::unique_ptr<ml::mlworker::GenericMLWorker<float>>& mlworker,
        const std::vector<size_t>& chunk_ids, int chunk_size, std::vector<float>& rets) {

    if (chunk_ids.empty()) {
        PullDummyChunk(mlworker, chunk_size);
        return;
    }
    std::vector<std::vector<float>> params(chunk_ids.size());
    std::vector<std::vector<float>*> ptrs; 
    init_ptrs(params, ptrs);
    mlworker->PullChunks(chunk_ids, ptrs);
    flattenTo(params, rets);
}

void CompressAndPushChunks(std::unique_ptr<ml::mlworker::GenericMLWorker<float>>& mlworker,
        const std::vector<size_t>& chunk_ids, int chunk_size, const std::vector<float>& delta) {
    
    if (chunk_ids.empty() || delta.empty()) {
        PushDummyChunk(mlworker, chunk_size);
        return;
    } 
    std::vector<std::vector<float>> params;
    compressTo(delta, params, chunk_size);
    std::vector<std::vector<float>*> ptrs;
    init_ptrs(params, ptrs);
    mlworker->PushChunks(chunk_ids, ptrs);
}
