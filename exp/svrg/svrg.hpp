#pragma once

#include <algorithm>
#include <cmath>
#include <vector>
#include <chrono>

#include "core/table_info.hpp"
#include "datastore/datastore.hpp"
#include "datastore/batch_data_sampler.hpp"
#include "datastore/datastore_utils.hpp"
#include "kvstore/kvstore.hpp"
#include "lib/load_data.hpp"
#include "ml/mlworker/mlworker.hpp"
#include "ml/ml.hpp"

#include "exp/svrg/svrg_helper.hpp"



void FGD_update_chunk(datastore::DataStore<LabeledPointHObj<float, float, true>>& data_store, 
                float alpha, int num_params, const Info& info, size_t data_size,
                const TableInfo& table_w, const TableInfo& table_u, int chunk_size) {
    auto mlworker_w = ml::CreateMLWorker<float>(info, table_w);
    auto mlworker_u = ml::CreateMLWorker<float>(info, table_u);

    int num_chunks = (num_params + chunk_size -1) / chunk_size;
    std::vector<husky::constants::Key> chunk_ids(num_chunks);
    std::iota(chunk_ids.begin(), chunk_ids.end(), 0);
    std::vector<float> rets_u;
    PullChunksAndFlatten(mlworker_u, chunk_ids, chunk_size, rets_u);
    std::vector<float> rets_w;
    PullChunksAndFlatten(mlworker_w, chunk_ids, chunk_size, rets_w);


    // 1. compute clear_delta to clear kv_u before each FGD to avoid u accumulating
    std::vector<float> clear_delta(rets_u.size());
    for (int i = 0; i < rets_u.size(); i++) {
        // each thread should push a part value(-1 * old_value / num_threads) to reset kvstore
        clear_delta[i] = -1 * (rets_u[i] / info.get_num_workers());
    }

    // 2. compute the avg gradient using rest_w 
    // position is used to balance the data load
    // when do FGD, we hope data can be loaded balancely
    int position;
    std::vector<int> tids = info.get_worker_info().get_local_tids();
    for (int k = 0; k < tids.size(); k++) {
        if (tids[k] == info.get_global_id()) {
            position = k;
            break;
        }
    }
    datastore::DataLoadBalance<LabeledPointHObj<float, float, true>> data_load_balance(
        data_store, info.get_num_workers(), position);

    std::vector<float> delta(num_params);
    // go through all data
    while (data_load_balance.has_next()) {
        auto& data = data_load_balance.next();
        auto& x = data.x;
        float y = data.y;
        float pred_y = 0.0;
        if (y < 0) y = 0;
        for (auto field : x) {
            pred_y += rets_w[field.fea] * field.val;
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
    
    CompressAndPushChunks(mlworker_u, chunk_ids, chunk_size, delta);
    // an empty push to ensure consitency
    CompressAndPushChunks(mlworker_w, chunk_ids, chunk_size, {});
}


void FGD_update(datastore::DataStore<LabeledPointHObj<float, float, true>>& data_store, int kv_w, int kv_u,
                float alpha, int num_params, const Info& info, size_t data_size) {

     ml::mlworker2::PSBspWorker<float> mlworker_u(info, *Context::get_zmq_context(), kv_u, num_params);
     ml::mlworker2::PSBspWorker<float> mlworker_w(info, *Context::get_zmq_context(), kv_w, num_params);
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
//
void SGD_update_chunk(datastore::DataStore<LabeledPointHObj<float, float, true>>& data_store,
                float alpha, int num_params, int inner_loop, int batch_size_per_worker, const Info& info,
                const TableInfo& table_w, const TableInfo& table_u, int chunk_size) {
    auto mlworker_w = ml::CreateMLWorker<float>(info, table_w);
    auto mlworker_u = ml::CreateMLWorker<float>(info, table_u);

    int num_chunks = (num_params + chunk_size -1) / chunk_size;
    std::vector<husky::constants::Key> chunk_ids(num_chunks);
    std::iota(chunk_ids.begin(), chunk_ids.end(), 0);

    std::vector<husky::constants::Key> all_keys(num_params);
    std::iota(all_keys.begin(), all_keys.end(), 0);

    datastore::BatchDataSampler<LabeledPointHObj<float, float, true>> mini_batch(data_store, batch_size_per_worker);
    std::vector<std::pair<int, float>> my_part_u;

    std::vector<float> w0;
    PullChunksAndFlatten(mlworker_w, chunk_ids, chunk_size, w0);
    PushDummyChunk(mlworker_w, chunk_size);

    auto start_time = std::chrono::steady_clock::now();
    mini_batch.random_start_point();
    // every inner loop contains one mini batch
    auto start_time_inner = std::chrono::steady_clock::now();
    for (int m=0; m<inner_loop; m++) {
        if (m == 1) {
            start_time_inner = std::chrono::steady_clock::now();
        }
        mini_batch.prepare_next_batch_data();
        std::vector<float> w;
        // mlworker_w.Pull(batch_keys, &w);
        // mlworker_w.Push(batch_keys, w);
        auto start_time_a = std::chrono::steady_clock::now();
        PullChunksAndFlatten(mlworker_w, chunk_ids, chunk_size, w);

        auto start_time_c = std::chrono::steady_clock::now();
        
        // gra(w0) - gra(wm)
        std::vector<float> w_updates0;
        get_gradient_diff(mini_batch.get_data_ptrs(), all_keys, all_keys, w, w0, &w_updates0);
        
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
        CompressAndPushChunks(mlworker_w, chunk_ids, chunk_size, w_updates0);
        auto start_time_e = std::chrono::steady_clock::now();
        if (info.get_cluster_id() == 0&& (m+1)%1000 == 0) {
            auto total = (std::chrono::duration_cast<std::chrono::milliseconds>(start_time_e-start_time_a)).count();
        }
	    //LOG_I<< "computation time:" << (std::chrono::duration_cast<std::chrono::milliseconds>(start_time_d - start_time_c)).count()<<"";
    }
    auto end_time_inner = std::chrono::steady_clock::now();
    auto iter_time = (std::chrono::duration_cast<std::chrono::milliseconds>(end_time_inner-start_time_inner)).count();
	LOG_I<< "avg"<<info.get_current_epoch()<<" time:"
    << iter_time/float(inner_loop-1) <<"ms";
    
}

void SGD_update(datastore::DataStore<LabeledPointHObj<float, float, true>>& data_store, int kv_w, int kv_u,
                float alpha, int num_params, int inner_loop, int batch_size_per_worker, const Info& info) {

    ml::mlworker2::PSBspWorker<float> mlworker_u(info, *Context::get_zmq_context(), kv_u, num_params);
    ml::mlworker2::PSBspWorker<float> mlworker_w(info, *Context::get_zmq_context(), kv_w, num_params);
    
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
        get_gradient_diff(mini_batch.get_data_ptrs(), batch_keys, all_keys, w, w0, &w_updates0);
        
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
        if (info.get_cluster_id() == 0&& (m+1)%1000 == 0) {
            auto total = (std::chrono::duration_cast<std::chrono::milliseconds>(start_time_e-start_time_a)).count();
	        LOG_I<< m <<" batchkeys size:" <<batch_keys.size()<< " total time:" << total;
        }
	    //LOG_I<< "computation time:" << (std::chrono::duration_cast<std::chrono::milliseconds>(start_time_d - start_time_c)).count()<<"";
    }
    auto end_time_inner = std::chrono::steady_clock::now();
    auto iter_time = (std::chrono::duration_cast<std::chrono::milliseconds>(end_time_inner-start_time_inner)).count();
	LOG_I<< "avg"<<info.get_current_epoch()<<" time:"
    << iter_time/float(inner_loop-1) <<"ms";
}
