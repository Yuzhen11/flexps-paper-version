#include "lib/sample_reader.hpp"

#include "worker/engine.hpp"

using namespace husky;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {
        "worker_port", "cluster_manager_host", "cluster_manager_port", "input", "hdfs_namenode", "hdfs_namenode_port", "batch_size", "batch_num", "num_features"
    });
    if (!rt)
        return 1;
    
    // batch_size=1000
    int batch_size = std::stoi(Context::get_param("batch_size"));
    // batch num=100   
    int batch_num = std::stoi(Context::get_param("batch_num"));
    int num_features = std::stoi(Context::get_param("num_features"));

    auto& engine = Engine::Get();

    // create the buffer for loading data from hdfs (a shared buffer in a process)
    auto task = TaskFactory::Get().CreateTask<HuskyTask>(2, 2); // 2 epoch, 2 workers
    AsyncReadBuffer buffer;
    // input=hdfs:///datasets/classification/a9
    engine.AddTask(std::move(task), [&buffer, batch_size, batch_num, num_features](const Info& info) {
        buffer.init(Context::get_param("input"), info.get_task_id(), 4, batch_size, batch_num);  // start buffer
        /* load with AsyncReadBuffer
        std::vector<boost::string_ref> batch;
        int count = 0;
        while (buffer->get_batch(batch)) {
            count += batch.size();
            husky::LOG_I << buffer->ask() << " unread batches in buffer";
        }
        husky::LOG_I << "loaded " << count << " records.";
        */

        // read samples in libsvm format
        LIBSVMSampleReader<float, float, true> reader(batch_size, num_features, &buffer);
        int count = 0;
        while (!reader.is_empty()) {
            // parse a batch of data and return parameters for the batch
            auto keys = reader.prepare_next_batch();
            // get the batch of samples in vector
            auto data = reader.get_data();
            count += data.size();
        }
        husky::LOG_I << "task0 read " << count << " records in total.";
    });
    
    // create the buffer for loading data from hdfs (a shared buffer in a process)
    auto task1 = TaskFactory::Get().CreateTask<HuskyTask>(1, 3); // 1 epoch, 3 workers
    AsyncReadBuffer buffer1;
    engine.AddTask(std::move(task1), [&buffer1, batch_size, batch_num, num_features](const Info& info) {
        buffer1.init(Context::get_param("input"), info.get_task_id(), 4, batch_size, batch_num);  // start buffer
        // read samples in libsvm format
        LIBSVMSampleReader<float, float, true> reader(batch_size, num_features, &buffer1);
        int count = 0;
        while (!reader.is_empty()) {
            // parse a batch of data and return parameters for the batch
            auto keys = reader.prepare_next_batch();
            // get the batch of samples in vector
            auto data = reader.get_data();
            count += data.size();
        }
        husky::LOG_I << "task1 read " << count << " records in total.";
    });
    
    engine.Submit();
    engine.Exit();
}
