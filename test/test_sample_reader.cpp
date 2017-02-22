#include "lib/sample_reader.hpp"

#include "worker/engine.hpp"

using namespace husky;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {
        "worker_port", "cluster_manager_host", "cluster_manager_port", "input", "hdfs_namenode", "hdfs_namenode_port", "batch_size", "batch_num", "num_features"
    });
    if (!rt)
        return 1;

    int batch_size = std::stoi(Context::get_param("batch_size"));
    int batch_num = std::stoi(Context::get_param("batch_num"));
    int num_features = std::stoi(Context::get_param("num_features"));

    auto& engine = Engine::Get();

    // create the buffer for loading data from hdfs (a shared buffer in a process)
    auto buffer = new AsyncReadBuffer(Context::get_param("input"), batch_size, batch_num, true);
    auto task = TaskFactory::Get().CreateTask<HuskyTask>(1, 4); // 1 epoch, 4 workers
    engine.AddTask(std::move(task), [buffer, &batch_size, &batch_num, &num_features](const Info& info) {
        buffer->init();  // start buffer

        /* load with AsyncReadBuffer
        std::vector<boost::string_ref> batch;
        int count = 0;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        while (buffer->get_batch(batch)) {
            count += batch.size();
            husky::LOG_I << buffer->ask() << " unread batches in buffer";
        }
        husky::LOG_I << "loaded " << count << " records.";
        */

        // read samples in libsvm format
        auto reader = LIBSVMSampleReader<float, float, true>(batch_size, num_features, buffer);
        int count = 0;
        while (!reader.is_empty()) {
            // parse a batch of data and return parameters for the batch
            auto keys = reader.prepare_next_batch();
            // get the batch of samples in vector
            auto data = reader.get_data();
            count += data.size();
        }
        husky::LOG_I << "read " << count << " records in total.";
    });
    engine.Submit();
    engine.Exit();
    delete buffer;
}
