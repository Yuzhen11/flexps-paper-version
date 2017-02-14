#include "examples/sample_reader.hpp"

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

    auto task = TaskFactory::Get().CreateTask<HuskyTask>(1, 4); // 1 epoch, 4 workers
    engine.AddTask(std::move(task), [&batch_size, &batch_num, &num_features](const Info& info) {
        // load
        auto buffer = new TextBuffer(Context::get_param("input"), batch_size, batch_num);
        /*
        std::vector<boost::string_ref>* batch = nullptr;
        int count = 0;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        while (buffer->get_batch(batch)) {
            count += batch->size();

            // std::this_thread::sleep_for(std::chrono::milliseconds(5));
            husky::LOG_I << buffer->ask() << " unread batches in buffer";
        }
        husky::LOG_I << "loaded " << count << " records.";
        */

        auto reader = LIBSVMSampleReader<float, float, true>(batch_size, num_features, buffer);
        auto keys = reader.prepare_next_batch();
        auto data = reader.get_data_ptrs();
        delete buffer;
    });
    engine.Submit();
    engine.Exit();
}
