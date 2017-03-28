#include "lib/sample_reader_parse.hpp"
#include "io/input/line_inputformat_ml.hpp"
#include "io/input/binary_inputformat_ml.hpp"

#include "worker/engine.hpp"

using namespace husky;
using husky::lib::ml::LabeledPointHObj;

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
    LIBSVMAsyncReadBinaryParseBuffer<LabeledPointHObj<float, float, true>, io::BinaryInputFormatML> buffer;
    engine.AddTask(std::move(task), [&buffer, batch_size, batch_num, num_features](const Info& info) {
        buffer.init(Context::get_param("input"), info.get_task_id(), 4, batch_size, batch_num, num_features);  // start buffer

        // create a reader
        std::unique_ptr<SimpleSampleReader<LabeledPointHObj<float,float,true>, io::BinaryInputFormatML>> reader(new SimpleSampleReader<LabeledPointHObj<float, float, true>, io::BinaryInputFormatML>(&buffer));
        int count = 0;
        while (!reader->is_empty()) {
            // parse a batch of data and return parameters for the batch
            auto keys = reader->prepare_next_batch();
            // get the batch of samples in vector
            auto data = reader->get_data();
            count += data.size();
        }
        husky::LOG_I << "task0 read " << count << " records in total.";
    });

    engine.Submit();
    engine.Exit();
}
