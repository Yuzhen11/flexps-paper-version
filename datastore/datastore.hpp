#include <cassert>
#include <vector>

namespace datastore {

/*
 * DataStore manages the training data
 */
template<typename DataType>
class DataStore {
public:
    DataStore() = default;
    DataStore(int num_local_workers) : data_(num_local_workers, nullptr) {
        for (auto& p : data_) {
            p = new std::vector<DataType>();
        }
    }
    ~DataStore() {
        for (auto p : data_) {
            delete p;
        }
    }

    /*
     * Push new data into local storage
     *
     * Cautions: Not thread-safe, suggested to push to my own id
     */
    void Push(int local_id, const DataType& data) {
        assert(data_[local_id] != nullptr);
        data_[local_id]->push_back(data);
    }

    void Push(int local_id, DataType&& data) {
        assert(data_[local_id] != nullptr);
        data_[local_id]->push_back(std::move(data));
    }

    /*
     * Pull the local storage
     *
     * Cautions: Not thread-safe 
     *
     * For read-only tasks, fine to read from all the DataStore
     */
    std::vector<DataType>& Pull(int local_id) {
        assert(data_[local_id] != nullptr);
        return *data_[local_id];
    }

private:
    DataStore(const DataStore&) = delete;
    DataStore& operator=(const DataStore&) = delete;

private:
    std::vector<std::vector<DataType>*> data_;

};

}  // namespace datastore
