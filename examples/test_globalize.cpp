#include <random>

#include "husky/core/channel/migrate_channel.hpp"
#include "husky/core/hash_ring.hpp"
#include "husky/core/objlist.hpp"

#include "core/color.hpp"
#include "worker/engine.hpp"

using namespace husky;

class IntObject {
   public:
    using KeyT = int;
    int key;
    IntObject() { this->key = 0; }
    explicit IntObject(KeyT key) { this->key = key; }
    const int& id() const { return key; }
};

template <typename ObjT>
void globalize(ObjList<ObjT>& obj_list, const Info& info) {
    MigrateChannel<IntObject> migrate_channel(&obj_list, &obj_list);
    auto* mailbox = Context::get_mailbox(info.get_local_id());
    migrate_channel.setup(info.get_local_id(), info.get_global_id(), info.get_worker_info(), mailbox);

    for (auto& obj : obj_list.get_data()) {
        int dst_thread_id = info.get_worker_info().get_hash_ring().hash_lookup(obj.id());
        migrate_channel.migrate(obj, dst_thread_id);
    }

    obj_list.deletion_finalize();
    migrate_channel.flush();
    migrate_channel.prepare_immigrants();
}

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port"});
    if (!rt)
        return 1;

    auto& engine = Engine::Get();

    auto task = TaskFactory::Get().CreateTask<HuskyTask>(1, 20);
    engine.AddTask(task, [](const Info& info) {

        ObjList<IntObject> obj_list;
        for (int i = 0; i < 100; i++) {
            IntObject pobj(i);
            obj_list.add_object(pobj);
        }

        globalize(obj_list, info);
        auto& list_content = obj_list.get_data();
        husky::LOG_I << list_content.size();
    });
    engine.Submit();
    engine.Exit();
}
