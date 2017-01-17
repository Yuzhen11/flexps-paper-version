#include <random>

#include "husky/core/channel/channel_store.hpp"
#include "husky/core/channel/migrate_channel.hpp"
#include "husky/core/objlist.hpp"
#include "husky/core/worker_info.hpp"

#include "worker/engine.hpp" 
#include "core/color.hpp"

using namespace husky;

class Object {
   public:
    using KeyT = int;
    int key;
    Object(){this->key = 0;}
    explicit Object(KeyT key) { this->key = key; }
    const int& id() const { return key; }
};

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port"});
    if (!rt)
        return 1;

    auto& engine = Engine::Get();

    auto task = TaskFactory::Get().CreateTask<HuskyTask>(1, 4);
    engine.AddTask(task, [](const Info& info) {
        ObjList<Object> src_list;
        ObjList<Object> dst_list;
        MigrateChannel<Object> migrate_channel(&src_list, &dst_list);
        auto* mailbox = Context::get_mailbox(info.get_local_id());
        migrate_channel.setup(info.get_local_id(), info.get_global_id(), info.get_worker_info(), mailbox);

        Object pobj(1);
        src_list.add_object(pobj);
        if (info.get_cluster_id() == 1) {
            // migrate one obj to thread with cluster id 0
            int dst_global_id = info.get_cluster_global().at(0);
            husky::LOG_I<<"dst global id is:" << dst_global_id;
            Object* p = src_list.find(1);
            husky::LOG_I<<"object id stored on 1 is :"<<(*p).id();
            migrate_channel.migrate(*p, dst_global_id);
            src_list.deletion_finalize();
        }

        migrate_channel.flush();
        migrate_channel.prepare_immigrants();

        if (info.get_cluster_id() == 1) {
            if (src_list.find(1) == nullptr)
                husky::LOG_I << "This is cluster id 1." << " Object already disapear";
        }
       
        if (dst_list.get_data().size() != 0){
            Object& obj = dst_list.get_data()[0];
            husky::LOG_I << "This is cluster id "<<info.get_cluster_id() << " Received object id:" << obj.id();
        }

    });
    engine.Submit();
    engine.Exit();
}
