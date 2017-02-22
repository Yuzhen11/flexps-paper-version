#pragma once

#include "hdfs/hdfs.h"

#include "husky/master/hdfs_assigner.hpp"
#include "husky/master/master.hpp"
#include "husky/core/context.hpp"

#include "core/constants.hpp"

namespace husky {

class HDFSBlockAssignerML : public HDFSBlockAssigner {
   public: 
    HDFSBlockAssignerML() {
        Master::get_instance().register_main_handler(constants::kIOHDFSSubsetLoad, std::bind(&HDFSBlockAssignerML::master_main_handler_ml, this));
    }

    void master_main_handler_ml() {
        auto& master = Master::get_instance();
        auto master_socket = master.get_socket();
        std::string url, host;
        int num_threads, id; 
        BinStream stream = zmq_recv_binstream(master_socket.get());
        stream >> url >> host >> num_threads >> id;
        
        // reset num_worker_alive
        num_workers_alive = num_threads;

        std::pair<std::string, size_t> ret = answer(host, url, id);
        stream.clear();
        stream << ret.first << ret.second;

        zmq_sendmore_string(master_socket.get(), master.get_cur_client());
        zmq_sendmore_dummy(master_socket.get());
        zmq_send_binstream(master_socket.get(), stream);
    }

    void browse_hdfs(int id, const std::string& url) {
        if (!fs_)
            return;

        int num_files;
        int dummy;
        hdfsFileInfo* file_info = hdfsListDirectory(fs_, url.c_str(), &num_files);
        auto& files_locality = files_locality_multi_dict[id][url];
        for (int i = 0; i < num_files; ++i) {
            // for every file in a directory
            if (file_info[i].mKind != kObjectKindFile)
                continue;
            size_t k = 0;
            while (k < file_info[i].mSize) {
                // for every block in a file
                auto blk_loc = hdfsGetFileBlockLocations(fs_, file_info[i].mName, k, 1, &dummy);
                for (int j = 0; j < blk_loc->numOfNodes; ++j) {
                    // for every replication in a block
                    files_locality.insert(
                        BlkDesc{std::string(file_info[i].mName) + '\0', k, std::string(blk_loc->hosts[j])});
                }
                k += file_info[i].mBlockSize;
            }
        }
        hdfsFreeFileInfo(file_info, num_files);
    }

    std::pair<std::string, size_t> answer(const std::string& host, const std::string& url, int id) {
        if (!fs_)
            return {"", 0};

        // cannot find id
        if (files_locality_multi_dict.find(id) == files_locality_multi_dict.end()) {
            browse_hdfs(id, url);
            finish_multi_dict[id][url] = 0;
        }

        // cannot find url
        if (files_locality_multi_dict[id].find(url) == files_locality_multi_dict[id].end()) {
            browse_hdfs(id, url);
            finish_multi_dict[id][url] = 0;
        }
         
        // empty url
        auto& files_locality = files_locality_multi_dict[id][url];
        if (files_locality.size() == 0) {
            finish_multi_dict[id][url] += 1;
            if (finish_multi_dict[id][url] == num_workers_alive)
                files_locality_multi_dict[id].erase(url);
            return {"", 0};
        }
        
        // selected_file, offset
        std::pair<std::string, size_t> ret = {"", 0};  
        for (auto& triplet : files_locality) {
            if (triplet.block_location == host) {
                ret = {triplet.filename, triplet.offset};
                break;
            }
        }

        if (ret.first.empty()) {
            ret = {files_locality.begin()->filename, files_locality.begin()->offset};
        }

        // remove
        for (auto it = files_locality.begin(); it != files_locality.end();) {
            if (it->filename == ret.first && it->offset == ret.second)
                it = files_locality.erase(it);
            else
                it++;
        }

        return ret;
    }

   private:
    std::map<int, std::map<std::string, std::unordered_set<BlkDesc>>> files_locality_multi_dict;
    std::map<int, std::map<std::string, int>> finish_multi_dict;

};

}
