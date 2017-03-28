// Copyright 2016 Husky Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "hdfs/hdfs.h"

#include <regex>
#include <string>
#include <unordered_map>
#include <vector>
#include <sstream>

#include "husky/base/serialization.hpp"
#include "husky/core/constants.hpp"
#include "husky/core/context.hpp"

#include "husky/master/hdfs_binary_assigner.hpp"
#include "husky/master/master.hpp"

#include "core/constants.hpp"

namespace husky {

class HDFSFileAssignerML : public HDFSFileAssigner {
   public:
    HDFSFileAssignerML() {
        Master::get_instance().register_main_handler(constants::kIOHDFSBinarySubsetLoad, std::bind(&HDFSFileAssignerML::response, this));
    }

    void response() {
        auto& master = Master::get_instance();
        auto socket = master.get_socket();
        base::BinStream request = zmq_recv_binstream(socket.get());
        std::string host = base::deser<std::string>(request);
        std::string fileurl = base::deser<std::string>(request);
        int id = base::deser<int>(request);
        int num_threads = base::deser<int>(request);
        std::string load_type = base::deser<std::string>(request);
        
        std::string filename = answer(host, fileurl, id, num_threads, load_type);
        base::BinStream response;
        response << filename;
        zmq_sendmore_string(socket.get(), master.get_cur_client());
        zmq_sendmore_dummy(socket.get());
        zmq_send_binstream(socket.get(), response);

        if (filename.empty())
            filename = "None";
        LOG_I << host << " => " << fileurl << "@" << filename;
    }

   private:
    std::string answer(const std::string& host, const std::string& fileurl, int id, int num_threads, const std::string& load_type) {
        if (file_infos_.find(id) == file_infos_.end()) {  // set finished_info, if user want to load again, different id is needed
            finish_multi_dict[id][fileurl].second = 0;
            prepare(host, fileurl, id);
        }
        // print_file_infos();
            
        std::unordered_map<std::string, std::unordered_map<std::string, HDFSFileInfo>>& all_info = file_infos_[id];

        std::string selected_file;
        std::string selected_info_base; 
        // default is load data globally
        if (load_type.empty() || load_type == husky::constants::kLoadHdfsGlobally) { 
            HDFSFileInfo& info_local = all_info[host][fileurl];
            std::vector<std::string>& files = info_local.files_to_assign;

            // if local is not empty
            if(!files.empty()) {
                // choose a file from local
                selected_file = files.back();
                selected_info_base = info_local.base;

                // pop the selected file from this host files
                files.pop_back();
            } else {
                // select from the unlocal
                bool is_finish = true;
                for(auto its = all_info.begin(); its != all_info.end(); its++) {  // for every host
                    for(auto it = its->second.begin(); it != its->second.end(); it++) {   // for every file in the host
                        if (it->first == fileurl && it->second.files_to_assign.size() > 0) {  // select it if there is any
                            is_finish = false;
                            selected_file = it->second.files_to_assign.back();
                            selected_info_base = it->second.base;
                            // pop the selected file from this host files
                            it->second.files_to_assign.pop_back();
                            break;
                        } 
                    }
                    if (!is_finish) {
                        break;
                    }
                }

                if (is_finish) {
                    finish_multi_dict[id][fileurl].second += 1;
                    if(finish_multi_dict[id][fileurl].second == num_threads) {
                    }
                    return ""; // no file to assign
                }
            }
        } else if (load_type == husky::constants::kLoadHdfsLocally) {  // load data locally
            HDFSFileInfo& info_local = all_info[host][fileurl];
            std::vector<std::string>& files = info_local.files_to_assign;

            if (files.empty()) {     // local data is empty
                (finish_multi_dict[id][fileurl].first)[host] += 1;
                if ((finish_multi_dict[id][fileurl].first)[host] == num_threads) {
                }
                return "";
            }
            // choose a file
            selected_file = files.back();
            selected_info_base = info_local.base;

            // pop the selected file from this host files
            files.pop_back();
        } else {
            throw base::HuskyException("[hdfs_binary_assigner_ml] kLoadHdfsType error.");
        }

        // here should pop_back all replicas
        // host fileurl []
        for (auto its = all_info.begin(); its != all_info.end(); its++) {
            for (auto it = its->second.begin(); it != its->second.end(); it++) {
                for (auto t = it->second.files_to_assign.begin(); t != it->second.files_to_assign.end(); t++) {
                    if (*t == selected_file) {
                        all_info[its->first][it->first].files_to_assign.erase(t);
                        break;
                    }
                }
            }   
        }

        // print_file_infos();
        return selected_info_base + selected_file;
    }

    void prepare(const std::string& host, const std::string& url, int id) {
        size_t first_colon = url.find(":");
        first_colon = first_colon == std::string::npos ? url.length() : first_colon;
        std::string base = url.substr(0, first_colon);
        std::string filter = first_colon >= url.length() - 1 ? ".*" : url.substr(first_colon + 1, url.length());

        struct hdfsBuilder* builder = hdfsNewBuilder();
        hdfsBuilderSetNameNode(builder, Context::get_param("hdfs_namenode").c_str());
        int port;
        try {
            port = std::stoi(Context::get_param("hdfs_namenode_port").c_str());
        } catch (...) {
            ASSERT_MSG(false, ("Failed to parse hdfs namenode port: " + Context::get_param("hdfs_namenode_port")).c_str());
        }
        hdfsBuilderSetNameNodePort(builder, port);
        hdfsFS fs = hdfsBuilderConnect(builder);
        hdfsFreeBuilder(builder);
        hdfsFileInfo* base_info = hdfsGetPathInfo(fs, base.c_str());

        ASSERT_MSG(base_info != nullptr, ("Given path not found on HDFS: " + base).c_str());

            std::unordered_map<std::string, std::unordered_map<std::string, HDFSFileInfo>>& files_info = file_infos_[id];
        if (base_info->mKind == kObjectKindDirectory) {
            int base_len = base.length();
            try {
                std::regex filter_regex(filter);
                std::function<void(const char*)> recursive_hdfs_visit = nullptr;
                recursive_hdfs_visit = [this, id, &fs, &filter, &files_info, &filter_regex, base_len,
                                        &recursive_hdfs_visit, base_info, &url](const char* base) {
                    int num_entries;
                    hdfsFileInfo* infos = hdfsListDirectory(fs, base, &num_entries);
                    for (int i = 0; i < num_entries; ++i) {
                        if (infos[i].mKind == kObjectKindFile) {
                            std::string filename = std::string(infos[i].mName).substr(base_len);
                            size_t k = 0;
                            int dummy;
                            auto blk_loc = hdfsGetFileBlockLocations(fs, infos[i].mName, k, 1, &dummy);
                            for (int j = 0; j < blk_loc->numOfNodes; ++j) {
                                std::string host = blk_loc->hosts[j];
                                files_info[host][url].files_to_assign.push_back(filename);
                                (finish_multi_dict[id][url].first)[host] = 0;

                                // store files_info base
                                files_info[host][url].base = std::string(base_info->mName);
                                if (files_info[host][url].base.back() != '/')
                                    files_info[host][url].base.push_back('/');
                            }
                        } else if (infos[i].mKind == kObjectKindDirectory) {
                            recursive_hdfs_visit(infos[i].mName);
                        }
                    }
                    hdfsFreeFileInfo(infos, num_entries);
                };
                recursive_hdfs_visit(base.c_str());
            } catch (std::regex_error e) {
                ASSERT_MSG(false, (std::string("Illegal regex expression(") + e.what() + "): " + filter).c_str());
            }
            hdfsFreeFileInfo(base_info, 1);
            hdfsDisconnect(fs);
        } else if (base_info->mKind == kObjectKindFile) {   // current is file
            hdfsFreeFileInfo(base_info, 1);
            int dummy;
            auto blk_loc = hdfsGetFileBlockLocations(fs, base.c_str(), 0, 1, &dummy);
            for (int j = 0; j < blk_loc->numOfNodes; ++j) {
                std::string host = blk_loc->hosts[j];
                files_info[host][url].files_to_assign.push_back(base);

                // store files_info base
                files_info[host][url].base = std::string(base_info->mName);
                if (files_info[host][url].base.back() != '/')
                    files_info[host][url].base.push_back('/');
            }
            hdfsDisconnect(fs);
        } else {
            hdfsFreeFileInfo(base_info, 1);
            hdfsDisconnect(fs);
            ASSERT_MSG(false, ("Given base path is neither a file nor a directory: " + base).c_str());
        }
    }

    void print_file_infos() {
        std::stringstream ss;
        for (auto& a : file_infos_) {
            for (auto& b : a.second) {
                for (auto& c : b.second) {
                    for (auto& d : c.second.files_to_assign) {
                        // id, host, file_url, file
                        ss << a.first << " " << b.first << " " << c.first << " " << d << "\n";
                    }
                }
            }
        }
        husky::LOG_I << ss.str();
    }

    // id: host: file_url: [files]
    std::unordered_map<size_t, std::unordered_map<std::string, std::unordered_map<std::string, HDFSFileInfo>>> file_infos_;
    // id: finished_count
    std::map<size_t, std::map<std::string, std::pair<std::map<std::string, size_t>, size_t>>> finish_multi_dict;
};

}  // namespace husky
