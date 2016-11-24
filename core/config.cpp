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

#include "core/config.hpp"

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "boost/program_options.hpp"

#include "base/log.hpp"
#include "core/context.hpp"
#include "core/network.hpp"

namespace husky {

using base::log_msg;
using base::LOG_TYPE;

Config::Config() {
    machines_.clear();
    params_.clear();
}

Config::~Config() {
    machines_.clear();
    params_.clear();
}

void Config::set_master_host(const std::string& master_host) { master_host_ = master_host; }

void Config::set_master_port(const int& master_port) { master_port_ = master_port; }

void Config::set_comm_port(const int& comm_port) { comm_port_ = comm_port; }

void Config::set_worker_port(const int& worker_port) { worker_port_ = worker_port; }

void Config::set_param(const std::string& key, const std::string& value) { params_[key] = value; }

bool Config::init_with_args(int ac, char** av, const std::vector<std::string>& customized) {
    namespace po = boost::program_options;

    po::options_description generic_options("Generic options");
    generic_options.add_options()("help,H", "print help message");

    std::string config_file_path;
    po::options_description config_file_options("Configuration File");
    config_file_options.add_options()("conf,C", po::value<std::string>(&config_file_path), "Configure file Path");

    std::string master_host;
    int master_port;
    int comm_port;
    int worker_port;
    po::options_description required_options("Required options");
    required_options.add_options()("master_host", po::value<std::string>(&master_host), "Master hostname")(
        "master_port", po::value<int>(&master_port), "Master port")("comm_port", po::value<int>(&comm_port),
                                "Communication port")("worker_port", po::value<int>(&worker_port));

    po::options_description worker_info_options("Worker Info options");
    worker_info_options.add_options()("worker.info", po::value<std::vector<std::string>>()->multitoken(),
                                      "Worker information.\nFormat is '%worker_hostname:%thread_number'.\nUse "
                                      "colon ':' as separator.");

    po::options_description worker_info_config("Worker Info from config");
    worker_info_config.add_options()("worker.info", po::value<std::vector<std::string>>()->multitoken(),
                                     "Worker information.\nFormat is '%worker_hostname:%thread_number'.\nUse "
                                     "colon ':' as separator.");

    po::options_description customized_options("Customized options");
    if (!customized.empty())
        for (auto& arg : customized)
            customized_options.add_options()(arg.c_str(), po::value<std::string>(), "");

    po::options_description cmdline_options;
    cmdline_options.add(generic_options).add(config_file_options).add(required_options).add(worker_info_options);
    po::options_description config_options;
    config_options.add(required_options).add(worker_info_config);
    if (!customized_options.options().empty()) {
        cmdline_options.add(customized_options);
        config_options.add(customized_options);
    }

    po::variables_map vm;
    po::store(po::command_line_parser(ac, av).options(cmdline_options).run(), vm);
    po::notify(vm);

    if (ac == 1 || vm.count("help")) {
        std::cout << "Usage:" << std::endl;
        std::cout << cmdline_options << std::endl;
        return false;
    }

    if (vm.count("conf")) {
        std::ifstream config_file(config_file_path.c_str());
        if (!config_file) {
            log_msg("Can not open config file: " + config_file_path, LOG_TYPE::LOG_ERROR);
            return false;
        }
        // The configure in config_file would be overwritten by cmdline.
        po::store(po::parse_config_file(config_file, config_options, true), vm);
        po::notify(vm);
    }

    int setup_all = 0;

    if (vm.count("master_host")) {
        set_master_host(master_host);
        setup_all += 1;
    } else {
        log_msg("arg master_host is needed", LOG_TYPE::LOG_ERROR);
    }

    if (vm.count("master_port")) {
        set_master_port(master_port);
        setup_all += 1;
    } else {
        log_msg("arg master_port is needed", LOG_TYPE::LOG_ERROR);
    }

    if (vm.count("comm_port")) {
        set_comm_port(comm_port);
        setup_all += 1;
    } else {
        log_msg("arg comm_port is needed", LOG_TYPE::LOG_ERROR);
    }

    if (vm.count("worker_port")) {
        set_worker_port(worker_port);
        setup_all += 1;
    } else {
        log_msg("arg worker_port is needed", LOG_TYPE::LOG_ERROR);
    }

    if (vm.count("worker.info")) {
        std::string hostname = get_hostname();
        int proc_id = -1;
        int num_workers = 0;
        int num_local_threads = 0;
        int num_global_threads = 0;
        std::vector<std::string> workers_info = vm["worker.info"].as<std::vector<std::string>>();
        for (auto& w : workers_info) {
            std::size_t colon_pos = w.find(':');
            if (colon_pos == std::string::npos || colon_pos == w.size() - 1) {
                // Cannot find colon ':' or lack number of threads.
                log_msg("arg worker.info '" + w + "' not match the format", LOG_TYPE::LOG_ERROR);
                return false;
            }
            std::string worker_hostname = w.substr(0, colon_pos);
            machines_.insert(worker_hostname);
            int num_threads = std::stoi(w.substr(colon_pos + 1, w.size() - colon_pos - 1));
            if (is_local(worker_hostname)) {
                num_local_threads = num_threads;
                proc_id = num_workers;
            }
            Context::get_global()->worker_info.add_proc(num_workers, worker_hostname);
            for (int i = 0; i < num_threads; i++) {
                // TODO: Disable hashring now
                // Context::get_global()->hash_ring.insert(num_global_threads, num_workers);
                Context::get_global()->worker_info.add_worker(num_workers, num_global_threads, i);
                ++num_global_threads;
            }
            num_workers += 1;
        }
        Context::get_global()->worker_info.set_num_processes(num_workers);
        Context::get_global()->worker_info.set_num_workers(num_global_threads);
        Context::get_global()->worker_info.set_proc_id(proc_id);
        set_param("hostname", hostname);
        setup_all += 1;
    } else {
        log_msg("arg worker.info is needed", LOG_TYPE::LOG_ERROR);
    }

    if (!customized.empty()) {
        for (auto& arg : customized)
            if (vm.count(arg.c_str())) {
                set_param(arg, vm[arg.c_str()].as<std::string>());
                setup_all += 1;
            } else {
                log_msg("arg " + arg + " is needed", LOG_TYPE::LOG_ERROR);
            }
    }

    if (setup_all != customized.size() + 5) {
        log_msg("Please provide all necessary args!", LOG_TYPE::LOG_ERROR);
        return false;
    }

    return true;
}

}  // namespace husky
