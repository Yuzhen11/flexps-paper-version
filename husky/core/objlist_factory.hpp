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

#include <string>
#include <unordered_map>

#include "core/context.hpp"
#include "husky/core/objlist.hpp"
#include "core/utils.hpp"

namespace husky {

/// 3 APIs are provided
/// create_objlist(), get_objlist(), drop_objlist()
/// See the unittest for the usages
class ObjListFactory {
   public:
    template <typename ObjT>
    static ObjList<ObjT>& create_objlist(const std::string& name = "") {
        std::string list_name = name.empty() ? objlist_name_prefix + std::to_string(default_objlist_id++) : name;
        ASSERT_MSG(objlist_map.find(list_name) == objlist_map.end(),
                   "ObjListFactory::create_objlist: ObjList name already exists");
        auto* objlist = new ObjList<ObjT>();
        // objlist->set_hash_ring(*(Context::get_hashring()));
        objlist_map.insert({list_name, objlist});
        return *objlist;
    }

    template <typename ObjT>
    static ObjList<ObjT>& get_objlist(const std::string& name) {
        ASSERT_MSG(objlist_map.find(name) != objlist_map.end(),
                   "ObjListFactory::get_objlist: ObjList name doesn't exist");
        auto* objlist = objlist_map[name];
        return *dynamic_cast<ObjList<ObjT>*>(objlist);
    }

    static void drop_objlist(const std::string& name) {
        ASSERT_MSG(objlist_map.find(name) != objlist_map.end(),
                   "ObjListFactory::drop_objlist: ObjList name doesn't exist");
        delete objlist_map[name];
        objlist_map.erase(name);
    }

    static bool has_objlist(const std::string& name) { return objlist_map.find(name) != objlist_map.end(); }

    static size_t size() { return objlist_map.size(); }

   protected:
    static thread_local std::unordered_map<std::string, ObjListBase*> objlist_map;
    static thread_local int default_objlist_id;
    static const char* objlist_name_prefix;
};

}  // namespace husky
