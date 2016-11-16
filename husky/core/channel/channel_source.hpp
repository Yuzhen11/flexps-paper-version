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

#include <unordered_map>
#include <vector>

#include "husky/core/channel/channel_base.hpp"

namespace husky {

class ChannelSource {
   public:
    std::vector<ChannelBase*> get_outchannels();
    void register_outchannel(size_t channel_id, ChannelBase* outchannel);
    void deregister_outchannel(size_t channel_id);

   protected:
    std::unordered_map<size_t, ChannelBase*> outchannels_;
};

}  // namespace husky
