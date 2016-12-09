#pragma once

#include <functional>

namespace husky {

class Info;
using FuncT = std::function<void(const Info&)>;
}
