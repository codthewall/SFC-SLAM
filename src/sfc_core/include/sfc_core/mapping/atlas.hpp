// 必须在最开头清除 sfc_core 宏
#ifdef sfc_core
#undef sfc_core
#endif

#ifndef SFC_CORE_MAPPING_ATLAS_HPP_
#define SFC_CORE_MAPPING_ATLAS_HPP_

#include <memory>

#include "sfc_core/mapping/map.hpp"

namespace sfc_core {
namespace mapping {

// ================================================================
// Atlas — 多地图管理器（纯前端存根）
// 纯前端 VO 模式下：只管理一个活动地图（mpMap）
// ================================================================
class Atlas {
public:
    using Ptr = std::shared_ptr<Atlas>;
    Atlas();
    ~Atlas();

    // 获取当前活动地图
    Map::Ptr getActiveMap() const { return mpMap_; }

private:
    Map::Ptr mpMap_;
};

} // namespace mapping
} // namespace sfc_core

#endif // SFC_CORE_MAPPING_ATLAS_HPP_
