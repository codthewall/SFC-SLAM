#ifdef sfc_core
#undef sfc_core
#endif

#include "sfc_core/mapping/atlas.hpp"
#include "sfc_core/mapping/map.hpp"

namespace sfc_core {
namespace mapping {

Atlas::Atlas() {
    mpMap_ = std::make_shared<Map>();
}

Atlas::~Atlas() = default;

} // namespace mapping
} // namespace sfc_core
