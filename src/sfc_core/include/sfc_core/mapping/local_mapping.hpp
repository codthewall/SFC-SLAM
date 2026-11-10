// 必须在最开头清除 sfc_core 宏
#ifdef sfc_core
#undef sfc_core
#endif

#ifndef SFC_CORE_MAPPING_LOCAL_MAPPING_HPP_
#define SFC_CORE_MAPPING_LOCAL_MAPPING_HPP_

#include <memory>
#include <thread>
#include <atomic>
#include <queue>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <functional>
#include <Eigen/Dense>

#include "sfc_core/mapping/map.hpp"

namespace sfc_core {
namespace mapping {

class KeyFrame;

class LocalMapping {
public:
    using Ptr = std::shared_ptr<LocalMapping>;

    explicit LocalMapping(Map::Ptr map);
    ~LocalMapping();

    void run();
    void shutDown();
    bool isRunning() const { return is_running_.load(); }

    void insertKeyFrame(KeyFrame* pKF);
    int numKeyFramesInQueue() const;

    void setReplacementCallback(
        std::function<void(int old_id, int new_id)> callback) {
        replacement_callback_ = std::move(callback);
    }

    std::unordered_map<int, Eigen::Vector3d> getLastUpdatedPoints() const;
    std::vector<int> getLastOptimizedKFIds() const;

private:
    void processKeyFrame(KeyFrame* pKF);

    Map::Ptr mpMap_;
    std::atomic<bool> is_running_{false};
    std::thread* thread_ = nullptr;
    mutable std::mutex mutex_;
    std::queue<KeyFrame*> keyframe_queue_;
    std::unordered_map<int, Eigen::Vector3d> last_updated_points_;
    std::vector<int> last_optimized_kf_ids_;
    std::function<void(int old_id, int new_id)> replacement_callback_;
};

} // namespace mapping
} // namespace sfc_core

#endif // SFC_CORE_MAPPING_LOCAL_MAPPING_HPP_
