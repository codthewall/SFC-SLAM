#ifdef sfc_core
#undef sfc_core
#endif

#include "sfc_core/mapping/local_mapping.hpp"
#include "sfc_core/mapping/map.hpp"
#include "sfc_core/mapping/key_frame.hpp"
#include "sfc_core/mapping/map_point.hpp"
#include <chrono>
#include <thread>

namespace sfc_core {
namespace mapping {

LocalMapping::LocalMapping(Map::Ptr map) : mpMap_(map) {}

LocalMapping::~LocalMapping() {
    shutDown();
}

void LocalMapping::run() {
    if (thread_) {
        return;
    }

    is_running_ = true;
    thread_ = new std::thread([this]() {
        while (is_running_) {
            KeyFrame* pKF = nullptr;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!keyframe_queue_.empty()) {
                    pKF = keyframe_queue_.front();
                    keyframe_queue_.pop();
                }
            }

            if (pKF) {
                processKeyFrame(pKF);
                continue;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });
}

void LocalMapping::shutDown() {
    is_running_ = false;
    if (thread_ && thread_->joinable()) {
        thread_->join();
        delete thread_;
        thread_ = nullptr;
    }
}

void LocalMapping::insertKeyFrame(KeyFrame* pKF) {
    if (!pKF) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        keyframe_queue_.push(pKF);
    }

    if (!is_running_) {
        processKeyFrame(pKF);
    }
}

int LocalMapping::numKeyFramesInQueue() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(keyframe_queue_.size());
}

std::unordered_map<int, Eigen::Vector3d> LocalMapping::getLastUpdatedPoints() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_updated_points_;
}

std::vector<int> LocalMapping::getLastOptimizedKFIds() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_optimized_kf_ids_;
}

void LocalMapping::processKeyFrame(KeyFrame* pKF) {
    if (!pKF || !mpMap_) {
        return;
    }

    std::unordered_map<int, Eigen::Vector3d> updated_points;
    std::vector<int> optimized_kf_ids{pKF->frameId()};

    for (const auto& kv : pKF->getMapPoints()) {
        MapPoint* pMP = kv.second;
        if (!pMP) {
            continue;
        }

        MapPoint* pCanonical = mpMap_->getMapPoint(pMP->id());
        if (!pCanonical) {
            mpMap_->addMapPoint(pMP);
            pCanonical = pMP;
        }

        pCanonical->addObservation(pKF->frameId(), kv.first);
        if (pCanonical->referenceKeyFrame() < 0) {
            pCanonical->setReferenceKeyFrame(pKF->frameId());
        }
        if (pCanonical->firstKFid() < 0) {
            pCanonical->setFirstKFid(pKF->frameId());
        }
        updated_points[pCanonical->id()] = pCanonical->getWorldPos();
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        last_updated_points_ = std::move(updated_points);
        last_optimized_kf_ids_ = std::move(optimized_kf_ids);
    }
}

} // namespace mapping
} // namespace sfc_core
