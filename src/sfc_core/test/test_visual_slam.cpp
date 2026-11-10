/**
 * test_visual_slam.cpp
 *
 * ROS2 图像话题 SLAM 跟踪测试：订阅 RGB + IR 图像话题，
 * 通过 Tracking 模块（PnP + g2o BA）进行位姿估计与跟踪。
 */

// 必须在最开头清除 CMake project() 注入的 sfc_core 宏
// 与 sfc_core 所有头文件保持一致，防止 namespace sfc_core { 被展开为 namespace 1 {
#ifdef sfc_core
#undef sfc_core
#endif

#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>
#include <thread>
#include <mutex>
#include <atomic>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/eigen.hpp>

#include "sfc_core/frontend/tracking.hpp"
#include "sfc_core/frontend/frame.hpp"
#include "sfc_core/feature/feature_extractor.hpp"
#include "sfc_core/feature/feature_matcher.hpp"
#include "sfc_core/mapping/sparse_mapper.hpp"
#include "sfc_core/loopdetection/loop_detector.hpp"
#include "sfc_core/loopdetection/pose_graph_optimizer.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>

// ============================================================================
// 相机内参结构与 YAML 加载
// ============================================================================

struct CameraInfo {
    cv::Mat K;          // 3x3 内参矩阵
    cv::Mat D;          // 畸变系数 (1x5)
    int width = 0;
    int height = 0;
    std::string name;
};

CameraInfo loadCameraYaml(const std::string& yaml_path) {
    CameraInfo info;

    std::ifstream test_file(yaml_path);
    if (!test_file.good()) {
        std::cerr << "[loadCameraYaml] 文件不存在或无法访问: " << yaml_path << "\n";
        return info;
    }
    test_file.close();

    try {
        cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "[loadCameraYaml] 无法打开文件: " << yaml_path << "\n";
            return info;
        }

        info.width  = static_cast<int>(fs["image_width"]);
        info.height = static_cast<int>(fs["image_height"]);
        info.name   = static_cast<std::string>(fs["camera_name"]);

        cv::FileNode cm = fs["camera_matrix"];
        std::vector<double> kdata;
        cm["data"] >> kdata;
        if (kdata.size() >= 9) {
            info.K = cv::Mat(3, 3, CV_64F);
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    info.K.at<double>(r, c) = kdata[r * 3 + c];
        }

        cv::FileNode dc = fs["distortion_coefficients"];
        std::vector<double> ddata;
        dc["data"] >> ddata;
        if (!ddata.empty()) {
            info.D = cv::Mat(1, static_cast<int>(ddata.size()), CV_64F, ddata.data()).clone();
        }

        std::cout << "[loadCameraYaml] " << info.name
                  << " " << info.width << "x" << info.height
                  << "  K=" << info.K.at<double>(0,0) << "," << info.K.at<double>(1,1) << "\n";
    } catch (const cv::Exception& e) {
        std::cerr << "[loadCameraYaml] OpenCV 异常: " << e.what() << "\n";
        std::cerr << "  文件路径: " << yaml_path << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[loadCameraYaml] 标准异常: " << e.what() << "\n";
        std::cerr << "  文件路径: " << yaml_path << "\n";
    }
    return info;
}

// ============================================================================
// 双目 / IR-RGB 外参结构与 YAML 加载
// ============================================================================

struct StereoExtrinsic {
    cv::Mat R_ir_to_rgb;   // 3x3 旋转矩阵：IR → RGB
    cv::Mat T_ir_to_rgb;   // 3x1 平移向量（米）：IR → RGB
    cv::Mat Q;             // 4x4 视差深度 Q 矩阵
    bool valid = false;
};

static cv::Mat matFromFlat(const std::vector<double>& data, int rows, int cols) {
    cv::Mat m(rows, cols, CV_64F);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            m.at<double>(r, c) = data[r * cols + c];
    return m;
}

StereoExtrinsic loadStereoExtrinsicYaml(const std::string& yaml_path) {
    StereoExtrinsic ex;
    std::ifstream test(yaml_path);
    if (!test.good()) {
        std::cerr << "[loadStereoExtrinsicYaml] 文件不存在: " << yaml_path << "\n";
        return ex;
    }
    test.close();

    try {
        cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "[loadStereoExtrinsicYaml] 无法打开: " << yaml_path << "\n";
            return ex;
        }

        // rotation_matrix (rows:3 cols:3 data:[...])
        cv::FileNode rn = fs["rotation_matrix"];
        std::vector<double> rdata;
        rn["data"] >> rdata;
        if (rdata.size() >= 9) {
            ex.R_ir_to_rgb = matFromFlat(rdata, 3, 3);
        }

        // translation_vector (rows:3 cols:1 data:[...])
        cv::FileNode tn = fs["translation_vector"];
        std::vector<double> tdata;
        tn["data"] >> tdata;
        if (!tdata.empty()) {
            ex.T_ir_to_rgb = cv::Mat(3, 1, CV_64F, tdata.data()).clone();
        }

        // Q_matrix (rows:4 cols:4 data:[...])
        cv::FileNode qn = fs["Q_matrix"];
        std::vector<double> qdata;
        qn["data"] >> qdata;
        if (qdata.size() >= 16) {
            ex.Q = matFromFlat(qdata, 4, 4);
        }

        if (!ex.R_ir_to_rgb.empty() && !ex.T_ir_to_rgb.empty()) {
            ex.valid = true;
            std::cout << "[loadStereoExtrinsicYaml] 已加载外参: ||T||="
                      << cv::norm(ex.T_ir_to_rgb) << " m  Q=" << ex.Q.at<double>(3, 2) << "\n";
        }
    } catch (const cv::Exception& e) {
        std::cerr << "[loadStereoExtrinsicYaml] OpenCV 异常: " << e.what() << "\n";
    }
    return ex;
}

/**
 * 将原始深度图（IR/深度相机，1280×720）重投影到 RGB 相机坐标系，
 * 输出与 RGB 同分辨率（640×480）的对齐深度图。
 *
 * 步骤：
 *   原始深度(u_d, v_d) → IR 去畸变归一化坐标 (u_d', v_d')
 *   → IR 相机系 3D: X_ir = u_d'·Z_ir, Y_ir = v_d'·Z_ir, Z_ir
 *   → RGB 相机系: P_rgb = R · P_ir + T
 *   → RGB 像素投影: (u_rgb, v_rgb) = (fx·X_rgb/Z_rgb + cx, fy·Y_rgb/Z_rgb + cy)
 *   → 深度填入 RGB 图像坐标系
 *
 * 适用于：RealSense 等设备，深度流为 1280×720、RGB 为 640×480，
 *         需要正确几何对齐（不要用 cv::resize 混叠深度）。
 *
 * @param depth_raw   原始深度图（IR相机，16UC1=mm 或 32FC1=m）
 * @param ir_K        IR 相机内参（对应 depth_raw 分辨率，标定参数）
 * @param ir_D        IR 相机畸变系数
 * @param rgb_K       RGB 相机内参（对应 rgb_target_W×H）
 * @param R_ir_to_rgb IR→RGB 旋转矩阵 (3×3)
 * @param T_ir_to_rgb IR→RGB 平移向量 (3×1)，米
 * @param rgb_W       RGB 目标宽度（640）
 * @param rgb_H       RGB 目标高度（480）
 * @param depth_scale 深度单位转换（如 0.001 = mm→m）
 * @return 对齐后的深度图（16UC1 mm，尺寸 rgb_W×rgb_H）
 */
static cv::Mat reprojectDepthToRGBAligned(
    cv::InputArray depth_raw,
    cv::InputArray ir_K,
    cv::InputArray ir_D,
    cv::InputArray rgb_K,
    cv::InputArray R_ir_to_rgb,
    cv::InputArray T_ir_to_rgb,
    int rgb_W, int rgb_H,
    double depth_scale)
{
    cv::Mat depth = depth_raw.getMat();
    if (depth.empty()) return {};

    const cv::Mat& Kd = ir_K.getMat();
    const cv::Mat& Dd = ir_D.getMat();
    const cv::Mat& Kr = rgb_K.getMat();
    const cv::Mat& R  = R_ir_to_rgb.getMat();
    const cv::Mat& T  = T_ir_to_rgb.getMat();

    const int dH = depth.rows, dW = depth.cols;
    const double rfx = Kr.at<double>(0, 0);
    const double rfy = Kr.at<double>(1, 1);
    const double rcx = Kr.at<double>(0, 2);
    const double rcy = Kr.at<double>(1, 2);

    const double r00 = R.at<double>(0, 0), r01 = R.at<double>(0, 1), r02 = R.at<double>(0, 2);
    const double r10 = R.at<double>(1, 0), r11 = R.at<double>(1, 1), r12 = R.at<double>(1, 2);
    const double r20 = R.at<double>(2, 0), r21 = R.at<double>(2, 1), r22 = R.at<double>(2, 2);
    const double tx  = T.at<double>(0, 0);
    const double ty  = T.at<double>(1, 0);
    const double tz  = T.at<double>(2, 0);

    std::vector<cv::Point2f> raw_pts(dH * dW);
    for (int y = 0; y < dH; ++y)
        for (int x = 0; x < dW; ++x)
            raw_pts[static_cast<size_t>(y) * dW + x] = cv::Point2f(static_cast<float>(x), static_cast<float>(y));

    std::vector<cv::Point2f> undist_pts;
    cv::undistortPoints(raw_pts, undist_pts, Kd, Dd, cv::noArray(), Kd);

    cv::Mat rgb_depth(rgb_H, rgb_W, CV_16UC1, cv::Scalar(0));

    for (int yd = 0; yd < dH; ++yd) {
        for (int xd = 0; xd < dW; ++xd) {
            size_t idx = static_cast<size_t>(yd) * dW + xd;
            float raw = (depth.type() == CV_32FC1)
                            ? depth.at<float>(yd, xd)
                            : static_cast<float>(depth.at<uint16_t>(yd, xd));
            float Z_ir = raw * static_cast<float>(depth_scale);
            if (Z_ir <= 0.f) continue;

            float uu = undist_pts[idx].x;
            float vv = undist_pts[idx].y;
            float X_ir = uu * Z_ir;
            float Y_ir = vv * Z_ir;

            float X_rgb = r00 * X_ir + r01 * Y_ir + r02 * Z_ir + tx;
            float Y_rgb = r10 * X_ir + r11 * Y_ir + r12 * Z_ir + ty;
            float Z_rgb = r20 * X_ir + r21 * Y_ir + r22 * Z_ir + tz;
            if (Z_rgb <= 0.f) continue;

            float u_rgb_f = rfx * X_rgb / Z_rgb + static_cast<float>(rcx);
            float v_rgb_f = rfy * Y_rgb / Z_rgb + static_cast<float>(rcy);

            int u_rgb = static_cast<int>(std::round(u_rgb_f));
            int v_rgb = static_cast<int>(std::round(v_rgb_f));
            if (u_rgb < 0 || u_rgb >= rgb_W || v_rgb < 0 || v_rgb >= rgb_H) continue;

            uint16_t depth_mm = static_cast<uint16_t>(std::round(Z_rgb * 1000.f));
            // 深度图可能有重叠区域，用最近像素覆盖（简单取最后写入值即可）
            rgb_depth.at<uint16_t>(v_rgb, u_rgb) = depth_mm;
        }
    }
    return rgb_depth;
}
// ============================================================================

class ImageSubscriberNode : public rclcpp::Node {
public:
    ImageSubscriberNode(const std::string& rgb_topic,
                        const std::string& ir_topic,
                        const CameraInfo& rgb_info,
                        const CameraInfo& ir_info,
                        const sfc_core::feature::ExtractorConfig& ext_cfg,
                        const sfc_core::feature::MatcherConfig& mat_cfg,
                        bool rgb_to_ir,
                        const StereoExtrinsic& stereo_ex)
        : Node("sfc_image_subscriber")
        , ext_cfg_(ext_cfg), mat_cfg_(mat_cfg)
        , rgb_info_(rgb_info), ir_info_(ir_info)
        , rgb_to_ir_(rgb_to_ir)
        , shutting_down_(false)
        , last_rgb_stamp_(this->now())
        , last_ir_stamp_(this->now())
        , last_depth_stamp_(this->now())
        , disp_last_pose_(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero())
        , stereo_ex_(stereo_ex)
    {
        sfc_core::frontend::TrackingConfig tcfg;
        tcfg.extractor_config = ext_cfg;
        tcfg.matcher_config = mat_cfg;
        tcfg.K = rgb_info.K;
        tcfg.D = rgb_info.D;
        tcfg.depth_scale = 0.001;
        tcfg.min_depth = 0.1;
        tcfg.max_depth = 8.0;
        tcfg.depth_edge_thresh = 0.2;
        tcfg.min_keyframe_translation = 0.02;
        tcfg.min_keyframe_rotation = 0.02;
        tcfg.min_features = 30;
        // R3：RealSense 的深度已配准到 RGB 坐标系 —— 这只影响"深度采样用哪个像素"，
        // 而 compute3DPoints 现在固定用原始（畸变）像素采样深度、用理想像素反投影，
        // 所以去畸变该照常打开（D 非空时）。
        tcfg.disable_undistort = tcfg.D.empty();
        tracking_ = std::make_unique<sfc_core::frontend::Tracking>(tcfg);

        // 初始化回环检测
        sfc_core::loopdetection::LoopDetectorConfig loop_cfg;
        loop_cfg.camera_K = rgb_info.K;
        loop_cfg.camera_D = rgb_info.D;
        loop_cfg.min_loop_separation = 10;
        loop_cfg.max_match_distance = 90;
        loop_cfg.min_geometric_inliers = 10;
        sfc_core::loopdetection::PoseGraphOptimizerOptions pgo_opts;
        pgo_opts.max_iterations = 50;
        tracking_->initLoopDetection(loop_cfg, pgo_opts);

        // 初始化稀疏建图器
        sparse_mapper_ = std::make_unique<sfc_core::mapping::SparseMapper>();

        // 初始化 rviz2 发布者
        pub_map_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/sfc/map_points", 10);
        pub_trajectory_ = this->create_publisher<geometry_msgs::msg::PoseArray>(
            "/sfc/trajectory", 10);

        if (stereo_ex.valid) {
            RCLCPP_INFO(this->get_logger(), "外参已加载：||T_ir→rgb||=%.4f m", cv::norm(stereo_ex.T_ir_to_rgb));
        } else {
            RCLCPP_WARN(this->get_logger(), "未加载外参，无法做 IR→RGB 重投影");
        }
    }

    // 通知节点准备关闭：阻止回调继续处理数据
    void requestShutdown() {
        shutting_down_ = true;
    }

    bool isShuttingDown() const {
        return shutting_down_;
    }

    void init(const std::string& rgb_topic, const std::string& ir_topic,
              const std::string& depth_topic) {
        rgb_topic_name_ = rgb_topic;
        ir_topic_name_ = ir_topic;
        depth_topic_ = depth_topic;
        RCLCPP_INFO(this->get_logger(), "订阅 RGB=%s IR=%s Depth=%s",
                    rgb_topic_name_.c_str(), ir_topic_name_.c_str(), depth_topic_.c_str());

        image_transport::ImageTransport it(shared_from_this());
        sub_rgb_ = it.subscribe(rgb_topic, 10,
            [this](const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
                std::lock_guard<std::mutex> lk(mtx_);
                cv_bridge::CvImagePtr cv_ptr;
                try {
                    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
                } catch (const cv_bridge::Exception& e) {
                    RCLCPP_ERROR(this->get_logger(), "RGB cv_bridge error: %s", e.what());
                    return;
                }
                rgb_frame_ = cv_ptr->image.clone();
                rgb_stamp_ = msg->header.stamp;
                last_rgb_stamp_ = this->get_clock()->now();
                tryDispatch();
            });

        sub_ir_ = it.subscribe(ir_topic, 10,
            [this](const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
                std::lock_guard<std::mutex> lk(mtx_);
                cv_bridge::CvImagePtr cv_ptr;
                try {
                    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
                } catch (const cv_bridge::Exception& e) {
                    RCLCPP_ERROR(this->get_logger(), "IR cv_bridge error: %s", e.what());
                    return;
                }
                ir_frame_ = cv_ptr->image.clone();
                ir_stamp_ = msg->header.stamp;
                last_ir_stamp_ = this->get_clock()->now();
                ir_received_once_ = true;
                tryDispatch();
            });

        sub_depth_ = it.subscribe(depth_topic_, 10,
            [this](const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
                std::lock_guard<std::mutex> lk(mtx_);
                cv_bridge::CvImagePtr cv_ptr;
                bool got = false;
                try {
                    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
                    depth_frame_ = cv_ptr->image.clone();
                    got = true;
                } catch (const cv_bridge::Exception&) {
                    try {
                        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
                        cv::Mat mm;
                        cv_ptr->image.convertTo(mm, CV_16UC1, 1000.0, 0);  // m → mm
                        depth_frame_ = mm;
                        got = true;
                    } catch (const cv_bridge::Exception&) {
                        try {
                            cv_ptr = cv_bridge::toCvCopy(msg);
                            if (cv_ptr->image.type() == CV_16UC1) {
                                depth_frame_ = cv_ptr->image.clone();
                                got = true;
                            } else if (cv_ptr->image.type() == CV_32FC1) {
                                cv::Mat mm;
                                cv_ptr->image.convertTo(mm, CV_16UC1, 1000.0, 0);
                                depth_frame_ = mm;
                                got = true;
                            } else {
                                RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                    "Depth: 不支持的类型 type=%d channels=%d",
                                    cv_ptr->image.type(), cv_ptr->image.channels());
                            }
                        } catch (const cv_bridge::Exception& e) {
                            RCLCPP_ERROR(this->get_logger(), "Depth cv_bridge error: %s", e.what());
                        }
                    }
                }
                if (got) {
                    // 诊断：打印图像尺寸、中心点深度值
                    if (!depth_frame_.empty()) {
                        int cy = depth_frame_.rows / 2, cx = depth_frame_.cols / 2;
                        uint16_t center_mm = depth_frame_.at<uint16_t>(cy, cx);
                        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                            "Depth OK: %dx%d center_mm=%d",
                            depth_frame_.cols, depth_frame_.rows, center_mm);
                    }
                    depth_stamp_ = msg->header.stamp;
                    last_depth_stamp_ = this->get_clock()->now();
                    depth_received_once_ = true;
                    tryDispatch();
                }
            });
    }

    bool spinOnce(cv::Mat& out_frame, std::vector<cv::KeyPoint>& out_kps,
                  cv::Mat& out_des, int& out_matches, int& out_inliers,
                  bool& out_tracking, Eigen::Vector3d& out_translation,
                  std::vector<Eigen::Vector3d>& out_trajectory,
                  sfc_core::frontend::Pose& out_pose) {
        // 清除旧数据，防止残留值导致主线程崩溃
        out_frame = cv::Mat();
        out_kps.clear();
        out_des = cv::Mat();
        out_matches = 0;
        out_inliers = 0;
        out_tracking = false;
        out_translation = Eigen::Vector3d::Zero();
        out_trajectory.clear();
        out_pose = sfc_core::frontend::Pose(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());

        std::lock_guard<std::mutex> lk(mtx_disp_);
        if (!disp_frame_.empty()) {
            disp_frame_.copyTo(out_frame);
            out_kps = disp_kps_;
            out_des = disp_des_.clone();
            out_matches = disp_matches_;
            out_inliers = disp_inliers_;
            out_tracking = disp_tracking_;
            out_translation = disp_translation_;
            out_trajectory = disp_trajectory_;
            out_pose = disp_pose_;
            disp_frame_ = cv::Mat();
            return true;
        }
        return false;
    }

    const CameraInfo& rgbInfo()  const { return rgb_info_; }
    const CameraInfo& irInfo()   const { return ir_info_;  }
    sfc_core::frontend::Tracking* getTracking() const { return tracking_.get(); }
    sfc_core::mapping::SparseMapper* getSparseMapper() const { return sparse_mapper_.get(); }

    // ========== rviz2 发布方法 ==========
    sensor_msgs::msg::PointCloud2 buildMapPointCloud2() {
        sensor_msgs::msg::PointCloud2 cloud;
        cloud.header.stamp = this->get_clock()->now();
        cloud.header.frame_id = "world";
        cloud.width = 0;
        cloud.height = 1;
        cloud.is_bigendian = false;
        cloud.point_step = 16;  // x,y,z(4+4+4) + rgb(4)
        cloud.row_step = 0;
        cloud.is_dense = true;

        // 定义 PointCloud2 字段：x, y, z, rgb
        sensor_msgs::msg::PointField f_x, f_y, f_z, f_rgb;
        f_x.name = "x"; f_x.offset = 0;  f_x.datatype = sensor_msgs::msg::PointField::FLOAT32; f_x.count = 1;
        f_y.name = "y"; f_y.offset = 4;  f_y.datatype = sensor_msgs::msg::PointField::FLOAT32; f_y.count = 1;
        f_z.name = "z"; f_z.offset = 8;  f_z.datatype = sensor_msgs::msg::PointField::FLOAT32; f_z.count = 1;
        f_rgb.name = "rgb"; f_rgb.offset = 12; f_rgb.datatype = sensor_msgs::msg::PointField::FLOAT32; f_rgb.count = 1;
        cloud.fields = {f_x, f_y, f_z, f_rgb};

        std::vector<float> data;
        if (sparse_mapper_) {
            for (const auto& kv : sparse_mapper_->getAllMapPoints()) {
                const auto& mp = kv.second;
                if (mp->status() != sfc_core::frontend::PointStatus::CONFIRMED) continue;
                const auto& p = mp->worldPoint();
                if (p.norm() < 1e-6) continue;
                data.push_back(static_cast<float>(p.x()));
                data.push_back(static_cast<float>(p.y()));
                data.push_back(static_cast<float>(p.z()));
                int obs = mp->obsCount();
                uint8_t r = static_cast<uint8_t>(std::max(0, 255 - obs * 15));
                uint8_t g = static_cast<uint8_t>(std::min(255, obs * 15));
                uint8_t b = 200;
                uint32_t rgb_val = (static_cast<uint32_t>(r) << 16) |
                                   (static_cast<uint32_t>(g) << 8) |
                                   static_cast<uint32_t>(b);
                float rgb_float = *reinterpret_cast<float*>(&rgb_val);
                data.push_back(rgb_float);
            }
        }

        cloud.width = static_cast<uint32_t>(data.size()) / 4;
        cloud.row_step = cloud.width * cloud.point_step;
        cloud.data.assign(reinterpret_cast<uint8_t*>(data.data()),
                           reinterpret_cast<uint8_t*>(data.data()) + data.size() * sizeof(float));

        return cloud;
    }

    geometry_msgs::msg::PoseArray buildPoseArray() {
        geometry_msgs::msg::PoseArray arr;
        arr.header.stamp = this->get_clock()->now();
        arr.header.frame_id = "world";
        if (sparse_mapper_) {
            for (const auto& kf : sparse_mapper_->getAllKeyFrames()) {
                if (!kf) continue;
                geometry_msgs::msg::Pose p;
                p.position.x = kf->pose.t.x();
                p.position.y = kf->pose.t.y();
                p.position.z = kf->pose.t.z();
                p.orientation.w = kf->pose.q.w();
                p.orientation.x = kf->pose.q.x();
                p.orientation.y = kf->pose.q.y();
                p.orientation.z = kf->pose.q.z();
                arr.poses.push_back(p);
            }
        }
        return arr;
    }

private:
    void setFallbackDisplay(const std::string& reason) {
        if (rgb_frame_.empty()) return;
        cv::Mat vis = rgb_frame_.clone();
        cv::putText(vis, reason, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
        std::string hint = "Check: ros2 topic echo " + depth_topic_ + " --once";
        cv::putText(vis, hint, cv::Point(20, 75), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(180, 180, 180), 1);
        std::lock_guard<std::mutex> lk(mtx_disp_);
        vis.copyTo(disp_frame_);
        disp_kps_.clear();
        disp_des_ = cv::Mat();
        disp_matches_ = 0;
        disp_inliers_ = 0;
        disp_tracking_ = false;
        disp_translation_ = Eigen::Vector3d::Zero();
        disp_trajectory_.clear();
        disp_pose_ = sfc_core::frontend::Pose(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
        vis_smooth_ready_ = false;
    }

    void tryDispatch() {
        if (shutting_down_) return;
        if (rgb_frame_.empty()) return;
        if (ir_frame_.empty()) {
            // 已收到过 IR 时多为「本帧 RGB 先到」的瞬态，勿用提示覆盖上一帧正常画面
            if (!ir_received_once_) {
                setFallbackDisplay("Waiting for IR: " + ir_topic_name_);
            }
            return;
        }
        if (depth_frame_.empty()) {
            if (!depth_received_once_) {
                setFallbackDisplay("Waiting for depth: " + depth_topic_);
            }
            return;
        }

        constexpr double k_sync_tol_sec = 0.5;
        double dt_ir    = std::abs(rclcpp::Time(rgb_stamp_).seconds() - rclcpp::Time(ir_stamp_).seconds());
        double dt_depth = std::abs(rclcpp::Time(rgb_stamp_).seconds() - rclcpp::Time(depth_stamp_).seconds());
        if (dt_ir > k_sync_tol_sec || dt_depth > k_sync_tol_sec) {
            // 已成功同步过后，偶发不同步时静默丢帧，避免黄字盖住正常画面
            if (!frame_sync_ok_once_) {
                setFallbackDisplay("Waiting for RGB/IR/Depth sync (dt<0.5s)");
            }
            return;
        }

        cv::Mat depth_aligned;
        if (depth_frame_.size() != rgb_frame_.size()) {
            cv::resize(depth_frame_, depth_aligned, rgb_frame_.size());
        } else {
            depth_aligned = depth_frame_;
        }

        // 诊断：打印深度图尺寸和有效像素比例
        {
            int valid = 0, zero = 0, total = depth_aligned.rows * depth_aligned.cols;
            for (int y = 0; y < depth_aligned.rows; ++y)
                for (int x = 0; x < depth_aligned.cols; ++x)
                    if (depth_aligned.type() == CV_32FC1) {
                        float v = depth_aligned.at<float>(y, x);
                        if (v > 0.f) ++valid; else ++zero;
                    } else {
                        if (depth_aligned.at<uint16_t>(y, x) > 0) ++valid; else ++zero;
                    }
            RCLCPP_INFO(this->get_logger(),
                "[DepthStat] %dx%d type=%d valid=%d/%d (%.1f%%) zero=%d",
                depth_aligned.cols, depth_aligned.rows, depth_aligned.type(),
                valid, total, 100.0*valid/total, zero);
        }

        // 深度图：RealSense 深度流为 1280×720，与 IR 标定 640×480 不同。
        // 必须用外参正确重投影到 RGB 像素坐标，不要用 cv::resize 混叠深度值。
        if (stereo_ex_.valid && !depth_aligned.empty() && !ir_info_.K.empty() && !rgb_info_.K.empty()) {
            int dW = depth_aligned.cols, dH = depth_aligned.rows;
            int rW = rgb_frame_.cols, rH = rgb_frame_.rows;

            if (dW != rW || dH != rH) {
                // 深度分辨率与 RGB 不同：缩放 IR 内参到实际深度分辨率
                double scale_x = static_cast<double>(dW) / ir_info_.width;
                double scale_y = static_cast<double>(dH) / ir_info_.height;

                cv::Mat ir_K_scaled = ir_info_.K.clone();
                ir_K_scaled.at<double>(0, 0) *= scale_x;
                ir_K_scaled.at<double>(1, 1) *= scale_y;
                ir_K_scaled.at<double>(0, 2) *= scale_x;
                ir_K_scaled.at<double>(1, 2) *= scale_y;

                RCLCPP_INFO(this->get_logger(),
                    "深度 %dx%d ≠ RGB %dx%d，使用外参重投影（scale=%.2f）",
                    dW, dH, rW, rH, scale_x);

                cv::Mat rgb_depth_aligned = reprojectDepthToRGBAligned(
                    depth_aligned, ir_K_scaled, ir_info_.D,
                    rgb_info_.K, stereo_ex_.R_ir_to_rgb, stereo_ex_.T_ir_to_rgb,
                    rW, rH, 0.001);
                if (!rgb_depth_aligned.empty()) depth_aligned = rgb_depth_aligned;
            }
            // dW==rW && dH==rH：分辨率相同，假设深度已与 RGB 对齐，直接用
        }

        cv::Mat ir_bgr;
        cv::cvtColor(ir_frame_, ir_bgr, cv::COLOR_GRAY2BGR);
        if (ir_bgr.size() != rgb_frame_.size())
            cv::resize(ir_bgr, ir_bgr, rgb_frame_.size());

        double ts = rclcpp::Time(rgb_stamp_).seconds();
        sfc_core::frontend::Pose pose = tracking_->track(rgb_frame_, depth_aligned, ir_bgr, ts);

        // ========== SparseMapper 接入：同步关键帧与地图点到全局地图 ==========
        {
            auto* covis = tracking_->getCovisManager();
            if (covis && sparse_mapper_) {
                // 获取所有已注册的关键帧，加入 SparseMapper
                for (const auto& kf : covis->getAllKeyFrames()) {
                    if (!kf) continue;
                    // 查找是否已注册（通过 frame_id 查找）
                    bool found = false;
                    for (const auto& existing : sparse_mapper_->getAllKeyFrames()) {
                        if (existing && existing->frame_id == kf->frame_id) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        sparse_mapper_->addKeyFrame(kf);
                    }
                }
                // 同步 BA 修正后的地图点世界坐标
                for (const auto& kv : covis->getLastBAMapPointUpdates()) {
                    sparse_mapper_->updateMapPoint(kv.first, kv.second);
                }
                // 同步已确认的候选地图点（新增的）
                for (const auto& kv : covis->getAllConfirmedMapPoints()) {
                    if (sparse_mapper_->getMapPoint(kv.first) == nullptr) {
                        sparse_mapper_->addMapPoint(kv.first, kv.second);
                    }
                }
                // 同步候选地图点（使 PLY 能在确认前也显示点云）
                for (const auto& kv : covis->getAllCandidateMapPoints()) {
                    if (sparse_mapper_->getMapPoint(kv.first) == nullptr) {
                        sparse_mapper_->addMapPoint(kv.first, kv.second);
                    }
                }

                // ========== rviz2 实时发布 ==========
                if (pub_map_cloud_ && sparse_mapper_->numConfirmedMapPoints() > 0) {
                    pub_map_cloud_->publish(buildMapPointCloud2());
                }
                if (pub_trajectory_ && sparse_mapper_->numKeyFrames() > 0) {
                    pub_trajectory_->publish(buildPoseArray());
                }
            }
        }

        Eigen::Vector3d frame_translation = pose.t - disp_last_pose_.t;
        Eigen::Quaterniond dq = pose.q * disp_last_pose_.q.inverse();
        double rot_deg = 2 * std::acos(std::abs(dq.w())) * 180.0 / M_PI;

        bool is_tracking = tracking_->isInitialized();

        cv::Mat vis = rgb_frame_.clone();
        std::vector<cv::KeyPoint> kps_vis = tracking_->getCurrentKeypoints();
        int valid_features = static_cast<int>(kps_vis.size());
        int inlier_count = tracking_->getLastInlierCount();

        // 在 RGB 上绘制当前帧特征点（原先只保存 kps 未画到图上）
        if (!kps_vis.empty()) {
            cv::drawKeypoints(vis, kps_vis, vis, cv::Scalar(0, 255, 100),
                              cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
        }

        {
            std::lock_guard<std::mutex> lk(mtx_disp_);
            vis.copyTo(disp_frame_);
            disp_kps_ = kps_vis;
            disp_des_ = cv::Mat();
            disp_matches_ = valid_features;
            disp_inliers_ = inlier_count;
            disp_tracking_ = is_tracking;
            disp_translation_ = frame_translation;
            disp_last_pose_ = pose;

            // 显示层平滑：深度噪声 + 每帧 PnP 微小误差经 compose 后会在世界坐标里表现为随机漂移，
            // 并非「特征闪烁误判运动」。对 UI 轨迹与 World t 做低通 + 轨迹顶点步长门限，Tracking 内部仍用原始 pose。
            constexpr double k_vis_lpf_alpha = 0.18;
            constexpr double k_traj_min_step_m = 0.008;
            if (is_tracking) {
                if (!vis_smooth_ready_) {
                    vis_smooth_t_ = pose.t;
                    vis_smooth_q_ = pose.q.normalized();
                    vis_smooth_ready_ = true;
                    vis_traj_last_commit_ = vis_smooth_t_;
                    disp_trajectory_.clear();
                    disp_trajectory_.push_back(vis_smooth_t_);
                } else {
                    vis_smooth_t_ =
                        (1.0 - k_vis_lpf_alpha) * vis_smooth_t_ + k_vis_lpf_alpha * pose.t;
                    vis_smooth_q_ = vis_smooth_q_.slerp(k_vis_lpf_alpha, pose.q.normalized());
                    vis_smooth_q_.normalize();

                    if ((vis_smooth_t_ - vis_traj_last_commit_).norm() >= k_traj_min_step_m) {
                        disp_trajectory_.push_back(vis_smooth_t_);
                        vis_traj_last_commit_ = vis_smooth_t_;
                        if (disp_trajectory_.size() > 500)
                            disp_trajectory_.erase(disp_trajectory_.begin());
                    } else if (!disp_trajectory_.empty()) {
                        disp_trajectory_.back() = vis_smooth_t_;
                    }
                }
                disp_pose_.q = vis_smooth_q_;
                disp_pose_.t = vis_smooth_t_;
            } else {
                vis_smooth_ready_ = false;
                disp_pose_ = pose;
            }
        }

        frame_sync_ok_once_ = true;

        // 释放缓冲，强制下一帧必须再次凑齐 RGB+IR+Depth，避免新 RGB 与旧 IR/Depth 错配
        rgb_frame_.release();
        ir_frame_.release();
        depth_frame_.release();
    }

    std::mutex mtx_;
    std::mutex mtx_disp_;
    std::atomic<bool> shutting_down_{false};

    image_transport::Subscriber sub_rgb_;
    image_transport::Subscriber sub_ir_;
    image_transport::Subscriber sub_depth_;

    cv::Mat rgb_frame_, ir_frame_, depth_frame_;
    rclcpp::Time rgb_stamp_, ir_stamp_, depth_stamp_;
    rclcpp::Time last_rgb_stamp_, last_ir_stamp_, last_depth_stamp_;

    std::string rgb_topic_name_;
    std::string ir_topic_name_;
    std::string depth_topic_{"/camera/depth/image_raw"};
    bool ir_received_once_{false};
    bool depth_received_once_{false};
    bool frame_sync_ok_once_{false};

    CameraInfo rgb_info_, ir_info_;
    bool rgb_to_ir_;
    StereoExtrinsic stereo_ex_;

    sfc_core::feature::ExtractorConfig ext_cfg_;
    sfc_core::feature::MatcherConfig mat_cfg_;

    std::unique_ptr<sfc_core::frontend::Tracking> tracking_;
    std::unique_ptr<sfc_core::mapping::SparseMapper> sparse_mapper_;

    // ========== rviz2 实时发布 ==========
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_map_cloud_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_trajectory_;

    cv::Mat disp_frame_;
    std::vector<cv::KeyPoint> disp_kps_;
    cv::Mat disp_des_;
    int disp_matches_ = 0;
    int disp_inliers_ = 0;
    bool disp_tracking_ = false;
    Eigen::Vector3d disp_translation_;
    sfc_core::frontend::Pose disp_pose_;
    sfc_core::frontend::Pose disp_last_pose_;
    std::vector<Eigen::Vector3d> disp_trajectory_;

    Eigen::Vector3d vis_smooth_t_{0, 0, 0};
    Eigen::Quaterniond vis_smooth_q_{1, 0, 0, 0};
    Eigen::Vector3d vis_traj_last_commit_{0, 0, 0};
    bool vis_smooth_ready_{false};
};

// ============================================================================
// 可视化工具
// ============================================================================

static void putTextCN(cv::Mat& img, const std::string& text, cv::Point org,
                      double fontScale, cv::Scalar color, int thickness = 1) {
    cv::putText(img, text, org,
                cv::FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv::LINE_AA);
}

cv::Mat drawTrajectory(const std::vector<Eigen::Vector3d>& trajectory, int width = 400, int height = 400) {
    cv::Mat traj_img(height, width, CV_8UC3, cv::Scalar(20, 20, 20));

    if (trajectory.empty()) {
        putTextCN(traj_img, "No trajectory data", cv::Point(10, height / 2), 0.6, cv::Scalar(150, 150, 150), 1);
        return traj_img;
    }

    double min_x = trajectory[0].x(), max_x = trajectory[0].x();
    double min_z = trajectory[0].z(), max_z = trajectory[0].z();
    for (const auto& pt : trajectory) {
        min_x = std::min(min_x, pt.x());
        max_x = std::max(max_x, pt.x());
        min_z = std::min(min_z, pt.z());
        max_z = std::max(max_z, pt.z());
    }

    double range_x = max_x - min_x;
    double range_z = max_z - min_z;
    const double k_degenerate = 1e-5;
    bool degenerate = (range_x < k_degenerate && range_z < k_degenerate);

    if (range_x < 0.01) range_x = 0.01;
    if (range_z < 0.01) range_z = 0.01;

    int margin = 30;
    int plot_w = width - 2 * margin;
    int plot_h = height - 2 * margin;

    for (size_t i = 1; i < trajectory.size(); ++i) {
        int x1 = margin + static_cast<int>((trajectory[i - 1].x() - min_x) / range_x * plot_w);
        int z1 = margin + static_cast<int>((trajectory[i - 1].z() - min_z) / range_z * plot_h);
        int x2 = margin + static_cast<int>((trajectory[i].x() - min_x) / range_x * plot_w);
        int z2 = margin + static_cast<int>((trajectory[i].z() - min_z) / range_z * plot_h);

        z1 = height - z1;
        z2 = height - z2;

        cv::line(traj_img, cv::Point(x1, z1), cv::Point(x2, z2), cv::Scalar(0, 255, 100), 2);
    }

    if (!trajectory.empty()) {
        int start_x = margin + static_cast<int>((trajectory[0].x() - min_x) / range_x * plot_w);
        int start_z = height - (margin + static_cast<int>((trajectory[0].z() - min_z) / range_z * plot_h));
        cv::circle(traj_img, cv::Point(start_x, start_z), 5, cv::Scalar(0, 255, 0), -1);

        int end_x = margin + static_cast<int>((trajectory.back().x() - min_x) / range_x * plot_w);
        int end_z = height - (margin + static_cast<int>((trajectory.back().z() - min_z) / range_z * plot_h));
        cv::circle(traj_img, cv::Point(end_x, end_z), 5, cv::Scalar(0, 0, 255), -1);
    }

    putTextCN(traj_img, "Top view: X (horiz) vs Z (vert)", cv::Point(8, 20), 0.42, cv::Scalar(200, 200, 200), 1);
    std::ostringstream ss; ss << "Points: " << trajectory.size();
    putTextCN(traj_img, ss.str(), cv::Point(8, height - 10), 0.4, cv::Scalar(150, 150, 150), 1);

    if (degenerate) {
        putTextCN(traj_img, "Hint: world pose.t almost constant", cv::Point(8, height - 32), 0.38,
                  cv::Scalar(0, 200, 255), 1);
        putTextCN(traj_img, "(check composeWorldPose / PnP frame)", cv::Point(8, height - 52), 0.35,
                  cv::Scalar(120, 120, 120), 1);
    }

    return traj_img;
}

cv::Mat drawTrackingStats(int frame_id, int features, int matches, int inliers,
                          const Eigen::Vector3d& frame_delta_t, double rotation_deg,
                          const Eigen::Vector3d& world_t,
                          bool is_tracking) {
    int width = 350, height = 300;
    cv::Mat stats_img(height, width, CV_8UC3, cv::Scalar(25, 25, 25));

    int y = 25;
    auto addLine = [&](const std::string& label, const std::string& value, cv::Scalar color) {
        std::string text = label + ": " + value;
        putTextCN(stats_img, text, cv::Point(10, y), 0.45, color, 1);
        y += 25;
    };

    putTextCN(stats_img, "=== Tracking Stats ===", cv::Point(10, y), 0.5, cv::Scalar(100, 200, 255), 1);
    y += 30;

    std::ostringstream ss;
    ss << frame_id;
    addLine("Frame ID", ss.str(), cv::Scalar(200, 200, 200));

    ss.str(""); ss << features;
    addLine("Features", ss.str(), cv::Scalar(0, 255, 100));

    ss.str(""); ss << matches;
    addLine("Matches", ss.str(), matches > 50 ? cv::Scalar(0, 255, 100) : cv::Scalar(80, 80, 255));

    ss.str(""); ss << inliers;
    addLine("Inliers", ss.str(), inliers > 20 ? cv::Scalar(0, 255, 100) : cv::Scalar(80, 80, 255));

    ss.str(""); ss << std::fixed << std::setprecision(5) << frame_delta_t.norm();
    addLine("Frame d|t| (world)", ss.str(), cv::Scalar(200, 200, 200));

    ss.str("");
    ss << std::fixed << std::setprecision(4)
       << world_t.x() << ", " << world_t.y() << ", " << world_t.z();
    addLine("World t (xyz)", ss.str(), cv::Scalar(180, 180, 255));

    double inlier_ratio = (matches > 0) ? (100.0 * inliers / matches) : 0.0;
    ss.str(""); ss << std::fixed << std::setprecision(1) << inlier_ratio << "%";
    cv::Scalar iq_color = (inlier_ratio > 30) ? cv::Scalar(0, 255, 100) :
                          (inlier_ratio > 15) ? cv::Scalar(0, 200, 200) : cv::Scalar(80, 80, 255);
    addLine("Inlier Ratio", ss.str(), iq_color);

    ss.str(""); ss << std::fixed << std::setprecision(2) << rotation_deg;
    addLine("Rotation", ss.str() + " deg", cv::Scalar(150, 150, 150));

    cv::Scalar status_color = is_tracking ? cv::Scalar(0, 255, 100) : cv::Scalar(80, 80, 255);
    std::string status = is_tracking ? "Status: TRACKING" : "Status: LOST";
    putTextCN(stats_img, status, cv::Point(10, height - 10), 0.5, status_color, 1);

    return stats_img;
}

// ============================================================================
// ROS2 图像话题模式
// ============================================================================

void runROS2Mode(const std::string& left_yaml_path,
                 const std::string& right_yaml_path,
                 const std::string& extrinsic_yaml_path,
                 const std::string& rgb_topic,
                 const std::string& ir_topic,
                 const std::string& depth_topic,
                 bool rgb_to_ir) {
    std::cout << "========================================\n";
    std::cout << "  ROS2 Image Topic Mode\n";
    std::cout << "  RGB topic   : " << rgb_topic << "\n";
    std::cout << "  IR  topic   : " << ir_topic << "\n";
    std::cout << "  Depth topic : " << depth_topic << "\n";
    std::cout << "  RGB YAML  : " << right_yaml_path << "\n";
    std::cout << "  IR  YAML  : " << left_yaml_path << "\n";
    std::cout << "  Extrinsic : " << extrinsic_yaml_path << "\n";
    std::cout << "  Mode      : " << (rgb_to_ir ? "RGB->IR" : "IR->RGB") << "\n";
    std::cout << "========================================\n";

    CameraInfo ir_info  = loadCameraYaml(left_yaml_path);
    CameraInfo rgb_info = loadCameraYaml(right_yaml_path);
    if (rgb_info.K.empty() || ir_info.K.empty()) {
        std::cerr << "[runROS2Mode] 内参加载失败，退出。\n";
        return;
    }

    StereoExtrinsic stereo_ex = loadStereoExtrinsicYaml(extrinsic_yaml_path);

    sfc_core::feature::ExtractorConfig ext_cfg;
    ext_cfg.n_features = 2000;
    ext_cfg.scale_factor = 1.2f;
    ext_cfg.n_levels = 8;
    ext_cfg.fast_threshold = 12;
    ext_cfg.edge_threshold = 12;
    ext_cfg.enable_uniform = true;
    ext_cfg.grid_cols = 10;
    ext_cfg.grid_rows = 8;
    ext_cfg.max_per_grid = 12;

    sfc_core::feature::MatcherConfig mat_cfg;
    mat_cfg.matcher_type = 0;
    mat_cfg.max_match_dist = 110.0f;
    mat_cfg.min_dist_ratio = 2.8f;
    mat_cfg.ransac_threshold = 5.0f;
    mat_cfg.lowe_ratio = 0.82f;
    mat_cfg.bidirectional_match = false;
    mat_cfg.min_dist_slack = 28.0f;

    rclcpp::init(0, nullptr);
    auto node = std::make_shared<ImageSubscriberNode>(
        rgb_topic, ir_topic, rgb_info, ir_info, ext_cfg, mat_cfg, rgb_to_ir, stereo_ex);
    node->init(rgb_topic, ir_topic, depth_topic);

    std::thread spin_thread([&]() {
        rclcpp::executors::SingleThreadedExecutor exec;
        exec.add_node(node);
        exec.spin();
    });

    std::vector<Eigen::Vector3d> cumulative_traj;
    Eigen::Vector3d last_t = Eigen::Vector3d::Zero();
    int total_matches = 0, total_inliers = 0;
    int frame_id = 0;
    sfc_core::frontend::Pose last_pose(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());

    cv::namedWindow("ROS2 SLAM", cv::WINDOW_NORMAL);

    while (rclcpp::ok()) {
        cv::Mat vis;
        std::vector<cv::KeyPoint> kps;
        cv::Mat des;
        int matches = 0, inliers = 0;
        bool tracking = false;
        Eigen::Vector3d translation = Eigen::Vector3d::Zero();
        std::vector<Eigen::Vector3d> display_traj;
        sfc_core::frontend::Pose current_pose;

        if (node->spinOnce(vis, kps, des, matches, inliers,
                           tracking, translation, display_traj, current_pose)) {
            if (!vis.empty()) {
                // 轨迹与位姿显示由节点内平滑后的 disp_* 统一维护，此处直接同步即可
                if (tracking && !display_traj.empty())
                    cumulative_traj = display_traj;
                display_traj = cumulative_traj;
                total_matches += matches;
                total_inliers += inliers;

                if (frame_id % 10 == 0) {
                    double inlier_ratio = (matches > 0) ? (100.0 * inliers / matches) : 0.0;
                    std::cout << "[DEBUG] Frame " << frame_id
                              << " | tracking=" << (tracking ? "YES" : "NO")
                              << " | features=" << kps.size()
                              << " | matches=" << matches
                              << " | inliers=" << inliers
                              << " | inlier_ratio=" << std::fixed << std::setprecision(1) << inlier_ratio << "%"
                              << " | pose_t=(x,y,z)=(" << std::fixed << std::setprecision(4)
                              << current_pose.t.x() << ", " << current_pose.t.y() << ", " << current_pose.t.z() << ")"
                              << " | frame_dt=" << translation.norm()
                              << " | cum_traj=" << cumulative_traj.size()
                              << std::endl;
                }

                // 必须先相对上一可视化位姿算旋转，再更新 last_pose（原先顺序反了导致 Rotation 恒为 0）
                double rot_deg = 2.0 * std::acos(std::clamp(
                    std::abs((current_pose.q * last_pose.q.inverse()).w()), 0.0, 1.0)) * 180.0 / M_PI;
                cv::Mat stats = drawTrackingStats(
                    frame_id, static_cast<int>(kps.size()),
                    matches, inliers, translation, rot_deg, current_pose.t, tracking);
                last_pose = current_pose;

                cv::Mat traj_img = drawTrajectory(display_traj, 400, 400);

                int panel_h = std::max(traj_img.rows, stats.rows);
                int panel_w = traj_img.cols + stats.cols;
                cv::Mat panel(panel_h, panel_w, CV_8UC3, cv::Scalar(20, 20, 20));
                if (!traj_img.empty()) traj_img.copyTo(panel(cv::Rect(0, 0, traj_img.cols, traj_img.rows)));
                if (!stats.empty())    stats.copyTo(panel(cv::Rect(traj_img.cols, 0, stats.cols, stats.rows)));

                int out_h = std::max(vis.rows, panel.rows);
                int out_w = vis.cols + panel.cols + 10;
                cv::Mat out(out_h, out_w, CV_8UC3, cv::Scalar(15, 15, 15));
                vis.copyTo(out(cv::Rect(0, 0, vis.cols, vis.rows)));
                panel.copyTo(out(cv::Rect(vis.cols + 10, 0, panel.cols, panel.rows)));

                cv::imshow("ROS2 SLAM", out);
                ++frame_id;
            }
        }

        char key = static_cast<char>(cv::waitKey(30));
        if (key == 27 || key == 'q') break;

        // 定期保存地图
        if (frame_id > 0 && frame_id % 100 == 0) {
            auto* trk = node->getTracking();
            auto* mapper = node->getSparseMapper();
            if (trk && mapper) {
                auto* covis = trk->getCovisManager();
                std::cout << "[SparseMapper] 帧=" << frame_id
                          << " KF=" << mapper->numKeyFrames()
                          << " 点=" << mapper->numMapPoints()
                          << " 确认=" << mapper->numConfirmedMapPoints() << "\n";
                std::string ply_path = "sfc_map_" + std::to_string(frame_id) + ".ply";
                mapper->saveMapPointsPLY(ply_path);
                std::string tum_path = "sfc_kf_" + std::to_string(frame_id) + ".tum";
                mapper->saveKeyFrameTrajectoryTUM(tum_path, 0.0, 0.033);
            }
        }
    }

    cv::destroyAllWindows();
    node->requestShutdown();                    // 阻止回调继续写入 disp_*
    rclcpp::shutdown();                         // 让 spin() 退出
    if (spin_thread.joinable()) spin_thread.join();
    std::cout << "ROS2 模式退出。累计 Matches: " << total_matches
              << "  Inliers: " << total_inliers << "\n";
}

// ============================================================================
// 离线图像文件夹模式 - 用于测试真实图像数据
// ============================================================================

#ifdef _WIN32
#include <windows.h>
#define sleep_ms(x) Sleep(x)
#else
#include <unistd.h>
#include <dirent.h>
#define sleep_ms(x) usleep((x) * 1000)
#endif

#include <sys/stat.h>

static bool fileExists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

static std::vector<std::string> listImageFiles(const std::string& dir) {
    std::vector<std::string> files;
    if (dir.empty()) return files;

    std::string search_dir = dir;
    if (search_dir.back() != '/' && search_dir.back() != '\\')
        search_dir += "/";

    const char* exts[] = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"};

#ifdef _WIN32
    WIN32_FIND_DATAW fdata;
    for (const char* ext : exts) {
        HANDLE hFind = FindFirstFileW((search_dir + "*" + ext).c_str(), &fdata);
        if (hFind == INVALID_HANDLE_VALUE) continue;
        do {
            std::wstring ws(fdata.cFileName);
            std::string s(ws.begin(), ws.end());
            if (s != "." && s != "..") {
                files.push_back(search_dir + s);
            }
        } while (FindNextFileW(hFind, &fdata));
        FindClose(hFind);
    }
#else
    DIR* dp = opendir(dir.c_str());
    if (!dp) return files;
    struct dirent* entry;
    while ((entry = readdir(dp))) {
        std::string name(entry->d_name);
        if (name == "." || name == "..") continue;
        for (const char* ext : exts) {
            if (name.length() >= strlen(ext) &&
                name.compare(name.length() - strlen(ext), strlen(ext), ext) == 0) {
                files.push_back(search_dir + name);
                break;
            }
        }
    }
    closedir(dp);
#endif

    std::sort(files.begin(), files.end());
    return files;
}

struct FrameData {
    cv::Mat rgb;
    cv::Mat depth;
    cv::Mat ir;
    double timestamp;
    std::string filename;
};

static bool loadImageDataset(const std::string& rgb_dir,
                            const std::string& depth_dir,
                            const std::string& ir_dir,
                            std::vector<FrameData>& frames) {

    std::cout << "[loadImageDataset]\n";
    std::cout << "  RGB: " << (rgb_dir.empty() ? "(none)" : rgb_dir) << "\n";
    std::cout << "  Depth: " << (depth_dir.empty() ? "(none)" : depth_dir) << "\n";
    std::cout << "  IR: " << (ir_dir.empty() ? "(none)" : ir_dir) << "\n";

    auto rgb_files = listImageFiles(rgb_dir);
    auto depth_files = listImageFiles(depth_dir);
    auto ir_files = listImageFiles(ir_dir);

    std::cout << "  Found RGB: " << rgb_files.size() << "\n";
    std::cout << "  Found Depth: " << depth_files.size() << "\n";
    std::cout << "  Found IR: " << ir_files.size() << "\n";

    if (rgb_files.empty()) {
        std::cerr << "[ERROR] No RGB images found!\n";
        return false;
    }

    // 按索引匹配
    for (size_t i = 0; i < rgb_files.size(); ++i) {
        FrameData fd;
        fd.rgb = cv::imread(rgb_files[i], cv::IMREAD_COLOR);
        if (fd.rgb.empty()) continue;

        fd.filename = rgb_files[i].substr(rgb_files[i].find_last_of("/\\") + 1);
        fd.timestamp = i / 30.0;  // 假设 30fps

        // 匹配 depth
        if (!depth_files.empty()) {
            size_t didx = i * depth_files.size() / rgb_files.size();
            if (didx < depth_files.size()) {
                fd.depth = cv::imread(depth_files[didx], cv::IMREAD_UNCHANGED);
                if (fd.depth.type() == CV_32FC1) {
                    cv::Mat converted;
                    fd.depth.convertTo(converted, CV_16UC1, 1000.0);
                    fd.depth = converted;
                }
            }
        }

        // 匹配 IR
        if (!ir_files.empty()) {
            size_t iidx = i * ir_files.size() / rgb_files.size();
            if (iidx < ir_files.size()) {
                fd.ir = cv::imread(ir_files[iidx], cv::IMREAD_GRAYSCALE);
            }
        }

        frames.push_back(fd);
    }

    std::cout << "  Loaded " << frames.size() << " frames\n";
    return !frames.empty();
}

static void runFolderMode(const std::string& rgb_dir,
                          const std::string& depth_dir,
                          const std::string& ir_dir,
                          const std::string& left_yaml,
                          const std::string& right_yaml,
                          const std::string& output_path,
                          int max_frames,
                          bool save_map,
                          double trajectory_scale = 1.0) {

    std::cout << "========================================\n";
    std::cout << "  Folder Mode - Offline Testing\n";
    std::cout << "========================================\n";

    // 加载相机内参
    CameraInfo rgb_info = loadCameraYaml(right_yaml);
    CameraInfo ir_info = loadCameraYaml(left_yaml);

    if (rgb_info.K.empty()) {
        std::cout << "[WARN] Using default camera intrinsics\n";
        rgb_info.K = cv::Mat(3, 3, CV_64F);
        rgb_info.K.at<double>(0, 0) = 547.89;
        rgb_info.K.at<double>(1, 1) = 549.98;
        rgb_info.K.at<double>(0, 2) = 322.76;
        rgb_info.K.at<double>(1, 2) = 272.82;
        rgb_info.K.at<double>(2, 2) = 1.0;
    }

    // 加载图像
    std::vector<FrameData> frames;
    if (!loadImageDataset(rgb_dir, depth_dir, ir_dir, frames)) {
        std::cerr << "[ERROR] Failed to load images\n";
        return;
    }

    // 配置 Tracking
    sfc_core::feature::ExtractorConfig ext_cfg;
    ext_cfg.n_features = 2000;
    ext_cfg.scale_factor = 1.2f;
    ext_cfg.n_levels = 8;
    ext_cfg.fast_threshold = 12;
    ext_cfg.edge_threshold = 12;
    ext_cfg.enable_uniform = true;
    ext_cfg.grid_cols = 10;
    ext_cfg.grid_rows = 8;
    ext_cfg.max_per_grid = 12;

    sfc_core::feature::MatcherConfig mat_cfg;
    mat_cfg.matcher_type = 0;
    mat_cfg.max_match_dist = 110.0f;
    mat_cfg.min_dist_ratio = 2.8f;
    mat_cfg.ransac_threshold = 5.0f;
    mat_cfg.lowe_ratio = 0.82f;
    mat_cfg.bidirectional_match = false;
    mat_cfg.min_dist_slack = 28.0f;

    sfc_core::frontend::TrackingConfig tcfg;
    tcfg.extractor_config = ext_cfg;
    tcfg.matcher_config = mat_cfg;
    tcfg.K = rgb_info.K;
    tcfg.D = rgb_info.D;
    tcfg.depth_scale = 0.001;
    tcfg.min_depth = 0.1;
    tcfg.max_depth = 8.0;
    tcfg.depth_edge_thresh = 0.2;
    tcfg.min_keyframe_translation = 0.02;
    tcfg.min_keyframe_rotation = 0.02;
    tcfg.min_features = 30;
    // R3：同上，去畸变由 D 是否为空决定，不再按"深度已配准"来关。
    tcfg.disable_undistort = tcfg.D.empty();

    auto tracking = std::make_unique<sfc_core::frontend::Tracking>(tcfg);

    // 初始化回环检测
    sfc_core::loopdetection::LoopDetectorConfig loop_cfg;
    loop_cfg.camera_K = rgb_info.K;
    loop_cfg.camera_D = rgb_info.D;
    loop_cfg.min_loop_separation = 10;
    loop_cfg.max_match_distance = 90;
    loop_cfg.min_geometric_inliers = 10;
    sfc_core::loopdetection::PoseGraphOptimizerOptions pgo_opts;
    pgo_opts.max_iterations = 50;
    tracking->initLoopDetection(loop_cfg, pgo_opts);

    // 初始化稀疏建图器
    sfc_core::mapping::SparseMapper sparse_mapper;

    std::vector<Eigen::Vector3d> trajectory;
    std::vector<Eigen::Quaterniond> trajectory_q;
    std::vector<double> timestamps;
    std::vector<int> inlier_counts;
    std::vector<bool> is_keyframe;

    int n = std::min((int)frames.size(), max_frames);
    std::cout << "[Processing] " << n << " frames...\n";

    for (int i = 0; i < n; ++i) {
        const auto& fd = frames[i];

        // 准备图像
        cv::Mat depth = fd.depth.empty() ? cv::Mat() : fd.depth.clone();
        cv::Mat ir = fd.ir.empty() ? cv::Mat() : fd.ir.clone();

        if (ir.empty()) {
            cv::cvtColor(fd.rgb, ir, cv::COLOR_BGR2GRAY);
        }

        // 执行跟踪
        auto pose = tracking->track(fd.rgb, depth, ir, fd.timestamp);

        // ========== SparseMapper 接入：同步关键帧与地图点到全局地图 ==========
        {
            auto* covis = tracking->getCovisManager();
            if (covis) {
                for (const auto& kf : covis->getAllKeyFrames()) {
                    if (!kf) continue;
                    bool found = false;
                    for (const auto& existing : sparse_mapper.getAllKeyFrames()) {
                        if (existing && existing->frame_id == kf->frame_id) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        sparse_mapper.addKeyFrame(kf);
                    }
                }
                for (const auto& kv : covis->getLastBAMapPointUpdates()) {
                    sparse_mapper.updateMapPoint(kv.first, kv.second);
                }
                for (const auto& kv : covis->getAllConfirmedMapPoints()) {
                    if (sparse_mapper.getMapPoint(kv.first) == nullptr) {
                        sparse_mapper.addMapPoint(kv.first, kv.second);
                    }
                }
            }
        }

        // 保存数据
        trajectory.push_back(pose.t);
        trajectory_q.push_back(pose.q);
        timestamps.push_back(fd.timestamp);
        inlier_counts.push_back(tracking->getLastInlierCount());

        // 检查是否为关键帧（第一帧或运动较大）
        bool kf = (i == 0);
        if (trajectory.size() >= 2) {
            double dt = (pose.t - trajectory[trajectory.size() - 2]).norm();
            if (dt > 0.02) kf = true;
        }
        is_keyframe.push_back(kf);

        if (i % 10 == 0) {
            std::cout << "  Frame " << i << "/" << n
                      << " inliers=" << inlier_counts.back()
                      << " kf=" << (kf ? "YES" : "NO")
                      << " pose=(" << std::fixed << std::setprecision(2)
                      << pose.t.x() << "," << pose.t.y() << "," << pose.t.z() << ")\n";
        }

        // 实时显示
        if (i % 5 == 0 && !fd.rgb.empty()) {
            cv::Mat vis = fd.rgb.clone();
            auto kps = tracking->getCurrentKeypoints();
            if (!kps.empty()) {
                cv::drawKeypoints(vis, kps, vis, cv::Scalar(0, 255, 100),
                                 cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
            }
            std::ostringstream ss;
            ss << "Frame: " << i << "/" << n;
            cv::putText(vis, ss.str(), cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            cv::imshow("Processing", vis);
#ifdef _WIN32
            cv::waitKey(30);  // Windows needs longer timeout for GUI event processing
#else
            cv::waitKey(1);
#endif
        }
    }

    cv::destroyAllWindows();

    // ========== SparseMapper 持久化：保存地图点云与轨迹 ==========
    if (save_map) {
        std::cout << "\n[SparseMapper] 关键帧=" << sparse_mapper.numKeyFrames()
                  << " 地图点=" << sparse_mapper.numMapPoints()
                  << " 确认点=" << sparse_mapper.numConfirmedMapPoints() << "\n";

        std::string ply_path = output_path;
        if (!ply_path.empty() && ply_path.size() > 4) {
            std::string base = ply_path.substr(0, ply_path.rfind('.'));
            if (!base.empty()) {
                std::string out_ply = base + "_map.ply";
                if (sparse_mapper.saveMapPointsPLY(out_ply)) {
                    std::cout << "  地图点云已保存: " << out_ply << "\n";
                } else {
                    std::cerr << "  地图点云保存失败: " << out_ply << "\n";
                }
                std::string out_tum = base + "_keyframe.tum";
                if (sparse_mapper.saveKeyFrameTrajectoryTUM(out_tum, 0.0, 0.033, trajectory_scale)) {
                    std::cout << "  关键帧轨迹已保存: " << out_tum << " (scale=" << trajectory_scale << ")\n";
                } else {
                    std::cerr << "  关键帧轨迹保存失败: " << out_tum << "\n";
                }
            }
        }
    } else {
        std::cout << "\n[SparseMapper] 地图保存已禁用（--no_save_map）\n";
    }

    // 统计分析
    std::cout << "\n========================================\n";
    std::cout << "  Results Summary\n";
    std::cout << "========================================\n";

    int kf_count = 0;
    for (bool kf : is_keyframe) if (kf) ++kf_count;
    std::cout << "  Total frames: " << trajectory.size() << "\n";
    std::cout << "  Keyframes: " << kf_count << "\n";

    if (trajectory.size() >= 2) {
        // 总路径长度
        double path_length = 0;
        for (size_t i = 1; i < trajectory.size(); ++i) {
            path_length += (trajectory[i] - trajectory[i-1]).norm();
        }
        std::cout << "  Path length: " << std::fixed << std::setprecision(3)
                  << path_length << " m\n";

        // 起终点漂移
        Eigen::Vector3d start = trajectory.front();
        Eigen::Vector3d end = trajectory.back();
        double drift = (end - start).norm();
        std::cout << "  Start: (" << start.x() << ", " << start.y() << ", " << start.z() << ")\n";
        std::cout << "  End:   (" << end.x() << ", " << end.y() << ", " << end.z() << ")\n";
        std::cout << "  Drift: " << drift << " m\n";

        if (path_length > 0.1) {
            double drift_ratio = drift / path_length;
            std::cout << "  Drift ratio: " << std::fixed << std::setprecision(1)
                      << drift_ratio * 100 << "%\n";

            if (drift_ratio > 0.1) {
                std::cout << "\n  [WARNING] Large drift detected!\n";
                std::cout << "  Check loop closure status.\n";
            } else {
                std::cout << "\n  [OK] Drift is acceptable.\n";
            }
        }
    }

    // 保存轨迹
    std::ofstream out(output_path);
    if (out.is_open()) {
        out << "# timestamp tx ty tz qx qy qz qw inliers is_keyframe\n";
        for (size_t i = 0; i < trajectory.size(); ++i) {
            out << std::fixed << std::setprecision(9) << timestamps[i] << " "
                << std::setprecision(6) << trajectory[i].x() << " "
                << trajectory[i].y() << " " << trajectory[i].z() << " "
                << trajectory_q[i].x() << " " << trajectory_q[i].y() << " "
                << trajectory_q[i].z() << " " << trajectory_q[i].w() << " "
                << inlier_counts[i] << " " << (is_keyframe[i] ? 1 : 0) << "\n";
        }
        out.close();
        std::cout << "\n  Trajectory saved: " << output_path << "\n";
    }

    // 可视化
    std::cout << "\n[Display] Final trajectory\n";
    cv::Mat traj_img = drawTrajectory(trajectory, 600, 600);
    cv::imshow("Trajectory", traj_img);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

// ============================================================================
// 主入口
// ============================================================================

void printUsage(const char* prog) {
    std::cout << "Usage:\n";
    std::cout << "  " << prog << " --ros2 [options]           # ROS2 图像话题模式\n";
    std::cout << "  " << prog << " --folder <path> [options]  # 离线图像文件夹模式\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --left_yaml  <path>      IR 相机内参 YAML 路径（默认: ../config/camera_yaml/left.yaml）\n";
    std::cout << "  --right_yaml <path>      RGB 相机内参 YAML 路径（默认: ../config/camera_yaml/right.yaml）\n";
    std::cout << "  --extrinsic_yaml <path>  双目外参 YAML 路径（默认: ../config/camera_yaml/stereo_extrinsics.yaml）\n";
    std::cout << "  --rgb_topic  <name>       RGB 图像话题（默认: /camera/color/image_raw）\n";
    std::cout << "  --ir_topic   <name>      IR  图像话题（默认: /camera/ir/image_raw）\n";
    std::cout << "  --depth_topic <name>     深度话题（默认: /camera/depth/image_raw）\n";
    std::cout << "  --mode       <mode>      跟踪模式: rgb_to_ir（默认）或 ir_to_rgb\n";
    std::cout << "\nFolder Mode Options:\n";
    std::cout << "  --rgb_dir    <path>      RGB 图像文件夹路径\n";
    std::cout << "  --depth_dir  <path>      Depth 图像文件夹路径\n";
    std::cout << "  --ir_dir     <path>      IR 图像文件夹路径\n";
    std::cout << "  --output     <path>      输出轨迹文件路径（默认: trajectory.txt）\n";
    std::cout << "  --max_frames <N>         最大处理帧数（默认: 1000）\n";
    std::cout << "  --save_map / --no_save_map  是否保存地图点云与关键帧轨迹（默认: 保存）\n";
    std::cout << "  --trajectory_scale <float> 轨迹事后尺度校正（默认: 1.0）\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << prog << " --ros2\n";
    std::cout << "  " << prog << " --ros2 --rgb_topic /usb_cam/image_raw --ir_topic /thermal/image_raw\n";
    std::cout << "  " << prog << " --folder --rgb_dir ./rgb --depth_dir ./depth --no_save_map\n";
    std::cout << "  " << prog << " --folder --rgb_dir ./rgb --depth_dir ./depth --output my_traj.txt --save_map\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 0;
    }

    std::string mode = argv[1];
    // ★★★ Bug Fix Q5: 默认 yaml_dir 改为相对路径，与 printUsage 文档一致 ★★★
    std::string yaml_dir = "../config/camera_yaml/";  // 相对于可执行文件所在目录
    std::string left_yaml  = yaml_dir + "left.yaml";
    std::string right_yaml = yaml_dir + "right.yaml";
    std::string extrinsic_yaml = yaml_dir + "stereo_extrinsics.yaml";
    std::string rgb_topic  = "/camera/color/image_raw";
    std::string ir_topic   = "/camera/ir/image_raw";
    std::string depth_topic = "/camera/depth/image_raw";
    bool rgb_to_ir = true;

    // 文件夹模式参数
    std::string rgb_dir;
    std::string depth_dir;
    std::string ir_dir;
    std::string output_path = "trajectory.txt";
    int max_frames = 1000;
    bool save_map = true;
    double trajectory_scale = 1.0;

    if (mode == "--folder") {
        // 解析文件夹模式参数
        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--left_yaml" && i + 1 < argc)
                left_yaml = argv[++i];
            else if (arg == "--right_yaml" && i + 1 < argc)
                right_yaml = argv[++i];
            else if (arg == "--rgb_dir" && i + 1 < argc)
                rgb_dir = argv[++i];
            else if (arg == "--depth_dir" && i + 1 < argc)
                depth_dir = argv[++i];
            else if (arg == "--ir_dir" && i + 1 < argc)
                ir_dir = argv[++i];
            else if (arg == "--output" && i + 1 < argc)
                output_path = argv[++i];
            else if (arg == "--max_frames" && i + 1 < argc)
                max_frames = std::atoi(argv[++i]);
            else if (arg == "--trajectory_scale" && i + 1 < argc)
                trajectory_scale = std::atof(argv[++i]);
            else if (arg == "--save_map")
                save_map = true;
            else if (arg == "--no_save_map")
                save_map = false;
        }

        if (rgb_dir.empty()) {
            std::cerr << "[ERROR] --rgb_dir is required for folder mode\n";
            printUsage(argv[0]);
            return 1;
        }

        runFolderMode(rgb_dir, depth_dir, ir_dir, left_yaml, right_yaml, output_path, max_frames, save_map, trajectory_scale);
        return 0;
    }

    if (mode != "--ros2") {
        printUsage(argv[0]);
        return 0;
    }

    // ROS2 模式参数解析
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--left_yaml" && i + 1 < argc)
            left_yaml = argv[++i];
        else if (arg == "--right_yaml" && i + 1 < argc)
            right_yaml = argv[++i];
        else if (arg == "--rgb_topic" && i + 1 < argc)
            rgb_topic = argv[++i];
        else if (arg == "--ir_topic" && i + 1 < argc)
            ir_topic = argv[++i];
        else if (arg == "--depth_topic" && i + 1 < argc)
            depth_topic = argv[++i];
        else if (arg == "--mode" && i + 1 < argc) {
            std::string m = argv[++i];
            rgb_to_ir = (m == "rgb_to_ir" || m == "rgb->ir");
        }
        else if (arg == "--extrinsic_yaml" && i + 1 < argc)
            extrinsic_yaml = argv[++i];
    }

    runROS2Mode(left_yaml, right_yaml, extrinsic_yaml, rgb_topic, ir_topic, depth_topic, rgb_to_ir);
    return 0;
}
