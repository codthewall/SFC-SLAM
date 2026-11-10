#include <iostream>
#include <string>
#include "sfc_core/feature/feature_extractor.hpp"
#include "sfc_core/feature/feature_matcher.hpp"
#include "sfc_core/utils/camera_utils.hpp"

using namespace sfc_core;

// 【关键】辅助函数：过滤无效匹配对
void filterInvalidMatches(const std::vector<cv::DMatch>& input,
                          const std::vector<cv::KeyPoint>& kps1,
                          const std::vector<cv::KeyPoint>& kps2,
                          std::vector<cv::DMatch>& output)
{
    output.clear();
    output.reserve(input.size());
    for (const auto& m : input)
    {
        // 检查索引是否在有效范围内
        if (m.queryIdx >= 0 && m.queryIdx < static_cast<int>(kps1.size()) &&
            m.trainIdx >= 0 && m.trainIdx < static_cast<int>(kps2.size()))
        {
            output.push_back(m);
        }
    }
}

int main(int argc, char** argv)
{
    // ========== 1. 准备测试数据 ==========
    std::string img1_path = "/home/syf/SFC-SLAM/picture/test1.png"; // 第一张图路径
    std::string img2_path = "/home/syf/SFC-SLAM/picture//test2.png"; // 第二张图路径

    if (argc == 3)
    {
        img1_path = argv[1];
        img2_path = argv[2];
    }

    cv::Mat img1_raw = cv::imread(img1_path, cv::IMREAD_GRAYSCALE);
    cv::Mat img2_raw = cv::imread(img2_path, cv::IMREAD_GRAYSCALE);

    if (img1_raw.empty() || img2_raw.empty())
    {
        std::cerr << "错误：无法读取图片，请检查路径！" << std::endl;
        return -1;
    }

    std::cout << "原始图片尺寸：图片1=" << img1_raw.size() << "，图片2=" << img2_raw.size() << std::endl;

    // 统一尺寸
    cv::Mat img1, img2;
    img1 = img1_raw;
    if (img2_raw.size() != img1.size())
    {
        cv::resize(img2_raw, img2, img1.size());
        std::cout << "已将图片2 resize到：" << img2.size() << std::endl;
    }
    else
    {
        img2 = img2_raw;
    }

    // ========== 2. 初始化模块 ==========
    feature::ExtractorConfig extract_config;
    extract_config.fast_threshold = 15;
    feature::FeatureExtractor extractor(extract_config);

    feature::MatcherConfig matcher_config;
    // 【关键修复1】测试时优先用暴力匹配，索引更稳定，不容易崩溃
    matcher_config.matcher_type = 1; // 0=暴力匹配(稳定)，1=FLANN(快速)
    matcher_config.min_dist_ratio = 2.0f;
    feature::FeatureMatcher matcher(matcher_config);

    // 模拟无畸变相机参数，避免干扰
    double fx = img1.cols * 1.0;
    double fy = img1.cols * 1.0;
    double cx = img1.cols / 2.0;
    double cy = img1.rows / 2.0;
    cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat D = (cv::Mat_<double>(1, 5) << 0.0, 0.0, 0.0, 0.0, 0.0);

    std::cout << "模块初始化完成（使用暴力匹配）" << std::endl;

    // ========== 3. 核心流程 ==========
    std::vector<cv::KeyPoint> kps1_raw, kps2_raw;
    cv::Mat des1, des2;

    extractor.extract(img1, kps1_raw, des1);
    extractor.extract(img2, kps2_raw, des2);

    std::cout << "特征提取完成：图片1=" << kps1_raw.size() << "，图片2=" << kps2_raw.size() << std::endl;

    if (kps1_raw.size() < 50 || kps2_raw.size() < 50)
    {
        std::cerr << "错误：特征点太少，请换纹理更丰富的图片！" << std::endl;
        return -1;
    }

    std::vector<cv::KeyPoint> kps1_undist, kps2_undist;
    utils::undistortKeypoints(kps1_raw, K, D, kps1_undist);
    utils::undistortKeypoints(kps2_raw, K, D, kps2_undist);

    std::vector<cv::DMatch> raw_matches;
    matcher.match(des1, des2, raw_matches);
    
    // 【关键修复2】第一次过滤：过滤掉无效索引的原始匹配
    std::vector<cv::DMatch> raw_matches_filtered;
    filterInvalidMatches(raw_matches, kps1_undist, kps2_undist, raw_matches_filtered);
    std::cout << "基础匹配完成：原始匹配=" << raw_matches.size() 
              << "，过滤后=" << raw_matches_filtered.size() << " 对" << std::endl;

    std::vector<cv::DMatch> inlier_matches;
    if (raw_matches_filtered.size() >= 8)
    {
        matcher.removeOutliers(raw_matches_filtered, kps1_undist, kps2_undist, inlier_matches);
        
        // 【关键修复3】第二次过滤：外点剔除后再次检查索引
        std::vector<cv::DMatch> inlier_matches_filtered;
        filterInvalidMatches(inlier_matches, kps1_undist, kps2_undist, inlier_matches_filtered);
        inlier_matches = std::move(inlier_matches_filtered);
    }
    std::cout << "外点剔除完成：内点匹配=" << inlier_matches.size() << " 对" << std::endl;

    // ========== 4. 可视化（加上空保护） ==========
    cv::Mat img1_color, img2_color;
    cv::cvtColor(img1, img1_color, cv::COLOR_GRAY2BGR);
    cv::cvtColor(img2, img2_color, cv::COLOR_GRAY2BGR);

    // 画原始匹配
    if (!raw_matches_filtered.empty())
    {
        cv::Mat img_raw_matches;
        cv::drawMatches(img1_color, kps1_undist, img2_color, kps2_undist, raw_matches_filtered, img_raw_matches,
                        cv::Scalar(0, 0, 255), cv::Scalar::all(-1),
                        std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::imshow("原始匹配（红色）- 按任意键继续", img_raw_matches);
    }
    else
    {
        cv::imshow("原始匹配（红色）- 无匹配", img1_color);
    }

    // 画内点匹配
    if (!inlier_matches.empty())
    {
        cv::Mat img_inlier_matches;
        cv::drawMatches(img1_color, kps1_undist, img2_color, kps2_undist, inlier_matches, img_inlier_matches,
                        cv::Scalar(0, 255, 0), cv::Scalar::all(-1),
                        std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::imshow("内点匹配（绿色）- 按任意键退出", img_inlier_matches);
    }
    else
    {
        std::cout << "提示：内点匹配为空，请换重叠区域更大的图片！" << std::endl;
        cv::imshow("内点匹配（绿色）- 无匹配", img2_color);
    }

    std::cout << "\n按任意键退出..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}