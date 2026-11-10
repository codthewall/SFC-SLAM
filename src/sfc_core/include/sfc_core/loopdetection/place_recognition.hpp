// 必须在最开头清除 CMake project() 注入的 sfc_core 宏
#ifdef sfc_core
#undef sfc_core
#endif

#ifndef SFC_CORE_LOOPDETECTION_PLACE_RECOGNITION_HPP_
#define SFC_CORE_LOOPDETECTION_PLACE_RECOGNITION_HPP_

#include "sfc_core/frontend/frame.hpp"
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <memory>
#include <random>
#include <set>
#include <atomic>

// OpenMP并行化支持
#ifdef _OPENMP
#include <omp.h>
#endif

namespace sfc_core {
namespace loopdetection {

// DBoW2-style place recognition using hierarchical vocabulary tree
// 
// This implements a simplified version of DBoW2 for loop detection:
// 1. Build a vocabulary tree offline from training images
// 2. For each keyframe, project descriptors into the tree and compute a TF-IDF weighted bow vector
// 3. For loop detection, compare bow vectors using L1 score and consistency check
//
// Key differences from simple BoW:
// - Hierarchical tree structure (not flat histogram)
// - TF-IDF weighting (Term Frequency - Inverse Document Frequency)
// - Direct index for fast geometric verification

struct PlaceRecognitionConfig {
    // 词汇树参数
    int vocab_build_threshold = 30;          // 建树阈值（关键帧数）
    int vocab_branching_factor = 10;          // 分支因子 k
    int vocab_depth = 6;                      // 树深度 L
    int kmeans_max_iterations = 20;          // K-means 最大迭代次数
    double new_word_distance_threshold = 25.0; // 新词距离阈值

    // 困难样本参数
    int hard_example_window = 100;            // 困难样本窗口大小
    double hard_threshold = 50.0;              // 困难样本硬阈值（匹配距离）

    // 时序一致性
    int temporal_window = 3;                  // 时序一致性窗口

    // 在线学习
    bool enable_online_learning = false;      // 是否启用在线学习
    int idf_update_interval = 10;            // IDF 权重更新间隔
    int hard_example_learn_interval = 50;     // 困难样本学习间隔
    int min_hard_examples_to_learn = 20;      // 触发学习的最小困难样本数
    int max_new_words_per_learn = 50;        // 每次学习最多新增词汇数

    // TF-IDF
    bool use_tfidf = true;                   // 是否使用 TF-IDF 权重
    bool normalize_l1 = true;                 // 是否使用 L1 范数归一化

    // 候选评分
    double consistency_bonus = 0.2;           // 时序一致性奖励权重
    double min_candidate_score = 0.0;         // 候选评分最小值
};

struct VocabNode {
    int node_id = -1;
    int parent_id = -1;
    std::vector<int> children_ids;
    cv::Mat descriptor;           // Centroid ORB descriptor
    std::vector<int> word_ids;   // ORB descriptors in this node
    double weight = 1.0;         // IDF weight
    bool is_leaf = false;
};

class PlaceRecognition {
public:
    PlaceRecognition() : vocab_ready_(false), vocab_build_threshold_(30) {}

    // 从配置初始化 PlaceRecognition
    explicit PlaceRecognition(const PlaceRecognitionConfig& config) {
        vocab_ready_ = false;
        vocab_build_threshold_ = config.vocab_build_threshold;
        k_ = config.vocab_branching_factor;
        L_ = config.vocab_depth;
        hard_example_window_ = config.hard_example_window;
        new_word_threshold_ = config.new_word_distance_threshold;
        incremental_vocab_enabled_ = config.enable_online_learning;
        // 注：consistency_bonus 和 min_candidate_score 在 queryWithConsistency / getBestCandidate 中硬编码
        // 可通过 YAML 配置文件中的 place_recognition.yaml 参数化
    }

    // Check if vocabulary is ready
    bool isVocabularyReady() const { return vocab_ready_; }

    // Get the threshold for building vocabulary
    int getBuildThreshold() const { return vocab_build_threshold_; }

    // Check if we have enough descriptors to build vocabulary
    bool isReadyToBuild() const { 
        return !vocab_ready_ && training_descriptors_.size() >= static_cast<size_t>(vocab_build_threshold_); 
    }

    // Collect training descriptors for online vocabulary building
    void collectForVocabulary(const cv::Mat& descriptors) {
        if (vocab_ready_) return;
        
        // 收集所有描述子以获得更好的词袋
        for (int i = 0; i < descriptors.rows; i++) {
            training_descriptors_.push_back(descriptors.row(i).clone());
        }
        
        if (training_descriptors_.size() % 500 == 0 || 
            static_cast<int>(training_descriptors_.size()) >= vocab_build_threshold_) {
            std::cerr << "[PlaceRecog] 收集描述子: " << training_descriptors_.size() 
                      << "/" << vocab_build_threshold_ << std::endl;
        }
    }

    // Build vocabulary from collected training descriptors
    // Returns true if vocabulary was built
    bool tryBuildVocabulary(int k = 10, int L = 6) {
        if (vocab_ready_) return true;
        
        if (static_cast<int>(training_descriptors_.size()) < vocab_build_threshold_) {
            return false;
        }
        
        std::cerr << "[PlaceRecog] ★★★ 开始构建词袋树 ★★★" << std::endl;
        std::cerr << "[PlaceRecog] 使用 " << training_descriptors_.size() 
                  << " 个描述子构建词袋 (k=" << k << ", L=" << L << ")" << std::endl;
        
        buildVocabulary(training_descriptors_, k, L);
        vocab_ready_ = true;
        
        // 释放训练数据以节省内存
        std::vector<cv::Mat>().swap(training_descriptors_);
        
        std::cerr << "[PlaceRecog] 词袋构建完成！" << std::endl;
        return true;
    }

    // ★★★ 词袋增量更新接口 ★★★
    // 用于长期运行时动态更新词袋，提高对新场景的召回率
    
    // 增量添加新词：当新描述子与现有所有词的距离都大于阈值时，创建新词
    void enableIncrementalVocabulary(double new_word_distance_threshold = 25.0);
    
    // 更新已有词的IDF权重（考虑新添加的关键帧）
    void updateIDFWeightsIncremental();
    
    // 检查词袋是否启用了增量更新
    bool isIncrementalVocabularyEnabled() const { return incremental_vocab_enabled_; }
    
    // 获取新添加的词数量
    int getNewWordsCount() const { return new_words_added_; }
    
    // 收集"困难"描述子用于增量学习（匹配距离较大的描述子）
    void collectHardExamples(const cv::Mat& descriptors, 
                             const std::vector<cv::DMatch>& matches,
                             int kf_id);
    
    // 从收集的困难样本中学习新词
    void learnFromHardExamples(int max_new_words = 50);

    // Build vocabulary from training descriptors
    void buildVocabulary(const std::vector<cv::Mat>& training_descriptors,
                         int k = 10, int L = 6);

    // Project a descriptor into the vocabulary, returns word id (leaf node)
    int project(const cv::Mat& descriptor) const;

    // Compute BoW vector for a set of descriptors (TF-IDF weighted)
    std::vector<std::pair<int, double>> computeBoWVector(const cv::Mat& descriptors) const;

    // Add a keyframe to the database with its BoW vector
    void addKeyFrame(int kf_id, const cv::Mat& descriptors, 
                     const std::vector<cv::KeyPoint>& keypoints,
                     const Eigen::Vector3d& position);

    // Query for similar keyframes using bag-of-words scoring
    // Returns vector of (kf_id, score, kf_position) sorted by score
    struct QueryResult {
        int kf_id;
        double score;
        Eigen::Vector3d position;
        double distance;  // Distance from current position
    };
    std::vector<QueryResult> query(const cv::Mat& descriptors,
                                   const Eigen::Vector3d& current_position,
                                   int max_results = 10,
                                   double min_distance_m = 1.0) const;

    // Query with temporal consistency check (DBoW2 style)
    // A match is only valid if it's consistent with recent matches
    std::vector<QueryResult> queryWithConsistency(const cv::Mat& descriptors,
                                                  const Eigen::Vector3d& current_position,
                                                  const std::vector<int>& recent_matches,
                                                  int temporal_window = 3,
                                                  int max_results = 10,
                                                  double min_distance_m = 1.0) const;

    // Get the best candidate that passes all checks
    QueryResult getBestCandidate(const cv::Mat& descriptors,
                                 const Eigen::Vector3d& current_position,
                                 const std::vector<int>& recent_matches,
                                 double min_distance_m = 1.0,
                                 double min_score = 0.05) const;

    // Clear the database
    void clear() { database_.clear(); }

    int numKeyFrames() const { return database_.size(); }

private:
    // Clustering using k-means
    void kMeansClustering(std::vector<cv::Mat>& descriptors, int k, 
                         std::vector<cv::Mat>& centroids);

    // Build tree structure from centroids
    void buildTreeRecursive(std::vector<cv::Mat>& descriptors, 
                            int parent_id, int level);

    // Compute IDF weights from database
    void computeIDFWeights();

    // Online vocabulary building
    bool vocab_ready_ = false;
    int vocab_build_threshold_ = 30;  // 收集 30 个关键帧后构建词袋
    std::vector<cv::Mat> training_descriptors_;

    // Vocabulary tree
    std::vector<VocabNode> vocab_tree_;
    int k_ = 10;           // Branching factor
    int L_ = 6;           // Tree depth
    int word_count_ = 0;  // Total number of words (leaves)

    // Database of keyframes
    struct DatabaseEntry {
        int kf_id;
        std::vector<std::pair<int, double>> bow_vector;  // (word_id, weight)
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        Eigen::Vector3d position;
        std::vector<int> word_ids;  // Direct index for fast geometric verification
    };
    std::vector<DatabaseEntry> database_;
    std::unordered_map<int, int> kf_id_to_idx_;  // kf_id -> database index

    // IDF: Inverse Document Frequency
    // N / ni where N = total docs, ni = docs containing word i
    std::vector<double> idf_;
    
    // ★★★ 词袋增量更新私有成员 ★★★
    bool incremental_vocab_enabled_ = false;          // 是否启用增量词袋
    double new_word_threshold_ = 25.0;               // 新词距离阈值
    int new_words_added_ = 0;                         // 新增词数量
    int kf_count_for_idf_ = 0;                       // 用于IDF计算的关键帧数量
    
    // 困难样本收集（用于增量学习）
    struct HardExample {
        cv::Mat descriptor;
        int kf_id;
        double match_distance;
    };
    std::vector<HardExample> hard_examples_;          // 困难样本集合
    int hard_example_window_ = 100;                   // 保留的困难样本窗口
    
    // 词到关键帧的倒排索引（用于IDF更新）
    std::unordered_map<int, std::vector<int>> word_to_kf_;
};

// ============================================================
// PlaceRecognition Implementation
// ============================================================

inline void PlaceRecognition::buildVocabulary(const std::vector<cv::Mat>& training_descriptors,
                                              int k, int L) {
    k_ = k;
    L_ = L;
    
    std::cerr << "[PlaceRecog] Building vocabulary: k=" << k << " L=" << L << std::endl;
    
    // Flatten all descriptors for clustering
    std::vector<cv::Mat> all_descriptors;
    for (const auto& desc : training_descriptors) {
        for (int i = 0; i < desc.rows; ++i) {
            all_descriptors.push_back(desc.row(i).clone());
        }
    }
    
    std::cerr << "[PlaceRecog] Training with " << all_descriptors.size() << " descriptors" << std::endl;
    
    if (all_descriptors.size() < static_cast<size_t>(k)) {
        std::cerr << "[PlaceRecog] Not enough descriptors for vocabulary building" << std::endl;
        return;
    }
    
    // Initialize root node
    vocab_tree_.clear();
    vocab_tree_.push_back(VocabNode{0, -1, {}, cv::Mat(), {}, 1.0, false});
    
    // Build tree recursively using k-means
    std::vector<cv::Mat> current_level_descriptors = all_descriptors;
    std::vector<int> current_level_nodes = {0};
    
    for (int level = 0; level < L; ++level) {
        std::vector<int> next_level_nodes;
        next_level_nodes.reserve(current_level_nodes.size() * k);
        
        for (size_t ni = 0; ni < current_level_nodes.size(); ++ni) {
            int parent_id = current_level_nodes[ni];
            
            // Determine cluster count: use min(k, size) but at least 2 for valid clustering
            int effective_k = std::min(k, std::max(2, static_cast<int>(current_level_descriptors.size())));
            
            // K-means clustering
            std::vector<cv::Mat> centroids;
            kMeansClustering(current_level_descriptors, effective_k, centroids);
            
            if (centroids.empty()) {
                // 无法聚类，创建叶子节点
                vocab_tree_[parent_id].children_ids.push_back(vocab_tree_.size());
                vocab_tree_.push_back(VocabNode{
                    static_cast<int>(vocab_tree_.size()),
                    parent_id, {}, current_level_descriptors.empty() ? cv::Mat() : current_level_descriptors[0],
                    {}, 1.0, true
                });
                continue;
            }
            
            // Assign each descriptor to nearest centroid
            std::vector<std::vector<cv::Mat>> cluster_descriptors(centroids.size());
            for (const auto& desc : current_level_descriptors) {
                int best_cluster = 0;
                int best_dist = 999999;
                for (int c = 0; c < static_cast<int>(centroids.size()); ++c) {
                    if (desc.size() != centroids[c].size()) continue;
                    int dist = cv::norm(desc, centroids[c], cv::NORM_HAMMING);
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_cluster = c;
                    }
                }
                cluster_descriptors[best_cluster].push_back(desc);
            }
            
            // Create child nodes
            for (int c = 0; c < static_cast<int>(centroids.size()); ++c) {
                int child_id = static_cast<int>(vocab_tree_.size());
                vocab_tree_[parent_id].children_ids.push_back(child_id);
                
                bool is_leaf = (level == L - 1) || cluster_descriptors[c].empty();
                vocab_tree_.push_back(VocabNode{
                    child_id, parent_id, {}, centroids[c], {},
                    1.0, is_leaf
                });
                
                if (!is_leaf) {
                    next_level_nodes.push_back(child_id);
                }
            }
        }
        
        current_level_nodes = next_level_nodes;
    }
    
    word_count_ = 0;
    for (auto& node : vocab_tree_) {
        if (node.is_leaf) word_count_++;
    }
    
    // Initialize IDF
    idf_.resize(word_count_, 1.0);
    
    std::cerr << "[PlaceRecog] Vocabulary built: " << vocab_tree_.size() << " nodes, "
              << word_count_ << " words" << std::endl;
}

inline void PlaceRecognition::kMeansClustering(std::vector<cv::Mat>& descriptors,
                                               int k, std::vector<cv::Mat>& centroids) {
    if (descriptors.empty()) return;
    
    // 如果描述子数量少于 k，直接复制
    if (static_cast<int>(descriptors.size()) <= k) {
        centroids.clear();
        for (const auto& d : descriptors) {
            centroids.push_back(d.clone());
        }
        return;
    }
    
    const int max_iterations = 20;
    
    // Initialize centroids using k-means++
    if (descriptors.empty()) return;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(descriptors.size()) - 1);
    
    centroids.clear();
    centroids.reserve(k);
    
    // First centroid: random
    centroids.push_back(descriptors[dist(rng)].clone());
    
    // Remaining centroids: weighted by distance squared
    for (int iter = 1; iter < k; ++iter) {
        std::vector<double> distances(descriptors.size(), 0.0);
        double total_dist = 0.0;
        
        for (size_t i = 0; i < descriptors.size(); ++i) {
            double min_dist = 1e10;
            for (const auto& c : centroids) {
                double d = cv::norm(descriptors[i], c, cv::NORM_HAMMING);
                if (d < min_dist) min_dist = d;
            }
            distances[i] = min_dist * min_dist;
            total_dist += distances[i];
        }
        
        // Pick next centroid with probability proportional to distance squared
        std::uniform_real_distribution<double> prob_dist(0.0, std::max(total_dist, 1e-10));
        double p = prob_dist(rng);
        double cumsum = 0.0;
        for (size_t i = 0; i < descriptors.size(); ++i) {
            cumsum += distances[i];
            if (cumsum >= p) {
                centroids.push_back(descriptors[i].clone());
                break;
            }
        }
        
        if (static_cast<int>(centroids.size()) < iter + 1) {
            centroids.push_back(descriptors[dist(rng)].clone());
        }
    }
    
    // Run k-means iterations
    std::vector<int> labels(descriptors.size(), 0);
    
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        // Assign descriptors to nearest centroid
        std::fill(labels.begin(), labels.end(), 0);
        bool changed = false;
        
        for (size_t i = 0; i < descriptors.size(); ++i) {
            int best_cluster = 0;
            int best_dist = 999999;
            for (int c = 0; c < static_cast<int>(centroids.size()); ++c) {
                int dist = cv::norm(descriptors[i], centroids[c], cv::NORM_HAMMING);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_cluster = c;
                }
            }
            if (labels[i] != best_cluster) {
                labels[i] = best_cluster;
                changed = true;
            }
        }
        
        if (!changed) break;
        
        // Update centroids: for binary descriptors, use median (majority bit)
        for (int c = 0; c < static_cast<int>(centroids.size()); ++c) {
            std::vector<cv::Mat> cluster_descs;
            for (size_t i = 0; i < descriptors.size(); ++i) {
                if (labels[i] == c) {
                    cluster_descs.push_back(descriptors[i]);
                }
            }
            
            if (cluster_descs.empty()) {
                // 空簇，随机选一个
                centroids[c] = descriptors[dist(rng)].clone();
            } else if (cluster_descs.size() == 1) {
                centroids[c] = cluster_descs[0].clone();
            } else {
                // 计算众数位 (majority vote per bit)
                cv::Mat new_centroid = cluster_descs[0].clone();
                for (int byte_idx = 0; byte_idx < new_centroid.cols; ++byte_idx) {
                    int sum = 0;
                    for (const auto& d : cluster_descs) {
                        sum += d.at<uchar>(0, byte_idx);
                    }
                    // 如果超过一半的描述子该位为1，则中心点该位为1
                    new_centroid.at<uchar>(0, byte_idx) = (sum > static_cast<int>(cluster_descs.size() / 2)) ? 255 : 0;
                }
                centroids[c] = new_centroid;
            }
        }
    }
}

inline int PlaceRecognition::project(const cv::Mat& descriptor) const {
    if (vocab_tree_.empty() || vocab_tree_[0].is_leaf) return 0;
    
    int node_id = 0;  // Start from root
    
    while (!vocab_tree_[node_id].is_leaf) {
        const auto& children = vocab_tree_[node_id].children_ids;
        if (children.empty()) break;
        
        int best_child = children[0];
        int best_dist = 999999;
        
        for (int child_id : children) {
            if (vocab_tree_[child_id].descriptor.empty()) continue;
            if (descriptor.size() != vocab_tree_[child_id].descriptor.size()) continue;
            int dist = cv::norm(descriptor, vocab_tree_[child_id].descriptor, cv::NORM_HAMMING);
            if (dist < best_dist) {
                best_dist = dist;
                best_child = child_id;
            }
        }
        
        node_id = best_child;
    }
    
    return node_id;
}

inline std::vector<std::pair<int, double>>
PlaceRecognition::computeBoWVector(const cv::Mat& descriptors) const {
    if (descriptors.rows == 0) return {};

    // 优化：OpenMP并行投影每个描述子到词袋树
    // vocab_tree_是只读的，可以安全地在多线程中访问
    std::vector<int> word_ids(descriptors.rows);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if (descriptors.rows > 50)
    #endif
    for (int i = 0; i < descriptors.rows; ++i) {
        word_ids[i] = project(descriptors.row(i));
    }

    // 串行聚合：合并词频统计（O(n_words)，数量级远小于n_descriptors）
    std::unordered_map<int, int> word_counts;
    word_counts.reserve(descriptors.rows / 10);
    for (int i = 0; i < descriptors.rows; ++i) {
        word_counts[word_ids[i]]++;
    }

    // Convert to TF-IDF weighted vector
    double total = descriptors.rows;
    std::vector<std::pair<int, double>> bow_vector;
    bow_vector.reserve(word_counts.size());
    for (const auto& kv : word_counts) {
        double tf = kv.second / total;
        double idf = idf_[kv.first];
        double weight = tf * idf;
        bow_vector.emplace_back(kv.first, weight);
    }

    return bow_vector;
}

inline void PlaceRecognition::addKeyFrame(int kf_id, const cv::Mat& descriptors,
                                         const std::vector<cv::KeyPoint>& keypoints,
                                         const Eigen::Vector3d& position) {
    DatabaseEntry entry;
    entry.kf_id = kf_id;
    entry.bow_vector = computeBoWVector(descriptors);
    entry.keypoints = keypoints;
    entry.descriptors = descriptors.clone();
    entry.position = position;
    
    // Build direct index for fast geometric verification
    // 优化：并行投影所有描述子（描述子数量通常数百个，值得并行化）
    entry.word_ids.resize(descriptors.rows);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if (descriptors.rows > 50)
    #endif
    for (int i = 0; i < descriptors.rows; ++i) {
        entry.word_ids[i] = project(descriptors.row(i));
    }

    // 串行：更新倒排索引（需要写操作，不可在多线程中直接并行）
    if (incremental_vocab_enabled_) {
        for (int word_id : entry.word_ids) {
            word_to_kf_[word_id].push_back(kf_id);
        }
    }
    
    int idx = database_.size();
    database_.push_back(entry);
    kf_id_to_idx_[kf_id] = idx;
    
    // Update IDF weights
    if (idf_.empty()) {
        idf_.resize(word_count_, 1.0);
    }
    
    // ★ 增量更新：定期更新IDF权重（每10个关键帧）
    if (incremental_vocab_enabled_ && database_.size() % 10 == 0) {
        updateIDFWeightsIncremental();
        
        // ★ 定期从困难样本学习新词（每50个关键帧）
        if (database_.size() % 50 == 0) {
            learnFromHardExamples(20);
        }
    }
}

inline std::vector<PlaceRecognition::QueryResult>
PlaceRecognition::query(const cv::Mat& descriptors,
                        const Eigen::Vector3d& current_position,
                        int max_results,
                        double min_distance_m) const {
    std::vector<QueryResult> results;

    if (database_.empty()) return results;

    // Compute query BoW vector
    auto query_bow = computeBoWVector(descriptors);

    // Build query weight map for faster lookup
    std::unordered_map<int, double> query_weights;
    double query_norm = 0.0;
    for (const auto& kv : query_bow) {
        query_weights[kv.first] = kv.second;
        query_norm += kv.second * kv.second;
    }
    query_norm = std::sqrt(query_norm);

    // ★★★ 优化 6：预计算所有数据库条目的 L2 范数（避免每轮循环重复计算）★★★
    std::vector<double> db_norms;
    db_norms.reserve(database_.size());
    for (const auto& entry : database_) {
        double n = 0.0;
        for (const auto& kv : entry.bow_vector) n += kv.second * kv.second;
        db_norms.push_back(std::sqrt(n));
    }

    // Score against all database entries
    struct RawScore {
        int idx;
        double score;
    };
    std::vector<RawScore> raw_scores;
    raw_scores.reserve(database_.size());

    for (size_t di = 0; di < database_.size(); ++di) {
        const auto& entry = database_[di];
        // L1 score (sum of min weights)
        double score = 0.0;
        for (const auto& kv : query_bow) {
            for (const auto& db_kv : entry.bow_vector) {
                if (kv.first == db_kv.first) {
                    score += std::min(kv.second, db_kv.second);
                    break;
                }
            }
        }

        double db_norm = db_norms[di];

        if (query_norm > 0 && db_norm > 0) {
            score = 2.0 * score / (query_norm + db_norm);
        } else {
            score = 0.0;
        }

        // Check minimum distance
        double dist = (entry.position - current_position).norm();
        if (dist >= min_distance_m) {
            raw_scores.push_back({static_cast<int>(di), score});
        }
    }

    // Sort by score
    std::sort(raw_scores.begin(), raw_scores.end(),
              [](const RawScore& a, const RawScore& b) {
                  return a.score > b.score;
              });

    // Take top results
    for (int i = 0; i < std::min(max_results, static_cast<int>(raw_scores.size())); ++i) {
        const auto& entry = database_[raw_scores[i].idx];
        QueryResult result;
        result.kf_id = entry.kf_id;
        result.score = raw_scores[i].score;
        result.position = entry.position;
        result.distance = (entry.position - current_position).norm();
        results.push_back(result);
    }

    return results;
}

inline std::vector<PlaceRecognition::QueryResult>
PlaceRecognition::queryWithConsistency(const cv::Mat& descriptors,
                                         const Eigen::Vector3d& current_position,
                                         const std::vector<int>& recent_matches,
                                         int temporal_window,
                                         int max_results,
                                         double min_distance_m) const {
    // First get regular query results
    auto results = query(descriptors, current_position, max_results * 3, min_distance_m);
    
    if (recent_matches.empty()) {
        results.resize(std::min(max_results, static_cast<int>(results.size())));
        return results;
    }
    
    // Build consistency set from recent matches
    std::set<int> consistency_set(recent_matches.begin(), recent_matches.end());
    
    // Score by consistency: matches that agree with recent matches get bonus
    for (auto& result : results) {
        // Find entry
        auto it = kf_id_to_idx_.find(result.kf_id);
        if (it == kf_id_to_idx_.end()) continue;
        
        const auto& entry = database_[it->second];
        
        // Count how many recent matches share words with this entry
        int consistency_count = 0;
        for (int recent_kf : recent_matches) {
            auto recent_it = kf_id_to_idx_.find(recent_kf);
            if (recent_it == kf_id_to_idx_.end()) continue;
            
            const auto& recent_entry = database_[recent_it->second];
            
            // Count shared words
            std::set<int> recent_words;
            for (int w : recent_entry.word_ids) recent_words.insert(w);
            
            int shared = 0;
            for (int w : entry.word_ids) {
                if (recent_words.count(w)) shared++;
            }
            
            if (shared > 5) consistency_count++;
        }
        
        // Consistency bonus
        result.score += 0.2 * consistency_count;
    }
    
    // Re-sort by updated score
    std::sort(results.begin(), results.end(),
              [](const QueryResult& a, const QueryResult& b) {
                  return a.score > b.score;
              });
    
    results.resize(std::min(max_results, static_cast<int>(results.size())));
    return results;
}

inline PlaceRecognition::QueryResult
PlaceRecognition::getBestCandidate(const cv::Mat& descriptors,
                                    const Eigen::Vector3d& current_position,
                                    const std::vector<int>& recent_matches,
                                    double min_distance_m,
                                    double min_score) const {
    auto results = queryWithConsistency(descriptors, current_position, recent_matches, 
                                        3, 5, min_distance_m);
    
    if (!results.empty() && results[0].score >= min_score) {
        return results[0];
    }
    
    return QueryResult{-1, 0.0, Eigen::Vector3d::Zero(), 0.0};
}

// =============================================================================
// 词袋增量更新实现
// =============================================================================

inline void PlaceRecognition::enableIncrementalVocabulary(double new_word_distance_threshold) {
    if (!vocab_ready_) {
        std::cerr << "[PlaceRecog] 词袋未就绪，无法启用增量更新" << std::endl;
        return;
    }
    
    incremental_vocab_enabled_ = true;
    new_word_threshold_ = new_word_distance_threshold;
    
    // 初始化倒排索引
    word_to_kf_.clear();
    for (const auto& entry : database_) {
        for (int word_id : entry.word_ids) {
            word_to_kf_[word_id].push_back(entry.kf_id);
        }
    }
    kf_count_for_idf_ = database_.size();
    
    std::cerr << "[PlaceRecog] 启用词袋增量更新，新词阈值=" << new_word_threshold_ << std::endl;
}

inline void PlaceRecognition::updateIDFWeightsIncremental() {
    if (!vocab_ready_ || database_.empty()) return;
    
    int N = static_cast<int>(database_.size());
    
    std::cerr << "[PlaceRecog] 更新IDF权重: N=" << N << std::endl;
    
    // 遍历所有词，更新IDF
    for (auto& kv : word_to_kf_) {
        int word_id = kv.first;
        if (word_id >= static_cast<int>(idf_.size())) continue;
        
        // ni = 包含该词的关键帧数量
        int ni = static_cast<int>(kv.second.size());
        
        // IDF = log(N / ni)
        // 当词在所有文档中出现时，IDF=0（无区分度）
        // 当词仅在很少文档中出现时，IDF值高（高区分度）
        if (ni > 0) {
            idf_[word_id] = std::log(static_cast<double>(N) / ni);
        } else {
            // 新词或未出现的词，赋予中等权重
            idf_[word_id] = 1.0;
        }
    }
    
    // 对空词（从未出现的词）赋予最大权重
    for (size_t i = 0; i < idf_.size(); ++i) {
        if (word_to_kf_.find(static_cast<int>(i)) == word_to_kf_.end()) {
            idf_[i] = std::log(static_cast<double>(N) + 1.0);
        }
    }
    
    kf_count_for_idf_ = N;
    std::cerr << "[PlaceRecog] IDF权重更新完成" << std::endl;
}

inline void PlaceRecognition::collectHardExamples(const cv::Mat& descriptors,
                                                   const std::vector<cv::DMatch>& matches,
                                                   int kf_id) {
    if (!incremental_vocab_enabled_) return;
    if (descriptors.empty()) return;
    
    // 收集匹配距离较大的描述子作为困难样本
    const double hard_threshold = 50.0;  // 匹配距离大于此值认为是困难样本
    
    for (const auto& m : matches) {
        if (m.distance > hard_threshold) {
            HardExample ex;
            ex.descriptor = descriptors.row(m.queryIdx).clone();
            ex.kf_id = kf_id;
            ex.match_distance = m.distance;
            hard_examples_.push_back(ex);
        }
    }
    
    // 限制困难样本数量
    if (static_cast<int>(hard_examples_.size()) > hard_example_window_) {
        // 删除最老的样本
        hard_examples_.erase(hard_examples_.begin(), 
                           hard_examples_.begin() + (hard_examples_.size() - hard_example_window_));
    }
    
    if (hard_examples_.size() % 100 == 0 && !hard_examples_.empty()) {
        std::cerr << "[PlaceRecog] 收集困难样本: " << hard_examples_.size() << " 个" << std::endl;
    }
}

inline void PlaceRecognition::learnFromHardExamples(int max_new_words) {
    if (!incremental_vocab_enabled_) return;
    if (hard_examples_.empty()) return;
    if (hard_examples_.size() < 20) return;  // 至少需要20个样本
    
    std::cerr << "[PlaceRecog] 从 " << hard_examples_.size() << " 个困难样本学习新词..." << std::endl;
    
    // 提取所有困难描述子
    std::vector<cv::Mat> hard_descriptors;
    hard_descriptors.reserve(hard_examples_.size());
    for (const auto& ex : hard_examples_) {
        hard_descriptors.push_back(ex.descriptor.clone());
    }
    
    // 对困难样本进行聚类，创建新词
    int new_words = 0;
    for (int i = 0; i < max_new_words && new_words < 10; ++i) {
        // 从未聚类的描述子开始
        std::vector<cv::Mat> remaining;
        std::vector<int> labels;
        
        if (hard_descriptors.empty()) break;
        
        // 简单的k-means聚类
        std::vector<cv::Mat> centroids;
        kMeansClustering(hard_descriptors, 3, centroids);  // 每次分成3簇
        
        if (centroids.empty()) break;
        
        // 选择每个簇的中心作为新词
        for (const auto& centroid : centroids) {
            if (centroid.empty()) continue;
            
            // 检查这个候选词与现有所有词的距离
            bool is_new = true;
            for (const auto& node : vocab_tree_) {
                if (node.is_leaf && !node.descriptor.empty()) {
                    int dist = cv::norm(centroid, node.descriptor, cv::NORM_HAMMING);
                    if (dist < new_word_threshold_) {
                        is_new = false;
                        break;
                    }
                }
            }
            
            if (is_new) {
                // 创建新词节点
                int new_word_id = static_cast<int>(vocab_tree_.size());
                
                // 找到根节点的第一个孩子作为父节点（简化处理）
                // 实际上应该找到最相似的内部节点作为父节点
                int parent_id = 0;
                int dist_to_parent = 999999;
                if (!vocab_tree_.empty() && !vocab_tree_[0].children_ids.empty()) {
                    // 选择与新词距离最小的孩子作为父节点
                    int best_parent = vocab_tree_[0].children_ids[0];
                    int best_dist = 999999;
                    for (int child_id : vocab_tree_[0].children_ids) {
                        if (vocab_tree_[child_id].descriptor.empty()) continue;
                        int dist = cv::norm(centroid, vocab_tree_[child_id].descriptor, cv::NORM_HAMMING);
                        if (dist < best_dist) {
                            best_dist = dist;
                            best_parent = child_id;
                        }
                    }
                    parent_id = best_parent;
                    dist_to_parent = best_dist;
                }
                
                vocab_tree_[parent_id].children_ids.push_back(new_word_id);
                vocab_tree_.push_back(VocabNode{
                    new_word_id, parent_id, {}, centroid, {},
                    1.0, true  // 新节点是叶子
                });
                
                // 更新IDF向量
                if (new_word_id >= static_cast<int>(idf_.size())) {
                    idf_.resize(new_word_id + 1, std::log(static_cast<double>(kf_count_for_idf_ + 1)));
                }
                
                word_to_kf_[new_word_id] = {};  // 新词暂无关键帧
                new_words++;
                
                std::cerr << "[PlaceRecog] 添加新词 #" << new_word_id 
                         << " (与最近词距离=" << dist_to_parent << ")" << std::endl;
            }
        }
        
        if (centroids.size() <= 1) break;
    }
    
    new_words_added_ += new_words;
    word_count_ += new_words;
    
    // 清理已使用的困难样本
    hard_examples_.clear();
    
    std::cerr << "[PlaceRecog] 增量学习完成，新增 " << new_words << " 个词，总计 " 
              << new_words_added_ << " 个新词" << std::endl;
    
    // 学习后更新IDF权重
    updateIDFWeightsIncremental();
}

}  // namespace loopdetection
}  // namespace sfc_core

#endif  // SFC_CORE_LOOPDETECTION_PLACE_RECOGNITION_HPP_
