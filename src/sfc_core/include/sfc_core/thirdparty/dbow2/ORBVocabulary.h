/**
 * File: ORBVocabulary.h
 * Based on: DBoW2/TemplatedVocabulary.h (Dorian Galvez-Lopez)
 * ORB-SLAM3 style ORB Vocabulary
 * Uses DBoW2's templated vocabulary with ORB (FORB) descriptor
 * Modified: removed Boost serialization, removed OpenCV FileStorage binary I/O
 */

#ifndef ORBVOCABULARY_H
#define ORBVOCABULARY_H

#include <cassert>
#include <vector>
#include <numeric>
#include <fstream>
#include <string>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <limits>

#include "BowVector.h"
#include "FeatureVector.h"
#include "FORB.h"
#include "ScoringObject.h"
#include "Random.h"

namespace DBoW2 {

template<class TDescriptor, class F>
class TemplatedVocabulary
{
public:
    TemplatedVocabulary(int k = 10, int L = 6,
                       WeightingType weighting = TF_IDF,
                       ScoringType scoring = L1_NORM);
    TemplatedVocabulary(const std::string& filename);
    TemplatedVocabulary(const char* filename);
    TemplatedVocabulary(const TemplatedVocabulary<TDescriptor, F>& voc);
    virtual ~TemplatedVocabulary();

    TemplatedVocabulary<TDescriptor, F>& operator=(
        const TemplatedVocabulary<TDescriptor, F>& voc);

    virtual void create(const std::vector<std::vector<TDescriptor> >& training_features);
    virtual void create(const std::vector<std::vector<TDescriptor> >& training_features,
                        int k, int L);
    virtual void create(const std::vector<std::vector<TDescriptor> >& training_features,
                        int k, int L, WeightingType weighting, ScoringType scoring);

    virtual inline unsigned int size() const;
    virtual inline bool empty() const;

    virtual void transform(const std::vector<TDescriptor>& features, BowVector& v) const;
    virtual void transform(const std::vector<TDescriptor>& features,
                          BowVector& v, FeatureVector& fv, int levelsup) const;
    virtual WordId transform(const TDescriptor& feature) const;

    inline double score(const BowVector& a, const BowVector& b) const;
    virtual NodeId getParentNode(WordId wid, int levelsup) const;
    void getWordsFromNode(NodeId nid, std::vector<WordId>& words) const;

    inline int getBranchingFactor() const { return m_k; }
    inline int getDepthLevels() const { return m_L; }
    float getEffectiveLevels() const;

    virtual inline TDescriptor getWord(WordId wid) const;
    virtual inline WordValue getWordWeight(WordId wid) const;

    inline WeightingType getWeightingType() const { return m_weighting; }
    inline ScoringType getScoringType() const { return m_scoring; }
    inline void setWeightingType(WeightingType type);
    void setScoringType(ScoringType type);

    bool loadFromTextFile(const std::string& filename);
    void saveToTextFile(const std::string& filename) const;

    struct Node
    {
        NodeId id = 0;
        WordValue weight = 0;
        std::vector<NodeId> children;
        NodeId parent = 0;
        TDescriptor descriptor;
        WordId word_id = 0;

        Node() = default;
        explicit Node(NodeId _id) : id(_id) {}
        inline bool isLeaf() const { return children.empty(); }
    };

protected:
    int m_k;
    int m_L;
    int m_NWords;
    WeightingType m_weighting;
    ScoringType m_scoring;
    double m_logN;

    std::vector<Node> m_nodes;
    std::vector<Node*> m_words;

    GeneralScoring* m_scoring_object;

    void createScoringObject();

    using pDescriptor = const TDescriptor*;

    void getFeatures(const std::vector<std::vector<TDescriptor> >& training_features,
                    std::vector<pDescriptor>& features) const;
    virtual void transform(const TDescriptor& feature,
                          WordId& id, WordValue& weight,
                          NodeId* nid = nullptr, int levelsup = 0) const;
    virtual void transform(const TDescriptor& feature, WordId& id) const;
    void HKmeansStep(NodeId parent_id, const std::vector<pDescriptor>& descriptors,
                     int current_level);
    void initiateClusters(const std::vector<pDescriptor>& descriptors,
                          std::vector<TDescriptor>& clusters) const;
    void initiateClustersKMpp(const std::vector<pDescriptor>& pfeatures,
                              std::vector<TDescriptor>& clusters) const;
    void createWords();
    void setNodeWeights(const std::vector<std::vector<TDescriptor> >& training_features);
};

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedVocabulary<TDescriptor, F>::TemplatedVocabulary(
    int k, int L, WeightingType weighting, ScoringType scoring)
    : m_k(k), m_L(L), m_NWords(0),
      m_weighting(weighting), m_scoring(scoring),
      m_logN(0.0), m_scoring_object(nullptr)
{
    createScoringObject();
    m_nodes.reserve(1000);
    m_words.reserve(1000);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedVocabulary<TDescriptor, F>::TemplatedVocabulary(const std::string& filename)
    : m_k(0), m_L(0), m_NWords(0),
      m_weighting(TF_IDF), m_scoring(L1_NORM),
      m_logN(0.0), m_scoring_object(nullptr)
{
    createScoringObject();
    loadFromTextFile(filename);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedVocabulary<TDescriptor, F>::TemplatedVocabulary(const char* filename)
    : m_k(0), m_L(0), m_NWords(0),
      m_weighting(TF_IDF), m_scoring(L1_NORM),
      m_logN(0.0), m_scoring_object(nullptr)
{
    createScoringObject();
    loadFromTextFile(std::string(filename));
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedVocabulary<TDescriptor, F>::TemplatedVocabulary(
    const TemplatedVocabulary<TDescriptor, F>& voc)
    : m_k(voc.m_k), m_L(voc.m_L), m_NWords(voc.m_NWords),
      m_weighting(voc.m_weighting), m_scoring(voc.m_scoring),
      m_logN(voc.m_logN), m_nodes(voc.m_nodes), m_words(voc.m_words),
      m_scoring_object(nullptr)
{
    createScoringObject();
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedVocabulary<TDescriptor, F>::~TemplatedVocabulary()
{
    if(m_scoring_object) delete m_scoring_object;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedVocabulary<TDescriptor, F>& TemplatedVocabulary<TDescriptor, F>::operator=(
    const TemplatedVocabulary<TDescriptor, F>& voc)
{
    if(this != &voc)
    {
        m_k = voc.m_k;
        m_L = voc.m_L;
        m_NWords = voc.m_NWords;
        m_weighting = voc.m_weighting;
        m_scoring = voc.m_scoring;
        m_logN = voc.m_logN;
        m_nodes = voc.m_nodes;
        m_words = voc.m_words;
        setScoringType(voc.m_scoring);
    }
    return *this;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor, F>::createScoringObject()
{
    if(m_scoring_object) delete m_scoring_object;

    switch(m_scoring)
    {
        case L1_NORM:    m_scoring_object = new L1Scoring; break;
        case L2_NORM:    m_scoring_object = new L2Scoring; break;
        case CHI_SQUARE: m_scoring_object = new ChiSquareScoring; break;
        case KL:         m_scoring_object = new KLScoring; break;
        case BHATTACHARYYA: m_scoring_object = new BhattacharyyaScoring; break;
        case DOT_PRODUCT: m_scoring_object = new DotProductScoring; break;
    }
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor, F>::create(
    const std::vector<std::vector<TDescriptor> >& training_features)
{
    create(training_features, m_k, m_L);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor, F>::create(
    const std::vector<std::vector<TDescriptor> >& training_features,
    int k, int L)
{
    create(training_features, k, L, m_weighting, m_scoring);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor, F>::create(
    const std::vector<std::vector<TDescriptor> >& training_features,
    int k, int L, WeightingType weighting, ScoringType scoring)
{
    m_k = k;
    m_L = L;
    m_weighting = weighting;
    m_scoring = scoring;
    createScoringObject();

    m_nodes.clear();
    m_words.clear();
    m_nodes.reserve(10000);
    m_words.reserve(10000);

    m_nodes.resize(1);
    m_nodes[0].id = 0;

    std::vector<pDescriptor> features;
    getFeatures(training_features, features);

    HKmeansStep(0, features, 0);

    createWords();
    setNodeWeights(training_features);

    m_logN = std::log((double)training_features.size());
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor, F>::getFeatures(
    const std::vector<std::vector<TDescriptor> >& training_features,
    std::vector<pDescriptor>& features) const
{
    features.clear();
    features.reserve(training_features.size());

    for(size_t i = 0; i < training_features.size(); ++i)
    {
        const std::vector<TDescriptor>& features_i = training_features[i];
        for(size_t j = 0; j < features_i.size(); ++j)
        {
            features.push_back(&features_i[j]);
        }
    }
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor, F>::HKmeansStep(
    NodeId parent_id, const std::vector<pDescriptor>& descriptors,
    int current_level)
{
    if(descriptors.empty()) return;

    std::vector<TDescriptor> clusters;
    initiateClusters(descriptors, clusters);

    std::vector<std::vector<unsigned int> > groups;
    groups.resize(m_k);

    int max_iter = 10;
    for(int iter = 0; iter < max_iter; ++iter)
    {
        for(auto& g : groups) g.clear();

        for(size_t i = 0; i < descriptors.size(); ++i)
        {
            const TDescriptor& d = *descriptors[i];
            int best_idx = 0;
            int best_dist = std::numeric_limits<int>::max();

            for(int j = 0; j < (int)clusters.size(); ++j)
            {
                int dist = F::distance(d, clusters[j]);
                if(dist < best_dist)
                {
                    best_dist = dist;
                    best_idx = j;
                }
            }

            groups[best_idx].push_back((unsigned int)i);
        }

        bool changed = false;
        for(int i = 0; i < (int)clusters.size(); ++i)
        {
            if(groups[i].empty()) continue;

            TDescriptor mean_desc;
            std::vector<pDescriptor> group_desc;
            group_desc.reserve(groups[i].size());
            for(unsigned int idx : groups[i])
            {
                group_desc.push_back(descriptors[idx]);
            }

            F::meanValue(group_desc, mean_desc);

            if(F::distance(clusters[i], mean_desc) > 1e-6)
            {
                clusters[i] = mean_desc;
                changed = true;
            }
        }

        if(!changed) break;
    }

    NodeId node_id = m_nodes.size();
    m_nodes[parent_id].children.reserve(m_k);

    for(const TDescriptor& c : clusters)
    {
        Node new_node((NodeId)node_id);
        new_node.descriptor = c.clone();
        new_node.parent = parent_id;
        m_nodes.push_back(new_node);
        m_nodes[parent_id].children.push_back(node_id);
        ++node_id;
    }

    if(current_level < m_L)
    {
        for(size_t i = 0; i < groups.size(); ++i)
        {
            if(groups[i].size() > 1)
            {
                std::vector<pDescriptor> child_features;
                child_features.reserve(groups[i].size());
                for(unsigned int idx : groups[i])
                {
                    child_features.push_back(descriptors[idx]);
                }
                HKmeansStep(m_nodes[parent_id].children[i], child_features,
                             current_level + 1);
            }
        }
    }
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor, F>::initiateClusters(
    const std::vector<pDescriptor>& descriptors,
    std::vector<TDescriptor>& clusters) const
{
    initiateClustersKMpp(descriptors, clusters);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor, F>::initiateClustersKMpp(
    const std::vector<pDescriptor>& pfeatures,
    std::vector<TDescriptor>& clusters) const
{
    DUtils::Random::SeedRandOnce();

    clusters.clear();
    clusters.reserve(m_k);
    std::vector<double> min_dists(pfeatures.size(), std::numeric_limits<double>::max());

    int ifeature = DUtils::Random::RandomInt(0, (int)pfeatures.size() - 1);
    clusters.push_back(pfeatures[ifeature]->clone());

    for(size_t i = 0; i < pfeatures.size(); ++i)
    {
        double dist = F::distance(*pfeatures[i], clusters.back());
        if(dist < min_dists[i]) min_dists[i] = dist;
    }

    while((int)clusters.size() < m_k)
    {
        double dist_sum = 0.0;
        for(double d : min_dists) dist_sum += d;

        if(dist_sum <= 0.0) break;

        double cut_d = 0.0;
        do {
            cut_d = DUtils::Random::RandomValue(0.0, dist_sum);
        } while(cut_d == 0.0);

        double d_up_now = 0.0;
        size_t selected_idx = pfeatures.size() - 1;
        for(size_t i = 0; i < min_dists.size(); ++i)
        {
            d_up_now += min_dists[i];
            if(d_up_now >= cut_d)
            {
                selected_idx = i;
                break;
            }
        }

        clusters.push_back(pfeatures[selected_idx]->clone());

        for(size_t i = 0; i < pfeatures.size(); ++i)
        {
            double dist = F::distance(*pfeatures[i], clusters.back());
            if(dist < min_dists[i]) min_dists[i] = dist;
        }
    }
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor, F>::createWords()
{
    m_words.resize(100000);
    int word_count = 0;

    for(NodeId nid = 1; nid < (NodeId)m_nodes.size(); ++nid)
    {
        if(m_nodes[nid].isLeaf())
        {
            Node& node = m_nodes[nid];
            WordId wid = (WordId)word_count;
            node.word_id = wid;
            m_words[wid] = &m_nodes[nid];
            ++word_count;
        }
    }

    m_words.resize(word_count);
    m_NWords = word_count;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor, F>::setNodeWeights(
    const std::vector<std::vector<TDescriptor> >& training_features)
{
    const unsigned int NWords = m_words.size();
    if(NWords == 0) return;

    std::vector<unsigned int> Ni(NWords, 0);
    std::vector<bool> counted(NWords, false);

    for(const auto& features : training_features)
    {
        std::fill(counted.begin(), counted.end(), false);

        for(const auto& feat : features)
        {
            WordId word_id;
            transform(feat, word_id);
            if(!counted[word_id])
            {
                Ni[word_id]++;
                counted[word_id] = true;
            }
        }
    }

    for(unsigned int i = 0; i < NWords; ++i)
    {
        if(Ni[i] > 0)
        {
            m_words[i]->weight = std::log((double)training_features.size() / (double)Ni[i]);
        }
    }
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
inline unsigned int TemplatedVocabulary<TDescriptor, F>::size() const
{
    return m_words.size();
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
inline bool TemplatedVocabulary<TDescriptor, F>::empty() const
{
    return m_words.empty();
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
float TemplatedVocabulary<TDescriptor, F>::getEffectiveLevels() const
{
    long sum = 0;
    for(const Node* word : m_words)
    {
        const Node* p = word;
        for(; p->id != 0; ++sum) p = &m_nodes[p->parent];
    }
    return m_words.empty() ? 0.0f : (float)((double)sum / (double)m_words.size());
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
inline TDescriptor TemplatedVocabulary<TDescriptor, F>::getWord(WordId wid) const
{
    return m_words[wid]->descriptor;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
inline WordValue TemplatedVocabulary<TDescriptor, F>::getWordWeight(WordId wid) const
{
    return m_words[wid]->weight;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
inline double TemplatedVocabulary<TDescriptor, F>::score(
    const BowVector& a, const BowVector& b) const
{
    return m_scoring_object->score(a, b);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
inline void TemplatedVocabulary<TDescriptor, F>::setWeightingType(WeightingType type)
{
    m_weighting = type;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor, F>::setScoringType(ScoringType type)
{
    m_scoring = type;
    createScoringObject();
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
WordId TemplatedVocabulary<TDescriptor, F>::transform(const TDescriptor& feature) const
{
    if(empty()) return 0;
    WordId wid;
    transform(feature, wid);
    return wid;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
inline void TemplatedVocabulary<TDescriptor, F>::transform(
    const TDescriptor& feature, WordId& id) const
{
    if(empty()) { id = 0; return; }
    WordValue w;
    transform(feature, id, w);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor, F>::transform(
    const TDescriptor& feature,
    WordId& id, WordValue& weight, NodeId* nid, int levelsup) const
{
    if(empty())
    {
        id = 0;
        weight = 0;
        return;
    }

    NodeId node_id = 0;
    NodeId parent_id = 0;
    int current_level = 0;

    do
    {
        const Node& node = m_nodes[node_id];
        const std::vector<NodeId>& children = node.children;

        if(children.empty())
        {
            break;
        }

        int best_child_idx = 0;
        int best_dist = std::numeric_limits<int>::max();

        for(size_t i = 0; i < children.size(); ++i)
        {
            NodeId child_id = children[i];
            int dist = F::distance(feature, m_nodes[child_id].descriptor);
            if(dist < best_dist)
            {
                best_dist = dist;
                best_child_idx = (int)i;
            }
        }

        parent_id = node_id;
        node_id = children[best_child_idx];
        ++current_level;

    } while(current_level < m_L);

    if(nid != nullptr)
    {
        if(levelsup == 0)
        {
            *nid = node_id;
        }
        else
        {
            int levels_up = current_level - levelsup;
            if(levels_up < 0) levels_up = 0;
            NodeId target_id = node_id;
            for(int i = current_level; i > levels_up; --i)
            {
                target_id = m_nodes[target_id].parent;
            }
            *nid = target_id;
        }
    }

    const Node& final_node = m_nodes[node_id];
    id = final_node.word_id;
    weight = final_node.weight;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor, F>::transform(
    const std::vector<TDescriptor>& features, BowVector& v) const
{
    v.clear();
    if(empty()) return;

    LNorm norm;
    bool must_normalize = m_scoring_object->mustNormalize(norm);

    if(m_weighting == TF || m_weighting == TF_IDF)
    {
        for(const TDescriptor& feat : features)
        {
            WordId id;
            WordValue w;
            transform(feat, id, w);
            if(w > 0) v.addWeight(id, w);
        }

        if(!v.empty() && !must_normalize)
        {
            double nd = (double)v.size();
            for(auto& kv : v) kv.second /= nd;
        }
    }
    else
    {
        for(const TDescriptor& feat : features)
        {
            WordId id;
            WordValue w;
            transform(feat, id, w);
            if(w > 0) v.addIfNotExist(id, w);
        }
    }

    if(must_normalize)
    {
        double n = v.norm(norm);
        if(n > 0.0)
        {
            for(auto& kv : v) kv.second /= n;
        }
    }
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor, F>::transform(
    const std::vector<TDescriptor>& features,
    BowVector& v, FeatureVector& fv, int levelsup) const
{
    v.clear();
    fv.clear();
    if(empty()) return;

    LNorm norm;
    bool must_normalize = m_scoring_object->mustNormalize(norm);

    if(m_weighting == TF || m_weighting == TF_IDF)
    {
        unsigned int i_feature = 0;
        for(const TDescriptor& feat : features)
        {
            WordId id;
            NodeId nid;
            WordValue w;
            transform(feat, id, w, &nid, levelsup);

            if(w > 0)
            {
                v.addWeight(id, w);
                fv.addFeature(nid, i_feature);
            }
            ++i_feature;
        }

        if(!v.empty() && !must_normalize)
        {
            double nd = (double)v.size();
            for(auto& kv : v) kv.second /= nd;
        }
    }
    else
    {
        unsigned int i_feature = 0;
        for(const TDescriptor& feat : features)
        {
            WordId id;
            NodeId nid;
            WordValue w;
            transform(feat, id, w, &nid, levelsup);

            if(w > 0)
            {
                v.addIfNotExist(id, w);
                fv.addFeature(nid, i_feature);
            }
            ++i_feature;
        }
    }

    if(must_normalize)
    {
        double n = v.norm(norm);
        if(n > 0.0)
        {
            for(auto& kv : v) kv.second /= n;
        }
    }
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
NodeId TemplatedVocabulary<TDescriptor, F>::getParentNode(WordId wid, int levelsup) const
{
    NodeId nid = m_words[wid]->id;
    for(int i = 0; i < levelsup; ++i)
    {
        if(nid == 0) break;
        nid = m_nodes[nid].parent;
    }
    return nid;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor, F>::getWordsFromNode(NodeId nid,
                                                             std::vector<WordId>& words) const
{
    words.clear();

    std::vector<NodeId> nodes_to_visit;
    nodes_to_visit.push_back(nid);

    while(!nodes_to_visit.empty())
    {
        NodeId current_id = nodes_to_visit.back();
        nodes_to_visit.pop_back();

        const Node& node = m_nodes[current_id];

        if(node.isLeaf())
        {
            words.push_back(node.word_id);
        }
        else
        {
            for(NodeId child_id : node.children)
            {
                nodes_to_visit.push_back(child_id);
            }
        }
    }
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
bool TemplatedVocabulary<TDescriptor, F>::loadFromTextFile(const std::string& filename)
{
    std::ifstream f(filename.c_str());
    if(f.eof()) return false;

    m_words.clear();
    m_nodes.clear();

    std::string s;
    std::getline(f, s);
    std::stringstream ss(s);

    ss >> m_k;
    ss >> m_L;
    int scoring_int, weighting_int;
    ss >> scoring_int;
    ss >> weighting_int;

    if(m_k < 0 || m_k > 20 || m_L < 1 || m_L > 10 ||
       scoring_int < 0 || scoring_int > 5 ||
       weighting_int < 0 || weighting_int > 3)
    {
        return false;
    }

    m_scoring = (ScoringType)scoring_int;
    m_weighting = (WeightingType)weighting_int;
    createScoringObject();

    int expected_nodes = (int)((std::pow((double)m_k, (double)m_L + 1) - 1) / (m_k - 1));
    m_nodes.reserve(expected_nodes + 10);
    m_words.reserve(expected_nodes * 2);

    m_nodes.resize(1);
    m_nodes[0].id = 0;

    int nid = 1;
    while(!f.eof())
    {
        std::string snode;
        std::getline(f, snode);
        if(snode.empty()) break;

        std::stringstream ssnode(snode);

        m_nodes.resize(nid + 1);
        Node& node = m_nodes[nid];
        node.id = nid;

        int pid;
        ssnode >> pid;
        node.parent = (NodeId)pid;
        m_nodes[pid].children.push_back((NodeId)nid);

        int nIsLeaf;
        ssnode >> nIsLeaf;

        std::stringstream ssd;
        for(int iD = 0; iD < F::L; ++iD)
        {
            std::string sElement;
            ssnode >> sElement;
            ssd << sElement << " ";
        }
        F::fromString(node.descriptor, ssd.str());

        ssnode >> node.weight;

        if(nIsLeaf > 0)
        {
            WordId wid = (WordId)m_words.size();
            node.word_id = wid;
            m_words.push_back(&m_nodes[nid]);
        }

        ++nid;
    }

    m_NWords = (int)m_words.size();
    m_logN = std::log((double)m_NWords);
    return true;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor, F>::saveToTextFile(const std::string& filename) const
{
    std::ofstream f(filename.c_str());
    f << m_k << " " << m_L << " " << (int)m_scoring << " " << (int)m_weighting << "\n";

    for(const Node& node : m_nodes)
    {
        f << node.parent << " " << (node.isLeaf() ? 1 : 0) << " ";

        for(int i = 0; i < F::L; ++i)
        {
            unsigned char byte = static_cast<unsigned char*>(node.descriptor.ptr())[i];
            f << (int)byte << " ";
        }

        f << node.weight << "\n";
    }

    f.close();
}

// --------------------------------------------------------------------------

typedef TemplatedVocabulary<FORB::TDescriptor, FORB> ORBVocabulary;

// --------------------------------------------------------------------------

} // namespace DBoW2

#endif // ORBVOCABULARY_H
