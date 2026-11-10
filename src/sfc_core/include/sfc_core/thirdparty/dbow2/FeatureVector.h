/**
 * File: FeatureVector.h
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: feature vector
 * License: see LICENSE.txt
 * Modified: removed boost serialization dependency
 */

#ifndef __D_T_FEATURE_VECTOR__
#define __D_T_FEATURE_VECTOR__

#include "BowVector.h"
#include <map>
#include <vector>
#include <iostream>

namespace DBoW2 {

/// Vector of nodes with indexes of local features
/// Key = NodeId, Value = vector of feature indices
class FeatureVector : public std::map<NodeId, std::vector<unsigned int> >
{
public:
    FeatureVector(void);
    ~FeatureVector(void);

    void addFeature(NodeId id, unsigned int i_feature);
};

} // namespace DBoW2

#endif
