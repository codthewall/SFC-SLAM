/**
 * File: FeatureVector.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: feature vector
 * License: see LICENSE.txt
 */

#include "sfc_core/thirdparty/dbow2/FeatureVector.h"

namespace DBoW2 {

// --------------------------------------------------------------------------

FeatureVector::FeatureVector(void) {}

FeatureVector::~FeatureVector(void) {}

// --------------------------------------------------------------------------

void FeatureVector::addFeature(NodeId id, unsigned int i_feature)
{
    FeatureVector::iterator it = this->lower_bound(id);

    if(it != this->end() && !(this->key_comp()(id, it->first)))
    {
        it->second.push_back(i_feature);
    }
    else
    {
        std::vector<unsigned int> v;
        v.push_back(i_feature);
        this->insert(it, FeatureVector::value_type(id, v));
    }
}

// --------------------------------------------------------------------------

} // namespace DBoW2
