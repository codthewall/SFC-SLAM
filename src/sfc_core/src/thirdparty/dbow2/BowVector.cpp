/**
 * File: BowVector.cpp
 * Date: March 2011
 * Author: Dorian Galvez-Lopez
 * Description: bag of words vector
 * License: see LICENSE.txt
 */

#include "sfc_core/thirdparty/dbow2/BowVector.h"
#include <cmath>

namespace DBoW2 {

// --------------------------------------------------------------------------

BowVector::BowVector(void) {}

BowVector::~BowVector(void) {}

// --------------------------------------------------------------------------

void BowVector::addWeight(WordId id, WordValue v)
{
    BowVector::iterator vit = this->lower_bound(id);
    if(vit != this->end() && !(this->key_comp()(id, vit->first)))
    {
        vit->second += v;
    }
    else
    {
        this->insert(vit, BowVector::value_type(id, v));
    }
}

// --------------------------------------------------------------------------

void BowVector::addIfNotExist(WordId id, WordValue v)
{
    BowVector::iterator vit = this->lower_bound(id);
    if(vit == this->end() || (this->key_comp()(id, vit->first)))
    {
        this->insert(vit, BowVector::value_type(id, v));
    }
}

// --------------------------------------------------------------------------

double BowVector::norm(LNorm norm_type)
{
    double sum = 0.0;

    if(norm_type == L1)
    {
        for(BowVector::const_iterator it = this->begin(); it != this->end(); ++it)
        {
            sum += std::abs(it->second);
        }
    }
    else if(norm_type == L2)
    {
        for(BowVector::const_iterator it = this->begin(); it != this->end(); ++it)
        {
            sum += it->second * it->second;
        }
        sum = std::sqrt(sum);
    }

    return sum;
}

// --------------------------------------------------------------------------

} // namespace DBoW2
