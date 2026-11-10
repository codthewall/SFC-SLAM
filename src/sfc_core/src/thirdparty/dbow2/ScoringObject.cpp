/**
 * File: ScoringObject.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: functions to compute bow scores
 * License: see LICENSE.txt
 */

#include "sfc_core/thirdparty/dbow2/ScoringObject.h"
#include <cfloat>
#include <cmath>

namespace DBoW2 {

const double GeneralScoring::LOG_EPS = std::log(DBL_EPSILON);

double L1Scoring::score(const BowVector &v1, const BowVector &v2) const
{
    BowVector::const_iterator v1_it = v1.begin();
    BowVector::const_iterator v2_it = v2.begin();
    const BowVector::const_iterator v1_end = v1.end();
    const BowVector::const_iterator v2_end = v2.end();

    double score = 0;

    while(v1_it != v1_end && v2_it != v2_end)
    {
        const WordValue& vi = v1_it->second;
        const WordValue& wi = v2_it->second;

        if(v1_it->first == v2_it->first)
        {
            score += std::abs(vi - wi) - std::abs(vi) - std::abs(wi);
            ++v1_it;
            ++v2_it;
        }
        else if(v1_it->first < v2_it->first)
        {
            v1_it = v1.lower_bound(v2_it->first);
        }
        else
        {
            v2_it = v2.lower_bound(v1_it->first);
        }
    }

    score = -score / 2.0;
    return score;
}

double L2Scoring::score(const BowVector &v1, const BowVector &v2) const
{
    BowVector::const_iterator v1_it = v1.begin();
    BowVector::const_iterator v2_it = v2.begin();
    const BowVector::const_iterator v1_end = v1.end();
    const BowVector::const_iterator v2_end = v2.end();

    double score = 0;

    while(v1_it != v1_end && v2_it != v2_end)
    {
        const WordValue& vi = v1_it->second;
        const WordValue& wi = v2_it->second;

        if(v1_it->first == v2_it->first)
        {
            score += vi * wi;
            ++v1_it;
            ++v2_it;
        }
        else if(v1_it->first < v2_it->first)
        {
            v1_it = v1.lower_bound(v2_it->first);
        }
        else
        {
            v2_it = v2.lower_bound(v1_it->first);
        }
    }

    if(score >= 1.0)
        score = 1.0;
    else
        score = 1.0 - std::sqrt(1.0 - score);
    return score;
}

double ChiSquareScoring::score(const BowVector &v1, const BowVector &v2) const
{
    BowVector::const_iterator v1_it = v1.begin();
    BowVector::const_iterator v2_it = v2.begin();
    const BowVector::const_iterator v1_end = v1.end();
    const BowVector::const_iterator v2_end = v2.end();

    double score = 0;

    while(v1_it != v1_end && v2_it != v2_end)
    {
        const WordValue& vi = v1_it->second;
        const WordValue& wi = v2_it->second;

        if(v1_it->first == v2_it->first)
        {
            if(vi + wi != 0.0) score += vi * wi / (vi + wi);
            ++v1_it;
            ++v2_it;
        }
        else if(v1_it->first < v2_it->first)
        {
            v1_it = v1.lower_bound(v2_it->first);
        }
        else
        {
            v2_it = v2.lower_bound(v1_it->first);
        }
    }

    return 2.0 * score;
}

double KLScoring::score(const BowVector &v1, const BowVector &v2) const
{
    BowVector::const_iterator v1_it = v1.begin();
    BowVector::const_iterator v2_it = v2.begin();
    const BowVector::const_iterator v1_end = v1.end();
    const BowVector::const_iterator v2_end = v2.end();

    double score = 0;

    while(v1_it != v1_end && v2_it != v2_end)
    {
        const WordValue& vi = v1_it->second;
        const WordValue& wi = v2_it->second;

        if(v1_it->first == v2_it->first)
        {
            if(vi != 0 && wi != 0) score += vi * std::log(vi / wi);
            ++v1_it;
            ++v2_it;
        }
        else if(v1_it->first < v2_it->first)
        {
            score += vi * (std::log(vi) - LOG_EPS);
            ++v1_it;
        }
        else
        {
            v2_it = v2.lower_bound(v1_it->first);
        }
    }

    for(; v1_it != v1_end; ++v1_it)
        if(v1_it->second != 0)
            score += v1_it->second * (std::log(v1_it->second) - LOG_EPS);

    return score;
}

double BhattacharyyaScoring::score(const BowVector &v1,
                                     const BowVector &v2) const
{
    BowVector::const_iterator v1_it = v1.begin();
    BowVector::const_iterator v2_it = v2.begin();
    const BowVector::const_iterator v1_end = v1.end();
    const BowVector::const_iterator v2_end = v2.end();

    double score = 0;

    while(v1_it != v1_end && v2_it != v2_end)
    {
        const WordValue& vi = v1_it->second;
        const WordValue& wi = v2_it->second;

        if(v1_it->first == v2_it->first)
        {
            score += std::sqrt(vi * wi);
            ++v1_it;
            ++v2_it;
        }
        else if(v1_it->first < v2_it->first)
        {
            v1_it = v1.lower_bound(v2_it->first);
        }
        else
        {
            v2_it = v2.lower_bound(v1_it->first);
        }
    }

    return score;
}

double DotProductScoring::score(const BowVector &v1,
                                  const BowVector &v2) const
{
    BowVector::const_iterator v1_it = v1.begin();
    BowVector::const_iterator v2_it = v2.begin();
    const BowVector::const_iterator v1_end = v1.end();
    const BowVector::const_iterator v2_end = v2.end();

    double score = 0;

    while(v1_it != v1_end && v2_it != v2_end)
    {
        const WordValue& vi = v1_it->second;
        const WordValue& wi = v2_it->second;

        if(v1_it->first == v2_it->first)
        {
            score += vi * wi;
            ++v1_it;
            ++v2_it;
        }
        else if(v1_it->first < v2_it->first)
        {
            v1_it = v1.lower_bound(v2_it->first);
        }
        else
        {
            v2_it = v2.lower_bound(v1_it->first);
        }
    }

    return score;
}

} // namespace DBoW2
