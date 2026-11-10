/**
 * File: ScoringObject.h
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: functions to compute bow scores
 * License: see LICENSE.txt
 * Modified: removed binary serialization, kept scoring functions
 */

#ifndef __D_T_SCORING_OBJECT__
#define __D_T_SCORING_OBJECT__

#include "BowVector.h"

namespace DBoW2 {

/// Base class of scoring functions
class GeneralScoring
{
public:
    virtual double score(const BowVector &v, const BowVector &w) const = 0;
    virtual bool mustNormalize(LNorm &norm) const = 0;
    static const double LOG_EPS;
    virtual ~GeneralScoring() {}
};

// --------------------------------------------------------------------------

class L1Scoring: public GeneralScoring
{
public:
    virtual double score(const BowVector &v, const BowVector &w) const override;
    virtual inline bool mustNormalize(LNorm &norm) const override
      { norm = L1; return true; }
};

class L2Scoring: public GeneralScoring
{
public:
    virtual double score(const BowVector &v, const BowVector &w) const override;
    virtual inline bool mustNormalize(LNorm &norm) const override
      { norm = L2; return true; }
};

class ChiSquareScoring: public GeneralScoring
{
public:
    virtual double score(const BowVector &v, const BowVector &w) const override;
    virtual inline bool mustNormalize(LNorm &norm) const override
      { norm = L1; return true; }
};

class KLScoring: public GeneralScoring
{
public:
    virtual double score(const BowVector &v, const BowVector &w) const override;
    virtual inline bool mustNormalize(LNorm &norm) const override
      { norm = L1; return true; }
};

class BhattacharyyaScoring: public GeneralScoring
{
public:
    virtual double score(const BowVector &v, const BowVector &w) const override;
    virtual inline bool mustNormalize(LNorm &norm) const override
      { norm = L1; return true; }
};

class DotProductScoring: public GeneralScoring
{
public:
    virtual double score(const BowVector &v, const BowVector &w) const override;
    virtual inline bool mustNormalize(LNorm &norm) const override
      { norm = L1; return false; }
};

} // namespace DBoW2

#endif
