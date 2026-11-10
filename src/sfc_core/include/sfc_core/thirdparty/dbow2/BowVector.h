/**
 * File: BowVector.h
 * Date: March 2011
 * Author: Dorian Galvez-Lopez
 * Description: bag of words vector
 * License: see LICENSE.txt
 * Modified: removed boost serialization dependency
 */

#ifndef __D_T_BOW_VECTOR__
#define __D_T_BOW_VECTOR__

#include <iostream>
#include <map>
#include <vector>

namespace DBoW2 {

/// Id of words
typedef unsigned int WordId;

/// Value of a word
typedef double WordValue;

/// Id of nodes in the vocabulary tree
typedef unsigned int NodeId;

/// L-norms for normalization
enum LNorm { L1, L2 };

/// Weighting type
enum WeightingType { TF_IDF, TF, IDF, BINARY };

/// Scoring type
enum ScoringType { L1_NORM, L2_NORM, CHI_SQUARE, KL, BHATTACHARYYA, DOT_PRODUCT };

/// Vector of words to represent images
class BowVector : public std::map<WordId, WordValue>
{
public:
    BowVector(void);
    ~BowVector(void);

    void addWeight(WordId id, WordValue v);
    void addIfNotExist(WordId id, WordValue v);

    /// L1-norm
    double norm(LNorm norm_type);
};

} // namespace DBoW2

#endif
