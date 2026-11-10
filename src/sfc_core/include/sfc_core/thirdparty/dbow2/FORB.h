/**
 * File: FORB.h
 * Date: June 2012
 * Author: Dorian Galvez-Lopez
 * Description: functions for ORB descriptors
 * License: see LICENSE.txt
 */

#ifndef __D_T_F_ORB__
#define __D_T_F_ORB__

#include <opencv2/core/core.hpp>
#include <vector>
#include <string>

namespace DBoW2 {

/// Functions to manipulate ORB descriptors
class FORB
{
public:
    typedef cv::Mat TDescriptor;
    typedef const TDescriptor* pDescriptor;
    static const int L = 32;  // ORB descriptor is 32 bytes = 256 bits

    static void meanValue(const std::vector<pDescriptor>& descriptors,
                          TDescriptor& mean);

    static int distance(const TDescriptor& a, const TDescriptor& b);

    static std::string toString(const TDescriptor& a);
    static void fromString(TDescriptor& a, const std::string& s);

    static void toMat32F(const std::vector<TDescriptor>& descriptors,
                          cv::Mat& mat);
    static void toMat8U(const std::vector<TDescriptor>& descriptors,
                         cv::Mat& mat);
};

} // namespace DBoW2

#endif
