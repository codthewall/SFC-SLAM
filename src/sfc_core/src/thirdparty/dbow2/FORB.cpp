/**
 * File: FORB.cpp
 * Date: June 2012
 * Author: Dorian Galvez-Lopez
 * Description: functions for ORB descriptors
 * License: see LICENSE.txt
 */

#include "sfc_core/thirdparty/dbow2/FORB.h"
#include <opencv2/core.hpp>

namespace DBoW2 {

// --------------------------------------------------------------------------

void FORB::meanValue(const std::vector<pDescriptor>& descriptors,
                     TDescriptor& mean)
{
    if(descriptors.empty())
    {
        mean.release();
        return;
    }

    if(descriptors.size() == 1)
    {
        mean = descriptors[0]->clone();
        return;
    }

    mean = cv::Mat::zeros(1, L, CV_8U);
    double inv_n = 1.0 / descriptors.size();

    for(size_t i = 0; i < descriptors.size(); ++i)
    {
        const TDescriptor& d = *descriptors[i];
        for(int j = 0; j < L; ++j)
        {
            mean.at<unsigned char>(j) += d.at<unsigned char>(j) * inv_n;
        }
    }
}

// --------------------------------------------------------------------------

int FORB::distance(const TDescriptor& a, const TDescriptor& b)
{
    if (a.empty() || b.empty()) return 256;
    if (a.cols != b.cols || a.rows != b.rows) return 256;
    return cv::norm(a, b, cv::NORM_HAMMING);
}

// --------------------------------------------------------------------------

std::string FORB::toString(const TDescriptor& a)
{
    std::stringstream ss;
    for(int i = 0; i < L; ++i)
    {
        ss << static_cast<int>(a.at<unsigned char>(i)) << " ";
    }
    return ss.str();
}

// --------------------------------------------------------------------------

void FORB::fromString(TDescriptor& a, const std::string& s)
{
    a.create(1, L, CV_8U);
    std::stringstream ss(s);
    for(int i = 0; i < L; ++i)
    {
        int val;
        ss >> val;
        a.at<unsigned char>(i) = static_cast<unsigned char>(val);
    }
}

// --------------------------------------------------------------------------

void FORB::toMat32F(const std::vector<TDescriptor>& descriptors,
                     cv::Mat& mat)
{
    mat.create(descriptors.size(), L, CV_32F);
    for(size_t i = 0; i < descriptors.size(); ++i)
    {
        const TDescriptor& d = descriptors[i];
        float* row = mat.ptr<float>(i);
        for(int j = 0; j < L; ++j)
        {
            row[j] = d.at<unsigned char>(j);
        }
    }
}

// --------------------------------------------------------------------------

void FORB::toMat8U(const std::vector<TDescriptor>& descriptors,
                    cv::Mat& mat)
{
    mat.create(descriptors.size(), L, CV_8U);
    for(size_t i = 0; i < descriptors.size(); ++i)
    {
        descriptors[i].copyTo(mat.row(i));
    }
}

// --------------------------------------------------------------------------

} // namespace DBoW2
