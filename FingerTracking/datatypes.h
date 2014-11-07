//
//  datatypes.h
//  FingerTracking
//
//  Created by Furkan Tektas on 11/8/14.
//  Copyright (c) 2014 Furkan Tektas. All rights reserved.
//

#ifndef FingerTracking_datatypes_h
#define FingerTracking_datatypes_h

#include <opencv2/opencv.hpp>

typedef struct Line_s{
    cv::Point start;
    cv::Point end;
    double angle;
} Line;

enum Hand_e{
    LEFT,
    RIGHT
};


const cv::Scalar red(255,0,0);
const cv::Scalar green(0,255,0);
const cv::Scalar blue(0,0,255);
const cv::Scalar yellow(255,255,0);

#endif
