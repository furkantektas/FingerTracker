//
//  common_utils.h
//  FingerTracking
//
//  Created by Furkan Tektas on 11/8/14.
//  Copyright (c) 2014 Furkan Tektas. All rights reserved.
//

#ifndef __FingerTracking__common_utils__
#define __FingerTracking__common_utils__

#include <string>
#include <opencv2/opencv.hpp>
class CommonUtils{
    public:
        static void printCameraInfo(cv::VideoCapture cam);
        static void setCameraResolution(cv::VideoCapture cam, int width, int height);
        static void putTextWrapper(cv::Mat& img, const char* text, int x=10, int y=30);
        static void displayFPS(cv::Mat&frame, double fps, int x_pos);
        static std::string getImageType(int number);
        static float pointDistance(const cv::Point& a, const cv::Point& b);
        static float getLineAngle(const cv::Point& p1, const cv::Point& p2);
};
#endif /* defined(__FingerTracking__common_utils__) */
