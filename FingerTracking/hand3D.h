#ifndef __hand3D_h__
#define __hand3D_h__

#include <opencv2/opencv.hpp>
#include "handDetection.h"
#include "calibration.h"

using namespace cv;

class Hand3D {
public:
    Hand3D(Calibration& calib, Mat& frameLeft, Mat& frameRight, Size frameSize) : calibration(calib),
                                                                                  leftHand(frameLeft,frameSize),
                                                                                  rightHand(frameRight,frameSize),
                                                                                  frameSize(frameSize){
        namedWindow(windowNameStereo);
        namedWindow(windowName3D);
    }
    void setFrames(Mat& frameLeft, Mat& frameRight);
    void find();
    std::vector<Point2f> getLeftFingers() const;
    std::vector<Point2f> getRightFingers() const;
    ~Hand3D();
private:
    Calibration& calibration;
    Size frameSize;
    HandDetection leftHand, rightHand;
    Mat scene;
    const char* windowNameStereo = "Stereo";
    const char* windowName3D = "Reprojection of 3D";
};

#endif