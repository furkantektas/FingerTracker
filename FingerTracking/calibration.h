/* Reference: https://github.com/upperwal/opencv/blob/master/samples/cpp/stereo_calibrate_real_time.cpp */
#ifndef FingerTracking_calibration_h
#define FingerTracking_calibration_h


#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <iostream>

#define timeGap 250000000U

using namespace cv;
using namespace std;

class Calibration {
public:
    Calibration();
    int calibrate(VideoCapture camLeft, VideoCapture camRight);
    const Mat& getR() const;
    const Mat& getT() const;
    const Mat& getE() const;
    const Mat& getF() const;
    const Mat& getCameraMatrix1() const;
    const Mat& getCameraMatrix2() const;
    const Mat& getDistCoeffs1() const;
    const Mat& getDistCoeffs2() const;
    const Mat& getProjection1() const;
    const Mat& getProjection2() const;
private:
    Mat R,T,E,F;
    Mat cameraMatrix[2], distCoeffs[2], projections[2];
    enum Modes { DETECTING, CAPTURING, CALIBRATING};
    Modes mode = DETECTING;
    const int noOfStereoPairs = 14;
    int stereoPairIndex = 0, cornerImageIndex=0;
    int goIn = 1;
    Mat _leftOri, _rightOri;
    int64 prevTickCount;
    vector<Point2f> cornersLeft, cornersRight;
    vector<vector<Point2f> > cameraImagePoints[2];
    Size boardSize;
    void displayPairIndex(Mat& img);
    bool findChessBoard();
    void displayImages();
    void saveImages(Mat leftImage, Mat rightImage, int pairIndex);
    void calibrateStereoCamera();
};

#endif
