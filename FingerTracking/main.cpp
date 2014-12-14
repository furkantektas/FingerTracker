#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include "common_utils.h"
#include "datatypes.h"
#include "hand.h"
#include "calibration.h"
using namespace cv;

Calibration calib;
std::vector<cv::Vec3f> lines2;
std::vector<cv::Point2f> points;


void CallBackFunc(int event, int x, int y, int flags, void* userdata) {
    if ( event == EVENT_LBUTTONDOWN ) {
        Point2f point(x,y);
        points.push_back(point);
        cv::computeCorrespondEpilines(cv::Mat(points),1,calib.getF(),lines2);
    }
}


int main(int argc, char** argv) {
    VideoCapture cap(0),cap2(1);

    if(!cap.isOpened() || !cap2.isOpened()) // check if we succeeded
        return -1;
    
    CommonUtils::setCameraResolution(cap, 320, 240);
    cap.set(CV_CAP_PROP_EXPOSURE, 0.0);
    cap.set(CV_CAP_PROP_AUTO_EXPOSURE, 0 );
    cap.set(CV_CAP_PROP_GAIN, 0.0);

    CommonUtils::setCameraResolution(cap2, 320, 240);
    cap2.set(CV_CAP_PROP_EXPOSURE, 0.0);
    cap2.set(CV_CAP_PROP_AUTO_EXPOSURE, 0 );
    cap2.set(CV_CAP_PROP_GAIN, 0.0);
    
    if(!calib.calibrate(cap,cap2)) {
        std::cout << "Cameras could not been calibrated. Exiting!" << std::endl;
        exit(8);
    }
    Mat f1,f2;

    int frameWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int frameHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    
    //create GUI windows
    namedWindow("F1");
    namedWindow("F2");
    setMouseCallback("F1", CallBackFunc, &f1);

    for(int keyboard=0;keyboard!=27 && cap.grab()&& cap2.grab();keyboard = waitKey(1)) {
        double t = (double)cv::getTickCount();
        cap >> f1;
        cap2 >> f2;
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        CommonUtils::displayFPS(f1,1.0/t,frameWidth*0.75);
        CommonUtils::displayFPS(f2,1.0/t,frameWidth*0.75);
        for (vector<cv::Vec3f>::const_iterator it= lines2.begin();
             it!=lines2.end(); ++it) {
            
            // draw the epipolar line between first and last column
            cv::line(f2,cv::Point(0,-(*it)[2]/(*it)[1]),
                     cv::Point(f2.cols,-((*it)[2]+(*it)[0]*f2.cols)/(*it)[1]),
                     cv::Scalar(255,255,255));
        }
        imshow("F1",f1);
        imshow("F2",f2);
    }
    cap.release();
    cap2.release();
 }
