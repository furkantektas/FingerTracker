#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include "opencv2/calib3d/calib3d.hpp"
#include "common_utils.h"
#include "datatypes.h"
#include "calibration.h"
#include "hand3D.h"

using namespace cv;

Calibration calib;
std::vector<cv::Vec3f> lines1, lines2;
std::vector<cv::Point2f> points1, points2;
bool showEpipLines = true;
Mat f1,f2;
Hand3D *hand;

void CallBackFunc(int event, int x, int y, int flags, void* userdata) {
    if ( event == EVENT_LBUTTONDOWN ) {
        Point2f point(x,y);
        std::vector<cv::Point2f> *points;

        if(userdata == &f1)
            points = &points1;
        else
            points = &points2;
        points->push_back(point);
    }
}

void process_keyevent(char key) {
    if(key == 'r' || key == 'R') {
        lines1.clear();
        lines2.clear();
        points1.clear();
        points2.clear();
    }
    if(key == 'l' || key == 'L')
        showEpipLines = !showEpipLines;

}


void drawLines(Mat& frame, std::vector<cv::Vec3f> lines) {
    for (vector<cv::Vec3f>::const_iterator it= lines.begin();
         it!=lines.end(); ++it) {

        // draw the epipolar line between first and last column
        cv::line(frame,cv::Point(0,-(*it)[2]/(*it)[1]),
                cv::Point(frame.cols,-((*it)[2]+(*it)[0]*frame.cols)/(*it)[1]),
                cv::Scalar(255,255,255));
    }
}

void drawEpipLines() {
    if(showEpipLines) {
        if(hand->getRightFingers().size() > 0) {
            std::vector<cv::Vec3f> fingerLines;
            cv::computeCorrespondEpilines(cv::Mat(hand->getRightFingers()), 2, calib.getF(), fingerLines);
            drawLines(f1, fingerLines);
        }
        if(hand->getLeftFingers().size() > 0) {
            std::vector<cv::Vec3f> fingerLines;
            cv::computeCorrespondEpilines(cv::Mat(hand->getLeftFingers()),1,calib.getF(),fingerLines);
            drawLines(f2, fingerLines);
        }
        if(points1.size() > 0) {
            cv::computeCorrespondEpilines(cv::Mat(points1),1,calib.getF(),lines1);
            drawLines(f2, lines1);
        }
        if(points2.size() > 0) {
            cv::computeCorrespondEpilines(cv::Mat(points2),2,calib.getF(),lines2);
            drawLines(f1, lines2);
        }
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

    int frameWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int frameHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

    hand = new Hand3D(calib,f1,f2,Size(frameWidth,frameHeight));

    //create GUI windows
    namedWindow("F1");
    namedWindow("F2");
    setMouseCallback("F1", CallBackFunc, &f1);
    setMouseCallback("F2", CallBackFunc, &f2);

    for(int key =0; key !=27 && cap.grab()&& cap2.grab(); key = waitKey(1)) {
        process_keyevent(key);
        double t = (double)cv::getTickCount();
        cap >> f1;
        cap2 >> f2;
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

        hand->setFrames(f1, f2);
        hand->find();

        CommonUtils::displayFPS(f1,1.0/t,frameWidth*0.75);
        CommonUtils::displayFPS(f2,1.0/t,frameWidth*0.75);
        drawEpipLines();

        imshow("F1",f1);
        imshow("F2",f2);
    }
    cap.release();
    cap2.release();
    free(hand);
 }
