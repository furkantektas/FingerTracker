#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <iostream>
#include <iomanip>
#include <thread>
#include "common_utils.h"
#include "datatypes.h"
#include "hand.h"

using namespace cv;
using namespace std;

int windowHeight = 0;
int windowWidth = 0;

int main(int argc, char** argv) {
    VideoCapture cap(0);
    if(!cap.isOpened()) // check if we succeeded
        return -1;
    CommonUtils::setCameraResolution(cap, 320, 240);
    cap.set(CV_CAP_PROP_EXPOSURE, 0.0);
    cap.set(CV_CAP_PROP_AUTO_EXPOSURE, 0 );
    cap.set(CV_CAP_PROP_GAIN, 0.0);
    
    windowWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    windowHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    
    CommonUtils::printCameraInfo(cap);
    //create GUI windows
    namedWindow("Raw Frame");
    namedWindow("Hand Frame");

    Mat frame, rawFrame;
    Hand h(frame);
    for(int keyboard=0;keyboard!=27 && cap.grab();keyboard = waitKey(1)) {
        double t = (double)cv::getTickCount();
        cap >> frame;
        
        // when working with video files sometimes rawFrame becomes null
        // to avoid that check if rawFrame.data is not null
        if(frame.data) {
            
            rawFrame = frame.clone();
            h.setFrame(frame);
            h.process_frame();

            // calculating elapsed time in sec
            t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
            CommonUtils::displayFPS(frame,1.0/t,windowWidth-100);
            
            imshow("Hand Frame", frame);
            imshow("Raw Frame", rawFrame);
            h.refresh();
        }
    }
    frame.release();
    rawFrame.release();
 }


