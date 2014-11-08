//
//  common_utils.cpp
//  FingerTracking
//
//  Created by Furkan Tektas on 11/8/14.
//  Copyright (c) 2014 Furkan Tektas. All rights reserved.
//

#include "common_utils.h"

void CommonUtils::printCameraInfo(cv::VideoCapture cam){
    std::cout<<"CV_CAP_PROP_FRAME_WIDTH " << cam.get(CV_CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout<<"CV_CAP_PROP_FRAME_HEIGHT " << cam.get(CV_CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout<<"CV_CAP_PROP_FPS " << cam.get(CV_CAP_PROP_FPS) << std::endl;
    std::cout<<"CV_CAP_PROP_EXPOSURE " << cam.get(CV_CAP_PROP_EXPOSURE) << std::endl;
    std::cout<<"CV_CAP_PROP_FORMAT " << cam.get(CV_CAP_PROP_FORMAT) << std::endl; //deafult CV_8UC3?!
    std::cout<<"CV_CAP_PROP_CONTRAST " << cam.get(CV_CAP_PROP_CONTRAST) << std::endl;
    std::cout<<"CV_CAP_PROP_BRIGHTNESS "<< cam.get(CV_CAP_PROP_BRIGHTNESS) << std::endl;
    std::cout<<"CV_CAP_PROP_SATURATION "<< cam.get(CV_CAP_PROP_SATURATION) << std::endl;
    std::cout<<"CV_CAP_PROP_HUE "<< cam.get(CV_CAP_PROP_HUE) << std::endl;
    std::cout<<"CV_CAP_PROP_POS_FRAMES "<< cam.get(CV_CAP_PROP_POS_FRAMES) << std::endl;
    std::cout<<"CV_CAP_PROP_FOURCC "<< cam.get(CV_CAP_PROP_FOURCC) << std::endl;
    
    int ex = static_cast<int>(cam.get(CV_CAP_PROP_FOURCC));     // Get Codec Type- Int form
    char EXT[] = {(char)(ex & 255) , (char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24), 0};
    std::cout << "Input codec type: " << EXT << std::endl;
}

void CommonUtils::setCameraResolution(cv::VideoCapture cam, int width, int height) {
    cam.set(CV_CAP_PROP_FRAME_WIDTH, width);
    cam.set(CV_CAP_PROP_FRAME_HEIGHT, height);
}

void CommonUtils::displayFPS(cv::Mat& frame, double fps, int x_pos) {
    char buf[20];
    sprintf(buf, "%.2f FPS",fps);
    putTextWrapper(frame,buf,x_pos);
}


std::string CommonUtils::getImageType(int number) {
    // find type
    int imgTypeInt = number % 8;
    std::string imgTypeString;
    
    switch (imgTypeInt) {
        case 0:
            imgTypeString = "8U";
            break;
        case 1:
            imgTypeString = "8S";
            break;
        case 2:
            imgTypeString = "16U";
            break;
        case 3:
            imgTypeString = "16S";
            break;
        case 4:
            imgTypeString = "32S";
            break;
        case 5:
            imgTypeString = "32F";
            break;
        case 6:
            imgTypeString = "64F";
            break;
        default:
            break;
    }
    
    // find channel
    int channel = (number/8) + 1;
    
    std::stringstream type;
    type << "CV_"<< imgTypeString << "C" << channel;
    
    return type.str();
}

void CommonUtils::putTextWrapper(cv::Mat& img, const char* text, int x, int y) {
    static const int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    static const double fontScale = 0.5;
    static const int thickness = 1;
    cv::Point textOrg(x,y);
    cv::putText(img, text, textOrg, fontFace, fontScale, cv::Scalar::all(255), thickness,8);
}

float CommonUtils::pointDistance(const cv::Point& a, const cv::Point& b){
    int dx = a.x - b.x;
    int dy = a.y - b.y;
    return sqrt(dx*dx + dy*dy);
}

float CommonUtils::getLineAngle(const cv::Point& p1, const cv::Point& p2){
    double dy = p1.y - p2.y;
    double dx = p1.x - p2.x;
    double theta = atan2(dy, dx);
    theta = theta*180/CV_PI + 180; // rads to degs
    return theta;
}