//
//  common_utils.cpp
//  FingerTracking
//
//  Created by Furkan Tektas on 11/8/14.
//  Copyright (c) 2014 Furkan Tektas. All rights reserved.
//

#include "common_utils.h"

void CommonUtils::printCameraInfo(cv::VideoCapture m_cam){
    std::cout<<"CV_CAP_PROP_FRAME_WIDTH " << m_cam.get(CV_CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout<<"CV_CAP_PROP_FRAME_HEIGHT " << m_cam.get(CV_CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout<<"CV_CAP_PROP_FPS " << m_cam.get(CV_CAP_PROP_FPS) << std::endl;
    std::cout<<"CV_CAP_PROP_EXPOSURE " << m_cam.get(CV_CAP_PROP_EXPOSURE) << std::endl;
    std::cout<<"CV_CAP_PROP_FORMAT " << m_cam.get(CV_CAP_PROP_FORMAT) << std::endl; //deafult CV_8UC3?!
    std::cout<<"CV_CAP_PROP_CONTRAST " << m_cam.get(CV_CAP_PROP_CONTRAST) << std::endl;
    std::cout<<"CV_CAP_PROP_BRIGHTNESS "<< m_cam.get(CV_CAP_PROP_BRIGHTNESS) << std::endl;
    std::cout<<"CV_CAP_PROP_SATURATION "<< m_cam.get(CV_CAP_PROP_SATURATION) << std::endl;
    std::cout<<"CV_CAP_PROP_HUE "<< m_cam.get(CV_CAP_PROP_HUE) << std::endl;
    std::cout<<"CV_CAP_PROP_POS_FRAMES "<< m_cam.get(CV_CAP_PROP_POS_FRAMES) << std::endl;
    std::cout<<"CV_CAP_PROP_FOURCC "<< m_cam.get(CV_CAP_PROP_FOURCC) << std::endl;
    
    int ex = static_cast<int>(m_cam.get(CV_CAP_PROP_FOURCC));     // Get Codec Type- Int form
    char EXT[] = {(char)(ex & 255) , (char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24), 0};
    std::cout << "Input codec type: " << EXT << std::endl;
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