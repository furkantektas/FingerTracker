//
//  calibration.cpp
//  FingerTracking
//
//  Created by Furkan Tektas on 12/13/14.
//  Copyright (c) 2014 Furkan Tektas. All rights reserved.
//

#include "calibration.h"

using namespace cv;
using namespace std;

Calibration::Calibration() {
    boardSize = Size(9,6);
}

void Calibration::displayPairIndex(Mat& img) {
    std::ostringstream imageIndex;
    imageIndex<<stereoPairIndex<<"/"<<noOfStereoPairs;
    int baseLine = 0;
    Size txtSize = getTextSize(imageIndex.str().c_str(),FONT_HERSHEY_SIMPLEX, 1.2, 2, &baseLine);
    putText(img, imageIndex.str().c_str(), Point(img.cols - 20 - txtSize.width, img.rows - 20), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255,255,255), 2, 8);
}

bool Calibration::findChessBoard() {
    Mat inputLeft = _leftOri, inputRight = _rightOri;

    bool foundLeft = false, foundRight = false;
    cvtColor(inputLeft, inputLeft, COLOR_BGR2GRAY);
    cvtColor(inputRight, inputRight, COLOR_BGR2GRAY);
    foundLeft = findChessboardCorners(inputLeft, boardSize, cornersLeft, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
    foundRight = findChessboardCorners(inputRight, boardSize, cornersRight, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
    drawChessboardCorners(_leftOri, boardSize, cornersLeft, foundLeft);
    drawChessboardCorners(_rightOri, boardSize, cornersRight, foundRight);
    displayPairIndex(_leftOri);
    displayPairIndex(_rightOri);
    return foundLeft && foundRight;
}

void Calibration::displayImages() {
    imshow("Left Image", _leftOri);
    imshow("Right Image", _rightOri);
}

void Calibration::saveImages(Mat leftImage, Mat rightImage, int pairIndex) {
    cameraImagePoints[0].push_back(cornersLeft);
    cameraImagePoints[1].push_back(cornersRight);
    cvtColor(leftImage, leftImage, COLOR_BGR2GRAY);
    cvtColor(rightImage, rightImage, COLOR_BGR2GRAY);
    std::ostringstream leftString, rightString;
    leftString<<"left"<<pairIndex<<".jpg";
    rightString<<"right"<<pairIndex<<".jpg";
    imwrite(leftString.str().c_str(), leftImage);
    imwrite(rightString.str().c_str(), rightImage);
}

void Calibration::calibrateStereoCamera() {
    Size imageSize = _leftOri.size();
    const float squareSize = 1.f;
    vector<vector<Point3f> > objectPoints;
    objectPoints.resize(noOfStereoPairs);
    for(int i = 0; i < noOfStereoPairs; i++ )
        for(int j = 0; j < boardSize.height; j++ )
            for(int k = 0; k < boardSize.width; k++ )
                objectPoints[i].push_back(Point3f(j*squareSize, k*squareSize, 0));
    
    cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
    cameraMatrix[1] = Mat::eye(3, 3, CV_64F);
    double rms = stereoCalibrate(objectPoints, cameraImagePoints[0], cameraImagePoints[1],
                                 cameraMatrix[0], distCoeffs[0],
                                 cameraMatrix[1], distCoeffs[1],
                                 imageSize, R, T, E, F,
                                 TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-5),
                                 CALIB_FIX_ASPECT_RATIO +
                                 CALIB_ZERO_TANGENT_DIST +
                                 CALIB_SAME_FOCAL_LENGTH +
                                 CALIB_RATIONAL_MODEL +
                                 CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5 );
    cout<<"Stereo calibraton done. RMS Error: "<<rms<<"\n";
    double err = 0;
    int npoints = 0;
    vector<Vec3f> lines[2];
    for(int i = 0; i < noOfStereoPairs; i++ )
    {
        int npt = (int)cameraImagePoints[0][i].size();
        Mat imgpt[2];
        for(int k = 0; k < 2; k++ )
        {
            imgpt[k] = Mat(cameraImagePoints[k][i]);
            undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
            computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
        }
        for(int j = 0; j < npt; j++ )
        {
            double errij = fabs(cameraImagePoints[0][i][j].x*lines[1][j][0] +
                                cameraImagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
            fabs(cameraImagePoints[1][i][j].x*lines[0][j][0] +
                 cameraImagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    cout << "Average Reprojection Error: " <<  err/npoints << endl;
    FileStorage fs("intrinsics.yml", FileStorage::WRITE);
    if (fs.isOpened()) {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
        "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
        fs.release();
    }
    else
        cout<<"Error: Could not open intrinsics file.";
}

int Calibration::calibrate(VideoCapture camLeft, VideoCapture camRight) {
    if (!camLeft.isOpened() || !camRight.isOpened()) {
        cout<<"Error: Stereo Cameras not found or there is some problem connecting them. Please check your cameras.\n";
        exit(-1);
    }
    system("pwd");
    
    Mat copyImageLeft, copyImageRight;
    bool foundCornersInBothImage = false;
    namedWindow("Left Image");
    namedWindow("Right Image");
    mode = CAPTURING;
    for( ; ; ) {
        camLeft>>_leftOri;
        camRight>>_rightOri;
        if ((_leftOri.rows != _rightOri.rows) || (_leftOri.cols != _rightOri.cols)) {
            cout<<"Error: Images from both cameras are not of some size. Please check the size of each camera.\n";
            exit(-1);
        }
        _leftOri.copyTo(copyImageLeft);
        _rightOri.copyTo(copyImageRight);
        foundCornersInBothImage = findChessBoard();
        if (foundCornersInBothImage && stereoPairIndex<14) {
            int64 thisTick = getTickCount();
            int64 diff = thisTick - prevTickCount;
            if (goIn==1 || diff >= timeGap) {
                goIn=0;
                saveImages(copyImageLeft, copyImageRight, ++stereoPairIndex);
                prevTickCount = getTickCount();
            }
        }
        displayImages();
        if( 27 == waitKey(10)) {
            std::cout << "Calibration Cancelled" << std::endl;
            return 0;
        }
        
        if(stereoPairIndex == 14)
            break;
    }
    mode = CALIBRATING;
    calibrateStereoCamera();
    destroyWindow("Left Image");
    destroyWindow("Right Image");
    return 1;
}

const Mat& Calibration::getR() const {
    return R;
}

const Mat& Calibration::getT() const {
    return T;
}

const Mat& Calibration::getE() const {
    return E;
}

const Mat& Calibration::getF() const {
    return F;
}

const Mat& Calibration::getCameraMatrix1() const {
    return cameraMatrix[0];
}

const Mat& Calibration::getCameraMatrix2() const {
    return cameraMatrix[1];
}

const Mat& Calibration::getDistCoeffs1() const {
    return distCoeffs[0];
}

const Mat& Calibration::getDistCoeffs2() const {
    return distCoeffs[1];
}