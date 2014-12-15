//#include <Python/Python.h>
#include "hand3D.h"
#include "calibration.h"

Hand3D::~Hand3D() {
    destroyWindow(windowNameStereo);
    destroyWindow(windowName3D);
}

std::vector<Point2f> Hand3D::getLeftFingers() const {
    return leftHand.getFingers();
}

std::vector<Point2f> Hand3D::getRightFingers() const {
    return rightHand.getFingers();
};

void Hand3D::setFrames(Mat& frameLeft, Mat& frameRight) {
    leftHand.setFrame(frameLeft);
    rightHand.setFrame(frameRight);
    scene = frameLeft.clone();
    imshow(windowNameStereo, scene);
}

void Hand3D::find() {
    leftHand.find();
    rightHand.find();



    if(leftHand.getFingerCount() == rightHand.getFingerCount()  &&
        leftHand.getFingerCount()> 1) {

        cerr << "LeftHand Fingers: ";


        cerr << "Begin left fingers" << endl;

        std::vector<Point2f> left = leftHand.getFingers();


        cerr << "Begin right fingers" << endl;

        std::vector<Point2f> right = rightHand.getFingers();


        for(vector<Point2f>::const_iterator it = left.cbegin();
            it != left.cend();
            ++it)
            cerr << *it << ", ";
        cerr << endl << endl;
        cerr << "RightHand Fingers: ";

        for(vector<Point2f>::const_iterator it = right.cbegin();
            it != right.cend();
            ++it)
            cerr << *it  << ", ";
        cerr << endl << endl;

        Mat left_mat = Mat(left).reshape(1, left.size()).t();
        Mat right_mat= Mat(right).reshape(1, right.size()).t();
        cerr << "Left fingers: #" << left.size() << " right fingers: #" << right.size() << endl;


        cv::Mat R0, R1, P0, P1, Q;
        Rect validRoi[2];
        cv::stereoRectify( calibration.getCameraMatrix1(), calibration.getDistCoeffs1(),
                calibration.getCameraMatrix2(), calibration.getDistCoeffs2(),
                leftHand.getFrame().size(), calibration.getR(), calibration.getT(), R0, R1, P0, P1, Q,
                CALIB_ZERO_DISPARITY, 1, rightHand.getFrame().size(), &validRoi[0], &validRoi[1]);
        cerr << "Q: "<< Q << endl;


        cv::Mat pnts3D;
        cv::triangulatePoints( P0, P1, left, right, pnts3D );
        cerr << "triangulatePoints: "<< pnts3D << endl;

        cv::Mat t = pnts3D.t();
        cv::Mat pnts3DT = cv::Mat(left.size(), 1, CV_32FC4, t.data );
        cerr << "pnts3DT: "<< pnts3DT << endl;

        cv::Mat resultPoints;
        cv::convertPointsFromHomogeneous( pnts3DT, resultPoints );
        cerr << "resultPoints: "<< resultPoints << endl;

        cv::Mat rvec(1,3,cv::DataType<double>::type);
        cv::Mat tvec(1,3,cv::DataType<double>::type);
        cv::Mat rotationMatrix(3,3,cv::DataType<double>::type);


        cerr << "begin solvepnp" << endl;
        if(solvePnP(resultPoints, left, calibration.getCameraMatrix1(), calibration.getDistCoeffs1(), rvec, tvec,  false, CV_EPNP)) {
            cerr << "rvec" << rvec << endl;
            cerr << "tvec" << tvec << endl;
            cerr << "begin rodrigues" << endl;
            cv::Rodrigues(rvec,rotationMatrix);
            cerr << "rodrigues" << rotationMatrix << endl;
            vector<cv::Point2f> proj;

            cerr << "begin projectpoints" << endl;
            projectPoints(resultPoints, rvec,tvec,calibration.getCameraMatrix1()
                          ,calibration.getDistCoeffs1(), proj);
            cerr << "Proj:" << proj << endl;
            cerr << "begin circle" << endl;

            for(size_t i=0; i<proj.size(); i++){
                circle(scene, proj[i], 5, Scalar(255, 255, 255), 5);
            }
            cerr << "begin imshow" << endl;
            imshow(windowName3D, scene);
        }
    };

}

