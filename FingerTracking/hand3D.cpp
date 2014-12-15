//#include <Python/Python.h>
#include "hand3D.h"
#include "calibration.h"

Hand3D::~Hand3D() {
    destroyWindow(windowNameStereo);
    destroyWindow(windowName3D);
}

void Hand3D::setFrames(Mat& frameLeft, Mat& frameRight) {
    leftHand.setFrame(frameLeft);
    rightHand.setFrame(frameRight);
    imshow(windowNameStereo, frameLeft);
}

void Hand3D::find() {
    leftHand.find();
    rightHand.find();



    if(leftHand.getFingerCount() == rightHand.getFingerCount()  &&
        leftHand.getFingerCount()> 1) {

        cerr << "LeftHand Fingers: ";


        cerr << "Begin left fingers" << endl;

        list<Point2f> leftFingerList = leftHand.getFingers();
        std::vector<Point2f> left{ std::begin(leftFingerList), std::end(leftFingerList) };


        cerr << "Begin right fingers" << endl;

        list<Point2f> rightFingerList = rightHand.getFingers();
        std::vector<Point2f> right{ std::begin(rightFingerList), std::end(rightFingerList) };


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
            projectPoints(resultPoints, calibration.getR(),calibration.getT(),calibration.getCameraMatrix1()
                          ,calibration.getDistCoeffs1(), proj);
            cerr << "begin circle" << endl;
            Mat reproj_canvas(leftHand.getFrame());
            for(size_t i=0; i<proj.size(); i++){
                Point2f pt = proj[i];
                circle(reproj_canvas, pt, 5, Scalar(255, 0, 0), 3);
            }
            cerr << "begin imshow" << endl;
            imshow(windowName3D, reproj_canvas);
        }
    };

}

