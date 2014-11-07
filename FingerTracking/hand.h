//
//  hand.h
//  FingerTracking
//
//  Created by Furkan Tektas on 11/8/14.
//  Copyright (c) 2014 Furkan Tektas. All rights reserved.
//

#ifndef __FingerTracking__hand__
#define __FingerTracking__hand__

#include <opencv2/opencv.hpp>
#include "common_utils.h"
#include "datatypes.h"

using namespace cv;

class Hand{
public:
    Hand(Mat& frame) : mFrame(frame) {}
    void process_frame();
    void setFrame(Mat& frame);
    void refresh();
private:
    Mat& mFrame;
    Mat mThreshFrame;
    vector<Line> fingerLines;
    vector<Point> handPolygon;
    vector<Point> handContour;
    vector<Point> fingers;
    Point palmCenter;
    Rect handBoundingRect;
    enum Hand_e hand;
    const Point* labeledFingers[5];
    
    
    
    int frameCount = 0;
    int totalHandGravity = 0;
    
    void findConvexHull(const Mat& img, Mat& drawingFrame);
    bool fingerDistanceComparator(const Point& f1, const Point& f2);
    
    float pointDistance(const Point& a, const Point& b);
    float getLineAngle(const Point& p1, const Point& p2);
    int findLargestContour(const vector<vector<Point>>& contours);
    void filterConvexes(vector<Vec4i>& convDefect, const vector<Point>& contours, const Rect& boundingRect);
    void filterFingers();
    void findFingerPoints(const vector<Vec4i>& convDefect, const vector<Point>& contours, const Rect& boundingRect);
    void findPalmCenter(const vector<Vec4i>& convDefect, const vector<Point>& contours);
    
    
    void printFingerCount(Mat& img, int fingerCount);
    bool isHand(const vector<Point>& contours, const vector<Vec4i>& convDefect);
    void drawConvexity(Mat& drawing, const vector<Vec4i>& convDefect, const vector<Point>& contours);
    void findHandOrientation();
    void findFingerLines();
    void drawFingerLines(Mat& drawing);
    void findWhichHand();
    bool isFingerOnLeft(const Point& f1);
    bool isFingerOnTop(const Point& f1);
    float fingerDistanceFromPalmCenter(const Point& f);
    bool palmCenterDistComparator (const Point* p1, const Point* p2);
    bool lineAngleCompare(const Line& l1, const Line& l2);

};
#endif /* defined(__FingerTracking__hand__) */
