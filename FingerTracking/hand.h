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
#include <list>
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
    std::list<Line> fingerLines;
    vector<Point> handPolygon;
    vector<Point> handContour;
    std::list<Point> fingers;
    Point palmCenter;
    Rect handBoundingRect;
    vector<Vec4i> handConvDefect;
    enum Hand_e hand;
    const Point* labeledFingers[5];
    int frameCount = 0;
    int totalHandGravity = 0;
    void deleteFinger(const Point& finger);
    void findConvexHull();
    bool fingerDistanceComparator(const Point& f1, const Point& f2);
    void drawHandPolygon() const;

    void filterConvexes();
    void filterFingers();
    void findFingerPoints();
    void findPalmCenter();
    void addFinger(const Point& finger);
    
    void printFingerCount(int fingerCount) const;
    bool isHand() const;
    void drawConvexity() const;
    void findHandOrientation();
    void findFingerLines();
    void drawFingerLines() const;
    void findWhichHand();
    bool isFingerOnLeft(const Point& f1) const;
    bool isFingerOnTop(const Point& f1) const;
    float fingerDistanceFromPalmCenter(const Point& f) const;
    bool palmCenterDistComparator (const Point* p1, const Point* p2) const;
    
    static bool lineAngleCompare(const Line& l1, const Line& l2);
    static int findLargestContour(const vector<vector<Point>>& contours);
};
#endif /* defined(__FingerTracking__hand__) */
