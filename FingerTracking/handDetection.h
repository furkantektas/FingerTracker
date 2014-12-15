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

class HandDetection {
public:
    HandDetection(Mat& frame, Size frameSize) : mFrame(frame), frameWidth(frameSize.width), frameHeight(frameSize.height) {}
    void find();
    void setFrame(Mat& frame);
    void refresh();
    std::list<Point2f> getFingers() const;
    const Point& getPalmCenter() const;
    int getFingerCount() const;
    Mat& getFrame();
private:
    Mat& mFrame;
    Mat mThreshFrame;
    std::list<Line> fingerLines;
    vector<Point> handPolygon;
    vector<Point> handContour;
    std::list<Point2f> fingers;
    const Line* middleFinger = 0;
    Point palmCenter;
    Rect handBoundingRect;
    vector<Vec4i> handConvDefect;
    enum Hand_e hand;
    const Point* labeledFingers[5];
    int frameCount = 0;
    int totalHandGravity = 0;
    int frameWidth;
    int frameHeight;
    int palmRadius = 0;

    void process_frame();
    void addFinger(const Point2f& finger);
    void deleteFinger(const Point2f& finger);
    std::list<Point2f>::const_iterator deleteFinger(const std::list<Point2f>::const_iterator finger);
    
    // Algorithm Functions
    void findConvexHull();
    void filterConvexes();
    void filterFingers();
    void findFingerPoints();
    void findPalmCenter();
    void findHandOrientation();
    void findWhichHand();
    
    // Drawing Functions
    void drawPalmCenter() const;
    void drawHandPolygon() const;
    void drawConvexity() const;
    void drawFingerLines() const;
    void drawFingerIds() const;
    void drawFingerPrints() const;
    void printFingerCount(int fingerCount) const;
    
    // Helper functions
    float fingerDistanceFromPalmCenter(const Point& f) const;
    static bool lineAngleCompare(const Line& l1, const Line& l2);
    static int findLargestContour(const vector<vector<Point>>& contours);
    bool palmCenterDistComparator (const Point* p1, const Point* p2) const;
    bool isHand() const;
    bool fingerDistanceComparator(const Point& f1, const Point& f2);
    bool isFingerOnLeft(const Point& f1) const;
    bool isFingerOnTop(const Point& f1) const;
    bool isFingerOnEdge(const Point& finger, int tolerance = 5) const;
};
#endif /* defined(__FingerTracking__hand__) */
