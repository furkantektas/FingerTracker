//
//  hand.cpp
//  FingerTracking
//
//  Created by Furkan Tektas on 11/8/14.
//  Copyright (c) 2014 Furkan Tektas. All rights reserved.
//

#include <numeric> // accumulate
#include "hand.h"
#include "colors.h"

void Hand::setFrame(Mat& frame) {
    mFrame = frame;
}

bool Hand::isFingerOnLeft(const Point& f1) const {
    return f1.x < palmCenter.x;
}

bool Hand::isFingerOnTop(const Point& f1) const {
    return f1.y < palmCenter.y;
}

float Hand::fingerDistanceFromPalmCenter(const Point& f) const {
    return CommonUtils::pointDistance(f, palmCenter);
}

void Hand::findWhichHand() {
    std::list<Point>::const_iterator fItr = fingers.cbegin();
    int maxDist = 0;
    const Point* farthestFinger = 0;
    
    if(fingers.size() < 1)
        return;
    
    // constraint: thumb should be the farthest finger from palm center
    // TODO: check thumb's position
    while(fItr != fingers.cend()) {
        int distance = CommonUtils::pointDistance(palmCenter, *fItr);
        if(distance > maxDist) {
            maxDist = distance;
            farthestFinger = &(*fItr);
        }
        ++fItr;
    }
    
    // constraint: palm center should be above than thumb finger
    if(farthestFinger->y < palmCenter.y) {
        ++frameCount;
        
        // note that camera is looking from below.
        if(farthestFinger->x < palmCenter.x)
            totalHandGravity += RIGHT;
        else
            totalHandGravity += LEFT;
        
        hand = (enum Hand_e) ( ( ( (float)totalHandGravity) / frameCount ) > 0.4999);
    }
}

// clear variables
void Hand::refresh() {
    fingers.clear();
    fingerLines.clear();
    handPolygon.clear();
    handContour.clear();
}

void Hand::filterFingers() {
    double mean = 0, std_dev = 0;
    
    if(fingers.size() < 3)
        return;
    
    for(std::list<Point>::const_iterator fItr = fingers.begin(); fItr != fingers.end(); ++fItr)
        mean += fingerDistanceFromPalmCenter(*fItr);
    mean /= fingers.size();
    
    std::vector<double> squares ;
    for(std::list<Point>::const_iterator fItr = fingers.cbegin(); fItr != fingers.cend(); ++fItr)
        squares.push_back( std::pow( fingerDistanceFromPalmCenter(*fItr) - mean , 2 ) ) ;
    std_dev = std::sqrt( std::accumulate( squares.begin( ) , squares.end( ) , 0 ) / squares.size( ) ) ;
    
    double minFingerDist = mean - 2 * std_dev,
    maxFingerDist = mean + 2 * std_dev;
    
    // filtering fingers
    for(std::list<Point>::const_iterator fItr = fingers.cbegin(); fItr != fingers.cend(); ++fItr) {
        double dist = fingerDistanceFromPalmCenter(*fItr);
        if(dist < minFingerDist || dist > maxFingerDist)
            fItr = fingers.erase(fItr);
    }
    
    // finding thumb
}


void Hand::process_frame() {
    if(!mFrame.data)
        return;
    cvtColor(mFrame, mThreshFrame, COLOR_RGB2GRAY);
    threshold(mThreshFrame,mThreshFrame,70,255,THRESH_TOZERO);
    threshold(mThreshFrame,mThreshFrame,0,255,THRESH_BINARY + THRESH_OTSU);
    
    int erosion_size = 1;
    const Mat element = getStructuringElement(cv::MORPH_ELLIPSE,
                                              cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                              cv::Point(erosion_size, erosion_size) );
    dilate(mThreshFrame, mThreshFrame, element);
    dilate(mThreshFrame, mThreshFrame, element);
    findConvexHull();
}

void Hand::findConvexHull(){
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    
    Mat thresholded_img = mThreshFrame.clone();
    findContours( thresholded_img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
    
    if(contours.size() < 1)
        return;
    
    int handInd = findLargestContour(contours);
    handContour = contours[handInd];
    handBoundingRect = boundingRect(contours[handInd]);
    
    vector<int>handHullI( contours[handInd].size() );
    
    convexHull( Mat(contours[handInd]), handPolygon, true);
    convexHull( Mat(contours[handInd]), handHullI, false);
    
    approxPolyDP( Mat(handPolygon), handPolygon,11,true);
    if (contours[handInd].size() > 3 ) {
        convexityDefects(contours[handInd], handHullI, handConvDefect);
    }
    
    
    findPalmCenter();
    filterConvexes();
    findFingerPoints();
    findFingerLines();
    findHandOrientation();
    findWhichHand();
    filterFingers();
    printFingerCount((int) fingers.size());
    drawFingerLines();
    drawConvexity();
//    rectangle(drawingFrame, handBoundingRect.tl(), handBoundingRect.br(), BLUE);
    
    drawHandPolygon();
    return;
}

void Hand::drawHandPolygon() const {
    for(int i=0;i<handPolygon.size();++i) {
        line(mFrame,handPolygon[i], handPolygon[(i+1) % handPolygon.size()],LIGHTGRAY, 2 );
    }
}

int Hand::findLargestContour(const vector<vector<Point>>& contours) {
    int largest_contour_index = -1;
    double largest_area = 0;
    for(int i = 0; i < contours.size(); i++) {
        double a = contourArea(contours[i], false);
        if(a > largest_area){
            largest_area = a;
            largest_contour_index = i;
        }
    }
    return largest_contour_index;
}

void Hand::filterConvexes() {
    int tolerance =  handBoundingRect.height/8;
    vector<Vec4i>::iterator d = handConvDefect.begin();
    
    int maxDegree = 110;
    while( d!=handConvDefect.end() ) {
        Vec4i& v = (*d);
        Point p1( handContour[v[0]]),
        p2( handContour[v[1]]),
        p3( handContour[v[2]]);
        int depth = v[3]/256;
        int angleBetween = abs(CommonUtils::getLineAngle(p1, p3) - CommonUtils::getLineAngle(p2, p3));
        if(depth < tolerance || angleBetween > maxDegree)
            d = handConvDefect.erase(d);
        else
            ++d;
    }
}

void Hand::findFingerPoints() {
    int handRectDiagonalLength = sqrt(handBoundingRect.width*handBoundingRect.width+handBoundingRect.height*handBoundingRect.height);
    int distanceTolerance = handRectDiagonalLength/5;
    
    int angleTolerance = 80;
    
    fingers.clear();
    
    for(vector<Point>::const_iterator pi = handPolygon.cbegin(); pi!=handPolygon.cend(); ++pi){
        const Point* p0 = &(*(pi-1)); // previous corner
        const Point* p1 = &(*pi);     // current corner
        const Point* p2 = &(*(pi+1)); // next corner
        
        if( (pi+1) == handPolygon.cend())
            p2 = &(*handPolygon.cbegin());

        if(pi == handPolygon.cbegin())
            p0 = &(*handPolygon.cend());
        
        for(vector<Vec4i>::const_iterator d = handConvDefect.cbegin(); d!=handConvDefect.cend(); ++d){
            const Vec4i& v1 = (*d);
            const Point* pt1Start = &handContour[v1[0]];
            const Point* pt1End   = &handContour[v1[1]];

            const Point* pt2Start = 0;
            const Point* pt2End   = 0;
            bool isMiddleFingers = false;
            if((d+1) != handConvDefect.cend()) {
                isMiddleFingers = true;
                const Vec4i& v2 = (*(d+1));
                pt2Start = &handContour[v2[0]];
                pt2End   = &handContour[v2[1]];
            }
            
            int distanceStart = CommonUtils::pointDistance(*pt1Start, *p1);
            int distanceEnd = CommonUtils::pointDistance(*pt1End, *p1);
            
            const Point* finger = 0;
            
            // detecting finger by intersection convexity defect points and handpolygon points
            if((distanceStart < distanceTolerance || distanceEnd < distanceTolerance))
                finger = p1;
            else if(isMiddleFingers && CommonUtils::pointDistance(*pt1Start, *pt2End) < distanceTolerance)
                finger = pt1Start;
            else if(isMiddleFingers && CommonUtils::pointDistance(*pt2Start, *pt1End) < distanceTolerance)
                finger = pt2Start;
            if(finger)
                addFinger(*finger);
        }
        
        std::cout<< CommonUtils::getLineAngle(*p0, *p1) << "      \t     " << CommonUtils::getLineAngle(*p1, *p2) << "     \t     " << abs(CommonUtils::getLineAngle(*p0, *p1) - CommonUtils::getLineAngle(*p1,*p2)) << std::endl;
        int angle1 = CommonUtils::getLineAngle(*p0, *p1);
        int angle2 = CommonUtils::getLineAngle(*p1, *p2);

        // angleN%90 is for eliminating wrist corners
        if((angle1%90) && (angle2%90) && (abs(angle1 - angle2) > angleTolerance)) {
            std::cout << "\t\t\t\t\t" << "Adding: " << angle1 << " " << angle2 << std::endl << std::endl;
            addFinger(*p1);
        }
    }
    
}

void Hand::addFinger(const Point& finger) {
    int handRectDiagonalLength = sqrt(handBoundingRect.width*handBoundingRect.width+handBoundingRect.height*handBoundingRect.height);
    int minFingerAffinity = handRectDiagonalLength/10;
    // avoid detecting a finger multiple times
    bool isFound = false;
    for(std::list<Point>::const_iterator fi = fingers.cbegin(); fi != fingers.cend() && !isFound; ++fi) {
        isFound = isFound || (CommonUtils::pointDistance(finger,*fi) < minFingerAffinity);
    }
    if(!isFound)
        fingers.push_back(finger);
}

bool Hand::palmCenterDistComparator (const Point* p1, const Point* p2) const {
    return (CommonUtils::pointDistance(palmCenter, *p1) < CommonUtils::pointDistance(palmCenter, *p2));
}


void Hand::findPalmCenter() {
    Moments mu;
    
    mu = moments( handContour, false );
    palmCenter = Point( mu.m10/mu.m00 , mu.m01/mu.m00 );
    
    if(handConvDefect.size() < 1 || handContour.size() < 1)
        return;
    
    
    vector<const Point*> points;
    vector<Vec4i>::const_iterator i;
    for(i = handConvDefect.cbegin(); i != handConvDefect.cend(); ++i) {
        const Vec4i& v = (*i);
        points.push_back( &handContour[v[0]]);
        points.push_back( &handContour[v[1]]);
        points.push_back( &handContour[v[2]]);
    }

    auto comparator = std::bind(&Hand::palmCenterDistComparator, this, std::placeholders::_1, std::placeholders::_2);
    std::sort(points.begin(),points.end(),comparator);
    
    int pointCount = 0;
    int mean = 0;
    for(vector<const Point*>::iterator i = points.begin(); i != points.end(); ++i) {
        int newDist = CommonUtils::pointDistance(palmCenter, **i);
        if(mean != 0 && abs(newDist - mean/pointCount) > handBoundingRect.height/10)
            break;
        
        mean += newDist;
        ++pointCount;
    }
    
    mean /= pointCount;
    if(mean > 0) {
        circle(mFrame,palmCenter,mean,RED,8,8);
    }
}

void Hand::printFingerCount(int fingerCount) const {
    char c[255];
    sprintf(c,"Finger #%d Hand: %s", fingerCount, (hand == LEFT) ? "LEFT" : "RIGHT");
    CommonUtils::putTextWrapper(mFrame, c);
}

bool Hand::isHand() const {
    return true;
}

void Hand::drawConvexity() const {
    vector<Vec4i>::const_iterator d = handConvDefect.cbegin();
    while( d!=handConvDefect.cend() ) {
        const Vec4i& v=(*d);
        int startidx=v[0]; Point ptStart( handContour[startidx] );
        int endidx=v[1]; Point ptEnd( handContour[endidx] );
        int faridx=v[2]; Point ptFar( handContour[faridx] );
        //        float depth = v[3] / 256.0;
        
        if(v[2] > 100) {
            line( mFrame, ptStart, ptFar, PURPLE, 1 );
            line( mFrame, ptEnd, ptFar, PURPLE, 1 );
            circle( mFrame, ptFar, 3, GREEN, 3 );
        }
        d++;
    }
}

bool Hand::lineAngleCompare(const Line& l1, const Line& l2) {
    return l1.angle < l2.angle;
}

void Hand::findHandOrientation() {
    double mean = 0;
    
    if(fingerLines.size() < 2)
        return;
    
    for(std::list<Line>::const_iterator lItr = fingerLines.cbegin(); lItr != fingerLines.cend(); ++lItr)
        mean += lItr->angle;
    mean /= fingerLines.size();
    
    std::vector<double> squares ;
    double min = 360;
    const Line* middle = 0;
    for(std::list<Line>::const_iterator lItr = fingerLines.cbegin(); lItr != fingerLines.cend(); ++lItr) {
        double angle = abs(lItr->angle - mean);
        squares.push_back( std::pow( angle , 2 ) ) ;
        if(angle < min) {
            min = angle;
            middle = &(*lItr);
        }
    }
    
    circle(mFrame,middle->end,10,YELLOW,4);
    
    int tolerance = 120;
    
    // Eliminating mismatched fingers and fingerLines
    for(std::list<Line>::iterator lItr = fingerLines.begin(); lItr != fingerLines.end(); ++lItr) {
        double angle = abs(lItr->angle - middle->angle);
        if(angle > tolerance) {
//            deleteFinger(lItr->end);
//            lItr = fingerLines.erase(lItr);
            circle(mFrame, lItr->end, 11, WHITE, 11);
            std::cout << "Finger Eliminated" << std::endl;
        }
    }
    
}

void Hand::deleteFinger(const Point& finger) {
    
    for(std::list<Point>::iterator fItr = fingers.begin(); fItr != fingers.end(); ++fItr) {
        if(finger == *fItr) {
            fingers.erase(fItr);
            return;
        }
    }
}

void Hand::findFingerLines() {
    if(handBoundingRect.height < 50 || handBoundingRect.width < 50)
        return;
    
    // if palm center is outside of handpolygon
    if(pointPolygonTest(handContour, palmCenter, false) < 0.01)
        return;
    
    fingerLines.clear();
    std::list<Point>::const_iterator v = fingers.cbegin();
    while(v!=fingers.cend()) {
        const Point& endPoint = (*v);
        fingerLines.push_back(Line{palmCenter, endPoint, CommonUtils::getLineAngle(endPoint, palmCenter)});
        ++v;
    }
}

void Hand::drawFingerLines() const {
    circle(mFrame,palmCenter,5,RED,15);
    std::list<Line>::const_iterator d = fingerLines.cbegin();
    while( d!=fingerLines.cend() ) {
        const Line& l = (*d++);
        
        char buff[100];
        sprintf(buff, "%.2f", l.angle);
        
        CommonUtils::putTextWrapper(mFrame, buff, l.end.x+50, l.end.y+50);
        line( mFrame, l.start, l.end, BLUE, 2 );
        circle(mFrame,l.end,7,CYAN,4);
    }
}