//
//  hand.cpp
//  FingerTracking
//
//  Created by Furkan Tektas on 11/8/14.
//  Copyright (c) 2014 Furkan Tektas. All rights reserved.
//

#include <numeric> // accumulate
#include "hand.h"

void Hand::setFrame(Mat& frame) {
    mFrame = frame;
}

bool Hand::isFingerOnLeft(const Point& f1) {
    return f1.x < palmCenter.x;
}

bool Hand::isFingerOnTop(const Point& f1) {
    return f1.y < palmCenter.y;
}

float Hand::fingerDistanceFromPalmCenter(const Point& f) {
    return pointDistance(f, palmCenter);
}

void Hand::findWhichHand() {
    vector<Point>::const_iterator fItr = fingers.cbegin();
    int maxDist = 0;
    const Point* farthestFinger = 0;
    
    if(fingers.size() < 1)
        return;
    
    // constraint: thumb should be the farthest finger from palm center
    // TODO: check thumb's position
    while(fItr != fingers.cend()) {
        int distance = pointDistance(palmCenter, *fItr);
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
    vector<Point>::iterator fItr;
    double mean = 0, std_dev = 0;
    
    if(fingers.size() < 3)
        return;
    
    for(fItr = fingers.begin(); fItr != fingers.end(); ++fItr)
        mean += fingerDistanceFromPalmCenter(*fItr);
    mean /= fingers.size();
    
    std::vector<double> squares ;
    for( fItr = fingers.begin(); fItr != fingers.end(); ++fItr)
        squares.push_back( std::pow( fingerDistanceFromPalmCenter(*fItr) - mean , 2 ) ) ;
    std_dev = std::sqrt( std::accumulate( squares.begin( ) , squares.end( ) , 0 ) / squares.size( ) ) ;
    
    double minFingerDist = mean - 2 * std_dev,
    maxFingerDist = mean + 2 * std_dev;
    
    // filtering fingers
    for(fItr = fingers.begin(); fItr != fingers.end(); ++fItr) {
        double dist = fingerDistanceFromPalmCenter(*fItr);
        if(dist < minFingerDist || dist > maxFingerDist)
            fItr = fingers.erase(fItr) - 1; // TODO : exception
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
    findConvexHull(mThreshFrame, mFrame);
}

void Hand::findConvexHull(const Mat& img, Mat& drawingFrame){
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    
    Mat thresholded_img = img.clone();
    findContours( thresholded_img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
    
    if(contours.size() < 1)
        return;
    
    int handInd = findLargestContour(contours);
    handContour = contours[handInd];
    handBoundingRect = boundingRect(contours[handInd]);
    
    vector<int>handHullI( contours[handInd].size() );
    vector<Vec4i> handConvDefect( contours[handInd].size() );
    
    convexHull( Mat(contours[handInd]), handPolygon, true);
    convexHull( Mat(contours[handInd]), handHullI, false);
    
    approxPolyDP( Mat(handPolygon), handPolygon,11,true);
    if (contours[handInd].size() > 3 ) {
        convexityDefects(contours[handInd], handHullI, handConvDefect);
    }
    
    
    findPalmCenter(handConvDefect, contours[handInd]);
    filterConvexes(handConvDefect, contours[handInd], handBoundingRect);
    findFingerPoints(handConvDefect, contours[handInd], handBoundingRect);
    findFingerLines();
    findHandOrientation();
    findWhichHand();
    filterFingers();
    printFingerCount(drawingFrame, (int) fingers.size());
    drawFingerLines(drawingFrame);
    drawConvexity(drawingFrame, handConvDefect, contours[handInd]);
    rectangle(drawingFrame, handBoundingRect.tl(), handBoundingRect.br(), blue);
    
    for(int i=0;i<handPolygon.size();++i) {
        line(drawingFrame,handPolygon[i], handPolygon[(i+1) % handPolygon.size()],Scalar(80,140,200), 2 );
    }
    return;
}

float Hand::pointDistance(const Point& a, const Point& b){
    int dx = a.x - b.x;
    int dy = a.y - b.y;
    return sqrt(dx*dx + dy*dy);
}

float Hand::getLineAngle(const Point& p1, const Point& p2){
    double dy = p1.y - p2.y;
    double dx = p1.x - p2.x;
    double theta = atan2(dy, dx);
    theta = theta*180/CV_PI + 180; // rads to degs
    return theta;
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

void Hand::filterConvexes(vector<Vec4i>& convDefect, const vector<Point>& contours, const Rect& boundingRect) {
    int tolerance =  boundingRect.height/8;
    vector<Vec4i>::iterator d = convDefect.begin();
    
    int maxDegree = 95;
    while( d!=convDefect.end() ) {
        Vec4i& v = (*d);
        Point p1( contours[v[0]]),
        p2( contours[v[1]]),
        p3( contours[v[2]]);
        int depth = v[3]/256;
        int angleBetween = abs(getLineAngle(p1, p3) - getLineAngle(p2, p3));
        if(depth < tolerance || angleBetween > maxDegree)
            d = convDefect.erase(d);
        else
            ++d;
    }
}

void Hand::findFingerPoints(const vector<Vec4i>& convDefect, const vector<Point>& contours, const Rect& boundingRect) {
    int handRectDiagonalLength = sqrt(boundingRect.width*boundingRect.width+boundingRect.height*boundingRect.height);
    int distanceTolerance = handRectDiagonalLength/5;
    int minFingerAffinity = handRectDiagonalLength/10;
    
    if(convDefect.size() < 2 || handPolygon.size() < 3)
        return;
    
    fingers.clear();
    
    for(vector<Point>::const_iterator pi = handPolygon.cbegin(); pi!=handPolygon.cend(); ++pi){
        const Point& p = (*pi);
        for(vector<Vec4i>::const_iterator d = convDefect.cbegin(); d!=convDefect.cend(); ++d){
            const Vec4i& v = (*d);
            Point ptStart( contours[v[0]] );
            Point ptEnd( contours[v[1]] );
            
            int distanceStart = pointDistance(ptStart, p);
            int distanceEnd = pointDistance(ptEnd, p);
            
            if((distanceStart < distanceTolerance || distanceEnd < distanceTolerance)) {
                // avoid detecting a finger multiple times
                bool isFound = false;
                for(vector<Point>::const_iterator fi = fingers.cbegin(); fi != fingers.cend() && !isFound; ++fi) {
                    isFound = isFound || (pointDistance(p,*fi) < minFingerAffinity);
                }
                if(!isFound)
                    fingers.push_back(p);
            }
        }
    }
    
}

bool Hand::palmCenterDistComparator (const Point* p1, const Point* p2) {
    return (pointDistance(palmCenter, *p1) < pointDistance(palmCenter, *p2));
}


void Hand::findPalmCenter(const vector<Vec4i>& convDefect, const vector<Point>& contours) {
    Moments mu;
    
    mu = moments( handContour, false );
    palmCenter = Point( mu.m10/mu.m00 , mu.m01/mu.m00 );
    
    if(convDefect.size() < 1 || contours.size() < 1)
        return;
    
    
    vector<const Point*> points;
    vector<Vec4i>::const_iterator i;
    for(i = convDefect.cbegin(); i != convDefect.cend(); ++i) {
        const Vec4i& v = (*i);
        points.push_back( &contours[v[0]]);
        points.push_back( &contours[v[1]]);
        points.push_back( &contours[v[2]]);
    }

    auto comparator = std::bind(&Hand::palmCenterDistComparator, this, std::placeholders::_1, std::placeholders::_2);
    std::sort(points.begin(),points.end(),comparator);
    
    int pointCount = 0;
    int mean = 0;
    for(vector<const Point*>::iterator i = points.begin(); i != points.end(); ++i) {
        int newDist = pointDistance(palmCenter, **i);
        if(mean != 0 && abs(newDist - mean/pointCount) > handBoundingRect.height/10)
            break;
        
        mean += newDist;
        ++pointCount;
    }
    
    mean /= pointCount;
    if(mean > 0) {
        circle(mFrame,palmCenter,mean,Scalar(200,255,100),8,8);
    }
}

void Hand::printFingerCount(Mat& img, int fingerCount) {
    char c[255];
    sprintf(c,"Finger #%d Hand: %s", fingerCount, (hand == LEFT) ? "LEFT" : "RIGHT");
    CommonUtils::putTextWrapper(img, c);
}

bool Hand::isHand(const vector<Point>& contours, const vector<Vec4i>& convDefect) {
    return true;
}

void Hand::drawConvexity(Mat& drawing, const vector<Vec4i>& convDefect, const vector<Point>& contours) {
    vector<Vec4i>::const_iterator d = convDefect.cbegin();
    while( d!=convDefect.cend() ) {
        const Vec4i& v=(*d);
        int startidx=v[0]; Point ptStart( contours[startidx] );
        int endidx=v[1]; Point ptEnd( contours[endidx] );
        int faridx=v[2]; Point ptFar( contours[faridx] );
        //        float depth = v[3] / 256.0;
        
        if(v[2] > 100) {
            line( drawing, ptStart, ptFar, Scalar(255,255,255), 1 );
            line( drawing, ptEnd, ptFar, Scalar(255,0,255), 1 );
            circle( drawing, ptFar, 4, Scalar(255,0,0), 4 );
        }
        d++;
    }
}

bool Hand::lineAngleCompare(const Line& l1, const Line& l2) {
    return l1.angle < l2.angle;
}

void Hand::findHandOrientation() {
    vector<Line>::iterator lItr;
    double mean = 0, std_dev = 0;
    
    if(fingerLines.size() < 2)
        return;
    
    for(lItr = fingerLines.begin(); lItr != fingerLines.end(); ++lItr)
        mean += lItr->angle;
    mean /= fingerLines.size();
    
    std::vector<double> squares ;
    double min = 360;
    vector<Line>::iterator middle;
    for( lItr = fingerLines.begin(); lItr != fingerLines.end(); ++lItr) {
        double angle = lItr->angle - mean;
        squares.push_back( std::pow( angle , 2 ) ) ;
        if(angle < min) {
            min = angle;
            middle = lItr;
        }
    }
    std_dev = std::sqrt( std::accumulate( squares.begin( ) , squares.end( ) , 0 ) / squares.size( ) ) ;
    
    int minAngle = ((int)(mean - std_dev)),
    maxAngle = ((int)(mean + std_dev));
    //
    //    cout<< "Finger Angle Mean: " << mean<< " StdDev: " << std_dev << " MinAngle: " << minAngle << " MaxAngle:" << maxAngle << endl;
    //
    //    for( lItr = fingerLines.begin(); lItr != fingerLines.end(); ++lItr) {
    //        if(lItr->angle < minAngle || lItr->angle > maxAngle)
    //            lItr = fingerLines.erase(lItr) - 1;
    //    }
    //
    //    circle(rawFrame,middle->end,10,Scalar(50,50,50),10);
}

void Hand::findFingerLines() {
    if(handBoundingRect.height < 50 || handBoundingRect.width < 50)
        return;
    
    // if palm center is outside of handpolygon
    if(pointPolygonTest(handContour, palmCenter, false) < 0.01)
        return;
    
    fingerLines.clear();
    vector<Point>::const_iterator v = fingers.cbegin();
    while(v!=fingers.cend()) {
        const Point& endPoint = (*v);
        fingerLines.push_back(Line{palmCenter, endPoint, getLineAngle(endPoint, palmCenter)});
        ++v;
    }
}

void Hand::drawFingerLines(Mat& drawing) {
    circle(drawing,palmCenter,20,red,40);
    vector<Line>::iterator d = fingerLines.begin();
    while( d!=fingerLines.end() ) {
        const Line& l = (*d++);
        
        char buff[100];
        sprintf(buff, "%.2f", l.angle);
        
        CommonUtils::putTextWrapper(drawing, buff, l.end.x+50, l.end.y+50);
        line( drawing, l.start, l.end, blue, 3 );
        circle(drawing,l.end,20,yellow,10);
    }
}