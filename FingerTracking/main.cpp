#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace std;

typedef struct Line_s{
    Point start;
    Point end;
} Line;

const Scalar red(255,0,0);
const Scalar green(0,255,0);
const Scalar blue(0,0,255);
const Scalar yellow(255,255,0);

Mat thresh_frame,rawFrame, fingerFrame;
vector<Line> fingerLines;
vector<Point> handPolygon;
vector<Point> fingers;

Mat findConvexHull(Mat& img);

void process_frame(Mat& frame);
void drawConvexity(Mat& drawing, vector<Vec4i>& convDefect, vector<Point>& contours);
void drawFingerLines(Mat& drawing);
void findFingerPoints(vector<Vec4i>& convDefect, vector<Point>& contours);
std::string getImageType(int number);

float pointDistance(Point& a, Point& b);
int findLargestContour(vector<vector<Point>>& contours);
float getAngle(Point& s, Point& f, Point& e);
void filterConvexes(vector<Vec4i>& convDefect, vector<Point>& contours, Rect& boundingRect);
    
bool isHand(vector<Point>& contours, vector<Vec4i>& convDefect);
void putTextWrapper(Mat& img, char* text);
void printFingerCount(Mat& img, int fingerCount);

int main(int argc, char** argv) {
    VideoCapture cap("/Users/ft/Development/FingerTracking/FingerTracking/hand.m4v");
    if(!cap.isOpened()) // check if we succeeded
        return -1;
    
    //create GUI windows
    namedWindow("Raw Frame");
    namedWindow("Thresholded Frame");
    namedWindow("FG Mask MOG");

    for(int keyboard=0;keyboard!=27 && cap.grab();keyboard = waitKey(300)) {
        cap >> rawFrame;
        
        thresh_frame = rawFrame.clone();
        process_frame(thresh_frame);
        
        fingerFrame = thresh_frame.clone();
        fingerFrame = findConvexHull(fingerFrame);
        
        
        imshow("Raw Frame", rawFrame);
        imshow("Thresholded Frame", thresh_frame);
        imshow("FG Mask MOG", fingerFrame);
    }
    
    thresh_frame.release();
    rawFrame.release();
    fingerFrame.release();
 }

void process_frame(Mat& frame) {
    cvtColor(frame, frame, COLOR_RGB2GRAY);
    threshold(frame,frame,0,255,THRESH_BINARY + THRESH_OTSU);
    
    int erosion_size = 1;
    const Mat element = getStructuringElement(cv::MORPH_ELLIPSE,
                                        cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                        cv::Point(erosion_size, erosion_size) );
    dilate(frame, frame, element);
    dilate(frame, frame, element);
}

Mat findConvexHull(Mat& img){
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    findContours( img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
    
    int handInd = findLargestContour(contours);


    vector<int>handHullI( contours[handInd].size() );
    vector<Vec4i> handConvDefect( contours[handInd].size() );

    convexHull( Mat(contours[handInd]), handPolygon, true);
    convexHull( Mat(contours[handInd]), handHullI, false);
    approxPolyDP( Mat(handPolygon), handPolygon,11,true);
    if (contours[handInd].size() > 3 ) {
        convexityDefects(contours[handInd], handHullI, handConvDefect);
    }

    Rect handRect = boundingRect(contours[handInd]);
    filterConvexes(handConvDefect, contours[handInd], handRect);
    findFingerPoints(handConvDefect, contours[handInd]);
    printFingerCount(rawFrame, (int) fingers.size());
//    drawFingerLines(rawFrame);
//    drawConvexity(rawFrame, handConvDefect, contours[handInd]);
    rectangle(rawFrame, handRect.tl(), handRect.br(), blue);
    for(int i=0; i<fingers.size();++i)
        circle(rawFrame,fingers[i],2,yellow,2);
    
    return rawFrame;
}

float pointDistance(Point& a, Point& b){
    int dx = a.x - b.x;
    int dy = a.y - b.y;
    return sqrt(dx*dx + dy*dy);
}

float getAngle(Point& p1, Point& p2, Point& p3){
    return abs(p1.y-p2.y)/sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)) * 180 / CV_PI;
}

int findLargestContour(vector<vector<Point>>& contours) {
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

void filterConvexes(vector<Vec4i>& convDefect, vector<Point>& contours, Rect& boundingRect) {
    int tolerance =  boundingRect.height/8;
    vector<Vec4i>::iterator d = convDefect.begin();

    while( d!=convDefect.end() ) {
        Vec4i& v = (*d);
        int depth = v[3]/256;
        
        if(depth < tolerance)
            d = convDefect.erase(d);
        else
            ++d;
    }
}

void findFingerPoints(vector<Vec4i>& convDefect, vector<Point>& contours) {
    int fingerTolerance = 50;
    
    if(convDefect.size() < 2)
        return;
    
    fingers.clear();
    fingerLines.clear();
    vector<Point>::iterator pi = handPolygon.begin();
    while(pi!=handPolygon.end()) {
        Point p = (*pi);
        vector<Vec4i>::iterator d = convDefect.begin();
        while( d!=convDefect.end()) {
            Vec4i& v = (*d);
            Point ptStart( contours[v[0]] );
            Point ptEnd( contours[v[1]] );
        
            int distanceStart = pointDistance(ptStart, p);
            int distanceEnd = pointDistance(ptEnd, p);
            if(distanceStart < fingerTolerance || distanceEnd < fingerTolerance) {
                fingers.push_back(p);
            }
            ++d;
        }
        ++pi;
    }

}

void putTextWrapper(Mat& img, char* text) {
    static const int fontFace = FONT_HERSHEY_SIMPLEX;
    static const double fontScale = 0.5;
    static const int thickness = 1;
    static const cv::Point textOrg(10, 30);
    cv::putText(img, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness,8);
}

void printFingerCount(Mat& img, int fingerCount) {
    char c[255];
    sprintf(c,"Finger #%d", fingerCount);
    putTextWrapper(img, c);
}

bool isHand(vector<Point>& contours, vector<Vec4i>& convDefect) {
    return true;
    bool isHand = false;
    int concavity = 0;
    vector<Vec4i>::iterator d = convDefect.begin();
    while( d!=convDefect.end() ) {
        Vec4i& v=(*d);
        if(v[2]>100)
            ++concavity;
//        double pos = pointPolygonTest(contours, v, false);
//        if(pos)
        d++;
    }
    return true;
//    return (concavity > 2);
}

void drawConvexity(Mat& drawing, vector<Vec4i>& convDefect, vector<Point>& contours) {
    vector<Vec4i>::iterator d = convDefect.begin();
    while( d!=convDefect.end() ) {
        Vec4i& v=(*d);
        int startidx=v[0]; Point ptStart( contours[startidx] );
        int endidx=v[1]; Point ptEnd( contours[endidx] );
        int faridx=v[2]; Point ptFar( contours[faridx] );
        float depth = v[3] / 256.0;
        
        if(v[2] > 100) {
            line( drawing, ptStart, ptFar, Scalar(255,255,255), 1 );
            line( drawing, ptEnd, ptFar, Scalar(255,0,255), 1 );
            circle( drawing, ptFar, 4, Scalar(255,0,0), 4 );
        }
        d++;
    }
}

void drawFingerLines(Mat& drawing) {
    vector<Line>::iterator d = fingerLines.begin();
    while( d!=fingerLines.end() ) {
        Line& l = (*d++);
        line( drawing, l.start, l.end, blue, 3 );
    }
}


std::string getImageType(int number) {
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