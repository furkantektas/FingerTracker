#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <iostream>
#include <iomanip>
#include <numeric> // accumulate
using namespace cv;
using namespace std;

typedef struct Line_s{
    Point start;
    Point end;
    double angle;
} Line;

enum Hand_e{
    LEFT,
    RIGHT
};

const Scalar red(255,0,0);
const Scalar green(0,255,0);
const Scalar blue(0,0,255);
const Scalar yellow(255,255,0);

Mat thresh_frame,rawFrame;
vector<Line> fingerLines;
vector<Point> handPolygon;
vector<Point> handContour;
vector<Point> fingers;
Point palmCenter;
Rect handBoundingRect;
enum Hand_e hand;

const Point* labeledFingers[5];

void findConvexHull(const Mat& img, Mat& drawingFrame);
bool fingerDistanceComparator(const Point& f1, const Point& f2);
void process_frame(Mat& frame);

float pointDistance(const Point& a, const Point& b);
float getLineAngle(const Point& p1, const Point& p2);
int findLargestContour(const vector<vector<Point>>& contours);
void filterConvexes(vector<Vec4i>& convDefect, const vector<Point>& contours, const Rect& boundingRect);
void filterFingers();
void findFingerPoints(const vector<Vec4i>& convDefect, const vector<Point>& contours, const Rect& boundingRect);
void findPalmCenter(const vector<Vec4i>& convDefect, const vector<Point>& contours);

void putTextWrapper(Mat& img, const char* text, int x=10, int y=30);
void printFingerCount(Mat& img, int fingerCount);
bool isHand(const vector<Point>& contours, const vector<Vec4i>& convDefect);
void drawConvexity(Mat& drawing, const vector<Vec4i>& convDefect, const vector<Point>& contours);
void findHandOrientation();
void findFingerLines();
void drawFingerLines(Mat& drawing);

int frameCount = 0;
int totalHandGravity = 0;

std::string getImageType(int number);

inline bool palmCenterDist (const Point* p1, const Point* p2) {
    return (pointDistance(palmCenter, *p1) < pointDistance(palmCenter, *p2));
}

bool isFingerOnLeft(const Point& f1) {
    return f1.x < palmCenter.x;
}

bool isFingerOnTop(const Point& f1) {
    return f1.y < palmCenter.y;
}

float fingerDistanceFromPalmCenter(const Point& f) {
    return pointDistance(f, palmCenter);
}

void findWhichHand() {
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
void refresh() {
    fingers.clear();
    fingerLines.clear();
    handPolygon.clear();
    handContour.clear();
}

void filterFingers() {
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

void getCameraInfo(VideoCapture m_cam){
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
    cout << "Input codec type: " << EXT << endl;
}

int main(int argc, char** argv) {
    VideoCapture cap(0);
    if(!cap.isOpened()) // check if we succeeded
        return -1;
    
    cap.set(CV_CAP_PROP_EXPOSURE, 0.0);
    cap.set(CV_CAP_PROP_AUTO_EXPOSURE, 0 );
    cap.set(CV_CAP_PROP_GAIN, 0.0);
    
    getCameraInfo(cap);
    //create GUI windows
    namedWindow("Thresholded Frame");
    namedWindow("FG Mask MOG");
    
    for(int keyboard=0;keyboard!=27 && cap.grab();keyboard = waitKey(0)) {
        cap >> rawFrame;
        
        // when working with video files sometimes rawFrame becomes null
        // to avoid that check if rawFrame.data is not null
        if(rawFrame.data) {
            thresh_frame = rawFrame.clone();
            process_frame(thresh_frame);
        
            findConvexHull(thresh_frame,rawFrame);
        
            imshow("Thresholded Frame", thresh_frame);
            imshow("FG Mask MOG", rawFrame);
            refresh();
        }
    }
    
    thresh_frame.release();
    rawFrame.release();
 }

void process_frame(Mat& frame) {
    cvtColor(frame, frame, COLOR_RGB2GRAY);
    threshold(frame,frame,70,255,THRESH_TOZERO);
    threshold(frame,frame,0,255,THRESH_BINARY + THRESH_OTSU);
    
    int erosion_size = 1;
    const Mat element = getStructuringElement(cv::MORPH_ELLIPSE,
                                        cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                        cv::Point(erosion_size, erosion_size) );
    dilate(frame, frame, element);
    dilate(frame, frame, element);
}

void findConvexHull(const Mat& img, Mat& drawingFrame){
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
    
   
    filterConvexes(handConvDefect, contours[handInd], handBoundingRect);
    findPalmCenter(handConvDefect, contours[handInd]);
    findFingerPoints(handConvDefect, contours[handInd], handBoundingRect);
    findFingerLines();
    findHandOrientation();
    findWhichHand();
    filterFingers();
    printFingerCount(drawingFrame, (int) fingers.size());
    drawFingerLines(drawingFrame);
    drawConvexity(drawingFrame, handConvDefect, contours[handInd]);
    rectangle(drawingFrame, handBoundingRect.tl(), handBoundingRect.br(), blue);
    return;
}

float pointDistance(const Point& a, const Point& b){
    int dx = a.x - b.x;
    int dy = a.y - b.y;
    return sqrt(dx*dx + dy*dy);
}

float getLineAngle(const Point& p1, const Point& p2){
    double dy = p1.y - p2.y;
    double dx = p1.x - p2.x;
    double theta = atan2(dy, dx);
    theta = theta*180/CV_PI + 180; // rads to degs
    return theta;
}

int findLargestContour(const vector<vector<Point>>& contours) {
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

void filterConvexes(vector<Vec4i>& convDefect, const vector<Point>& contours, const Rect& boundingRect) {
    int tolerance =  boundingRect.height/8;
    vector<Vec4i>::iterator d = convDefect.begin();

    while( d!=convDefect.end() ) {
        Vec4i& v = (*d);
        Point p1( contours[v[0]]),
              p2( contours[v[1]]),
              p3( contours[v[2]]);
        int depth = v[3]/256;
        
//        if(depth < tolerance)
//            d = convDefect.erase(d);
//        else
            ++d;
    }
}

void findFingerPoints(const vector<Vec4i>& convDefect, const vector<Point>& contours, const Rect& boundingRect) {
    int fingerTolerance = 20;
    
    if(convDefect.size() < 2)
        return;
    
    fingers.clear();
    vector<Point>::const_iterator pi = handPolygon.cbegin();
    while(pi!=handPolygon.cend()) {
        const Point& p = (*pi);
        vector<Vec4i>::const_iterator d = convDefect.cbegin();
        while( d!=convDefect.cend()) {
            const Vec4i& v = (*d);
            Point ptStart( contours[v[0]] );
            Point ptEnd( contours[v[1]] );
        
            int distanceStart = pointDistance(ptStart, p);
            int distanceEnd = pointDistance(ptEnd, p);
            if((distanceStart < fingerTolerance || distanceEnd < fingerTolerance) &&
               (pointPolygonTest(handContour, p, false) < 0.01)) {
                if(std::find(fingers.cbegin(), fingers.cend(), p) == fingers.cend())
                    fingers.push_back(p);
            }
            ++d;
        }
        ++pi;
    }

}

void findPalmCenter(const vector<Vec4i>& convDefect, const vector<Point>& contours) {
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

    std::sort(points.begin(),points.end(),palmCenterDist);
    
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
        circle(rawFrame,palmCenter,mean,Scalar(200,255,100),8,8);
    }
}

void putTextWrapper(Mat& img, const char* text, int x, int y) {
    static const int fontFace = FONT_HERSHEY_SIMPLEX;
    static const double fontScale = 0.5;
    static const int thickness = 1;
    cv::Point textOrg(x,y);
    cv::putText(img, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness,8);
}

void printFingerCount(Mat& img, int fingerCount) {
    char c[255];
    sprintf(c,"Finger #%d Hand: %s", fingerCount, (hand == LEFT) ? "LEFT" : "RIGHT");
    putTextWrapper(img, c);
}

bool isHand(const vector<Point>& contours, const vector<Vec4i>& convDefect) {
    return true;
}

void drawConvexity(Mat& drawing, const vector<Vec4i>& convDefect, const vector<Point>& contours) {
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

bool lineAngleCompare(const Line& l1, const Line& l2) {
    return l1.angle < l2.angle;
}

void findHandOrientation() {
    vector<Line>::iterator lItr;
    double mean = 0, std_dev = 0;
    
    if(fingerLines.size() < 2)
        return;
    
    for(lItr = fingerLines.begin(); lItr != fingerLines.end(); ++lItr)
        mean += lItr->angle;
    mean /= fingerLines.size();
    
    std::vector<double> squares ;
    for( lItr = fingerLines.begin(); lItr != fingerLines.end(); ++lItr)
        squares.push_back( std::pow( lItr->angle - mean , 2 ) ) ;
    std_dev = std::sqrt( std::accumulate( squares.begin( ) , squares.end( ) , 0 ) / squares.size( ) ) ;
    
    int minAngle = ((int)(mean - std_dev)) % 360,
        maxAngle = ((int)(mean + std_dev)) % 360;

    cout<< "Finger Angle Mean: " << mean<< " StdDev: " << std_dev << " MinAngle: " << minAngle << " MaxAngle:" << maxAngle << endl;
}

void findFingerLines() {
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

void drawFingerLines(Mat& drawing) {
    circle(rawFrame,palmCenter,20,red,40);
    vector<Line>::iterator d = fingerLines.begin();
    while( d!=fingerLines.end() ) {
        const Line& l = (*d++);
        
        char buff[100];
        sprintf(buff, "%.2f", l.angle);
        
        putTextWrapper(drawing, buff, l.end.x+50, l.end.y+50);
        line( drawing, l.start, l.end, blue, 3 );
        circle(drawing,l.end,20,yellow,10);
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