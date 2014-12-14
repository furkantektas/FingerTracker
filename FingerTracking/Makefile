Cxx=g++
CXXFLAGS = -I/usr/local/include/opencv2 -O3 -g0 -Wall

LIBS =    -L/usr/lib -ldl -std=gnu++0x -std=c++0x -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video -lopencv_objdetect -lopencv_calib3d
 
# put the soruce files here example here face.cpp and function.cpp are commom source files used by both targets
COMMON_SOURCES = face.cpp functions.cpp
EXEC_NAME = fingerTracking

all: main.o hand.o common_utils.o calibration.o
	$(Cxx) -o $(EXEC_NAME) main.o common_utils.o hand.o calibration.o $(CXXFLAGS) $(LIBS)

main.o : main.cpp 
	$(Cxx) -c  main.cpp $(CXXFLAGS) -I $(LIBS) 

common_utils.o : common_utils.cpp 
	$(Cxx) -c  common_utils.cpp $(CXXFLAGS) -I $(LIBS)

hand.o : hand.cpp 
	$(Cxx) -c  hand.cpp $(CXXFLAGS) -I $(LIBS) 

calibration.o : calibration.cpp
	$(Cxx) -c  calibration.cpp $(CXXFLAGS) -I $(LIBS) 	

clean:
	rm *.o $(EXEC_NAME)