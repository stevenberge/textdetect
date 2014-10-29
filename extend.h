#ifndef EXTEND_H
#define EXTEND_H

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>
using namespace std;
using namespace cv;

void printImg(Mat& img, bool bin=false);
void printImg(IplImage *out, bool bin=false);
void strenth(Mat &, Mat &);
void thin2(Mat& bwImg);
void mask(Mat&src, Mat&dst);


#endif
