#ifndef EXTEND_H
#define EXTEND_H

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>
using namespace std;
using namespace cv;

void printImg(Mat& img, bool bin=false);
void printImg(IplImage *out, bool bin=false);
void thin(const Mat &src, Mat &dst, const int iterations);
void thin2(Mat& bwImg);


void xihua(const Mat &o, Mat &tmp);
void scantable(IplImage *src);
void cvHilditchThin(cv::Mat& src, cv::Mat& dst);

void mask(Mat&src, Mat&dst, int n=10);




//void mask(Mat &src, Mat &dst);
#endif
