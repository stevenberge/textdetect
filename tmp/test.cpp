#include <opencv2/opencv.hpp>


// #include <iostream>
// #include <fstream>
// #include <stdio.h>

// #include "region.h"
// #include "mser.h"
// #include "max_meaningful_clustering.h"
// #include "region_classifier.h"
// #include "group_classifier.h"
// #include "RegionLine.hpp"
// #define NUM_FEATURES 11 

// #define DECISION_THRESHOLD_EA 0.5
// #define DECISION_THRESHOLD_SF 0.999999999
using namespace std;
using namespace cv;
// #include "utils.h"
int main( int argc, char** argv )
{
	// bool record=true;
	char str[100];
	Mat img, grey, lab_img, gradient_magnitude, segmentation, all_segmentations;
	std::system("/usr/bin/tesseract hello");

	// vector<Region> regions;
	// ::MSER mser8(false,25,0.000008,0.03,1,0.7);

	// RegionClassifier region_boost("boost_train/trained_boost_char.xml", 0); 
	// GroupClassifier  group_boost("boost_train/trained_boost_groups.xml", &region_boost); 

	// img = imread("test.jpg");
	// imshow("img",img);
	// waitKey(0);

	// cvtColor(img, grey, CV_BGR2GRAY);
	// //@lab_img 是lab色
	// cvtColor(img, lab_img, CV_BGR2Lab);
	// gradient_magnitude = Mat_<double>(img.size());
	// get_gradient_magnitude( grey, gradient_magnitude);

	// segmentation = Mat::zeros(img.size(),CV_8UC3);
	return 0;
}
