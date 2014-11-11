#include <opencv/highgui.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include "region.h"
#include "mser.h"
#include "max_meaningful_clustering.h"
#include "region_classifier.h"
#include "group_classifier.h"
#include "RegionLine.hpp"
#include "utils.h"
#include <queue>

#define DECISION_THRESHOLD_EA 0.5
// #define DECISION_THRESHOLD_SF 0.3  //999999999

using namespace std;
using namespace cv;

// #define DECISION_THRESHOLD_EA 0.5
// #define DECISION_THRESHOLD_SF 0.3  //999999999
using cv::Mat;

int main( int argc, char** argv )
{
	Mat img, grey;
	char str[1000]; vector<Region> regions;
  cout<<"mser xx threshold > xx.mser.jpg"<<endl;
	if(argc<3) return 1;
	img = imread(argv[1]);
	int threshold=100; 
	sscanf(argv[2],"%d",&threshold);
	::MSER mser8(false,threshold,0.00001,0.45,1,0.7);
	cvtColor(img, grey, CV_BGR2GRAY);

	for (int step =1; step<=2; step++)
	{
		if (step == 2)
			grey = 255-grey;
		regions.clear();
		mser8((uchar*)grey.data, grey.cols, grey.rows, regions);
		for (int i=0; i<regions.size(); i++)
			regions[i].er_fill(grey);
		cout<<"region size:"<<regions.size()<<endl;
    for(int i=0; i<regions.size(); i++){
    }
		{
			Mat tmp= Mat::zeros(cvSize(img.cols, img.rows),CV_8UC1);
			// drawMSERs(tmp,&regions,true,NULL,true);
			fillRegions(tmp,regions);
			sprintf(str,"%s.%d.mser.jpg",argv[1],step);
			imwrite(str,tmp);
		}
	}
	return 1;
}
