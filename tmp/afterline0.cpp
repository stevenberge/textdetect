#define _MAIN

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
#define NUM_FEATURES 11 

#define DECISION_THRESHOLD_EA 0.5
#define DECISION_THRESHOLD_SF 0.999999999

using namespace std;
using namespace cv;

#include "utils.h"
// void genDimVector(const Mat &co_occurrence_matrix,t_float* D){
	// int pos = 0;
	// for (int i = 0; i<co_occurrence_matrix.rows; i++)
	// {
		// for (int j = i+1; j<co_occurrence_matrix.cols; j++)
		// {
			// D[pos] = (t_float)co_occurrence_matrix.at<double>(i, j);
			// pos++;
		// }
	// }
// }
int main( int argc, char** argv )
{

	Mat img, grey, lab_img, gradient_magnitude, segmentation, all_segmentations;
	char str[1000];

	vector<Region> regions;
	::MSER mser8(false,15,0.000008,0.4,1,0.7);

	RegionClassifier region_boost("boost_train/trained_boost_char.xml", 0); 
	GroupClassifier  group_boost("boost_train/trained_boost_groups.xml", &region_boost); 

	img = imread(argv[1]);

	cvtColor(img, grey, CV_BGR2GRAY);

	//@lab_img 是lab色
	cvtColor(img, lab_img, CV_BGR2Lab);
	gradient_magnitude = Mat_<double>(img.size());
	get_gradient_magnitude( grey, gradient_magnitude);

	//////////////////////////////////////////////////////////////////////////////////////////
	//////////step 1: mser regions
	segmentation = Mat::zeros(img.size(),CV_8UC3);
	all_segmentations = Mat::zeros(240,320*11,CV_8UC3);


	for (int step =1; step<=2; step++)
	{
		vector<Region> tmpRegions;

		if (step == 2)
			grey = 255-grey;

		// IplImage *blue = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);    // 制作一个单通道图像  
		// cvSplit(img, blue, green, red, NULL);   // 分割颜色通道  

		mser8((uchar*)grey.data, grey.cols, grey.rows, tmpRegions);
		//@填充region的pixels_[]
		for (int i=0; i<tmpRegions.size(); i++)
			tmpRegions[i].er_fill(grey);

		Mat tmp=Mat::zeros(img.size(),CV_8UC3);
		//drawRegions(tmp,tmpRegions);
		drawMSERs(tmp,&tmpRegions,true,&img,false);
		sprintf(str,"%s.%d.out.jpg",argv[1],step);
		imwrite(str,tmp);

		tmp=Mat::zeros(img.size(),CV_8UC3);
		drawMSERs(tmp,&tmpRegions,true,&img,true);
		sprintf(str,"%s.%d.out.jpg",argv[1],step+2);
		imwrite(str,tmp);

		{
			Mat temp=img;
			// drawMSERs(temp,&tmp);
			sprintf(str,"%s.%d.mser.jpg",argv[1],step);
			imwrite(str,temp);
		}

		regions.insert(regions.begin(),tmpRegions.begin(),tmpRegions.end());

	}



	//////////////////////////////////////////////////////////////////////////////////////////
	///step 2: \brief erases

	vector<Region> erases;
	//@filter
	for (int i=regions.size()-1; i>=0; i--)
	{
		regions[i].extract_features(lab_img, grey, gradient_magnitude);
		//@使用一些方法过滤mser,显然这样过滤过于武断,可以优化
		if ( (regions.at(i).stroke_std_/regions.at(i).stroke_mean_ > 0.8) ||
				(regions.at(i).num_holes_>3) || (regions.at(i).bbox_.width <=1) ||
				(regions.at(i).bbox_.height <=5) ||
				regions.at(i).bbox_.width>2.5*regions.at(i).bbox_.height)
		{
			// cout<<"erased:"<<i<<endl;
			// erases.push_back(regions.at(i));
			//erase严重降低recall！！
			regions.erase(regions.begin()+i);
		}
		else if( regions.at(i).bbox_.height>2.5*regions.at(i).bbox_.width)
		{
			if(regions.at(i).num_holes_>0)
				regions.erase(regions.begin()+i);
		}
	}
	/*
	// cout<<"max_area:"<<max_area<<" areas:"<<endl;
	for (int i=regions.size()-1; i>=0; i--)
	{
	// cout<<regions.at(i).area_<<" ";
	// if(regions.at(i).area_<max_area/100){
	// regions.erase(regions.begin()+i);
	// }
	}
	*/

	{
		/*
			 Mat eraseMat= Mat::zeros(cvSize(img.cols, img.rows),CV_8UC1);
			 drawRegions(eraseMat,erases);
			 imshow("erased",eraseMat);
			 waitKey(0);
		 */
	}
	
	Mat tmp=Mat::zeros(img.size(),CV_8UC3);
	drawMSERs(tmp,&regions,true,&img,false);
	sprintf(str,"%s.erase.jpg",argv[1]);
	imwrite(str,tmp);

	//////////////////////////////////////////////////////////////////////////////////////////
	//step 3: \brief cluster 11 features
	//@定义一个聚类器
	MaxMeaningfulClustering 	mm_clustering(METHOD_METR_SINGLE, METRIC_SEUCLIDEAN);
	vector< vector<int> > meaningful_clusters;
	vector< vector<int> > final_clusters;
	//@region之间的相似度
	Mat co_occurrence_matrix = Mat::zeros((int)regions.size(), (int)regions.size(), CV_64F);

	for (int f=0; f<NUM_FEATURES; f++)
	{
		int dim=-1;
		t_float * data=NULL;
		extractFeatureArr(img,regions,f,dim,&data);
		if(data==NULL) continue;

		//@对刚刚记录的region属性作为feature进行聚类
		//@mm_clustering N个dim维向量聚类
		mm_clustering(data, regions.size(), dim, METHOD_METR_SINGLE, METRIC_SEUCLIDEAN, &meaningful_clusters);
		free(data);

		for (int k=0; k<meaningful_clusters.size(); k++)
		{
			//if ( group_boost(&meaningful_clusters.at(k), &regions))
			//@对regions第k类中的各个region两两之间累积相关度
			accumulate_evidence(&meaningful_clusters.at(k), 1, &co_occurrence_matrix);

			//@如果该类region相对于regions存在明显的优势
			if ( (group_boost(&meaningful_clusters.at(k), &regions) >= DECISION_THRESHOLD_SF) )
			{
				// cout<<"group_boost..."<<endl;
				final_clusters.push_back(meaningful_clusters.at(k));
			}
		}

		//the meaningful_clusters generated by feature f
		{
			Mat tmp_segmentation = Mat::zeros(img.size(),CV_8UC3);
			Mat tmp_all_segmentations = Mat::zeros(240,320*11,CV_8UC3);
			drawClusters(tmp_segmentation, &regions, &meaningful_clusters);
			Mat tmp = Mat::zeros(240,320,CV_8UC3);
			resize(tmp_segmentation,tmp,tmp.size());
			tmp.copyTo(tmp_all_segmentations(Rect(320*f,0,320,240)));
			all_segmentations = all_segmentations + tmp_all_segmentations;
		}

		meaningful_clusters.clear();
	}
	{
		//now all 11 pictures represent 11 features' meaningful-clusters in all_segmentation
		sprintf(str,"%s.meaningfulcluster.jpg",argv[1]);
		imwrite(str,all_segmentations);
	}

	//////////////////////////////////////////////////////////////////////////////////////////
	//step 4: co-occurence matrix clustering

	//@@after all features has been evaluated and co_xxx has been calculated
	double minVal;
	double maxVal;
	minMaxLoc(co_occurrence_matrix, &minVal, &maxVal);

	maxVal = NUM_FEATURES - 1; //TODO this is true only if you are using "grow == 1" in accumulate_evidence function
	minVal=0;

	co_occurrence_matrix = maxVal - co_occurrence_matrix;
	co_occurrence_matrix = co_occurrence_matrix / maxVal;

	// fast clustering from the co-occurrence matrix
	// @every region is considered as a regions.size()-dimed-vector
	// @in this way to cluster it out
	// @this is time-consuming, since regions.size()-dimed-vector is big
	// @why donot use 11 dim feature vector to classify?
	t_float *D = (t_float*)malloc((regions.size()*regions.size()) * sizeof(t_float));
	genDimVector(co_occurrence_matrix,D);
	mm_clustering(D, regions.size(), METHOD_METR_AVERAGE, &meaningful_clusters); //  TODO try with METHOD_METR_COMPLETE
	free(D);

	///@KEY
	for (int i=meaningful_clusters.size()-1; i>=0; i--)
	{
		//if ( (! group_boost(&meaningful_clusters.at(i), &regions)) || (meaningful_clusters.at(i).size()<3) )
		if ( (group_boost(&meaningful_clusters.at(i), &regions) >= DECISION_THRESHOLD_EA) )
		{
			final_clusters.push_back(meaningful_clusters.at(i));
		}
	}

	/*
		 {
	//@draw text-line regions
	Mat m=img;
	RegionLine rl;
	rl.classify(regions);
	drawClusters(m,&regions,&rl.lines_);
	imshow("lines",m);
	waitKey(0);
	}
	*/
	//////////////////////////////////////////////////////////////////////////////////////////
	//step 5: post processing


	//@去重
	{
		vector<int> tmpR;
		for(int j=0;j<final_clusters.size();j++){
			for(int k=0;k<final_clusters[j].size();k++){
				int i=final_clusters[j][k];
				tmpR.push_back(i);
			}
		}
		sort(tmpR.begin(),tmpR.end());
		vector<int>::iterator it=std::unique(tmpR.begin(),tmpR.end());
		tmpR.erase(it,tmpR.end());


		// cout<<"drawContour:"<<tmpR.size()<<endl<<":";
		// for(int i=0;i<tmpR.size();i++) cout<<tmpR[i]<<" ";cout<<endl;
		vector<Region> tmpRegions;
		for(int i=0;i<tmpR.size();i++) tmpRegions.push_back(regions[tmpR[i]]);

		{
			Mat tmp=Mat::zeros(img.size(),CV_8UC3);
			drawMSERs(tmp,&tmpRegions,true,&img,false);
			sprintf(str,"%s.out.jpg",argv[1]);
			imwrite(str,tmp);
		}


		/////////////////////////
		//filter tmpRegions at the end
		int max_area=1;
		for (int i=tmpRegions.size()-1; i>=0; i--){
			max_area=max(max_area,tmpRegions[i].area_);
		}
		for (int i=tmpRegions.size()-1; i>=0; i--)
		{
			// cout<<tmpRegions.at(i).area_<<" ";
			if(tmpRegions.at(i).area_<max_area/100){
				tmpRegions.erase(tmpRegions.begin()+i);
			}
		}

		Mat tmp=img;
		drawRegions(tmp,tmpRegions);

		//@draw text-line regions
		RegionLine rl;
		vector<cv::Rect> lines;
		// rl.classify(tmpRegions,lines);
		rl.dump(tmpRegions,lines);

		// drawRects(tmp,lines);
		sprintf(str,"%s.lines.jpg",argv[1]);
		imwrite(str,tmp);

		cout<<"<image>"<<endl<<"<imageName>"<<argv[1]<<"</imageName>"<<endl
			<<"<taggedRectangles>"<<endl;

		// for(int j=0;j<tmpR.size();j++){
		// int i=tmpR[j];
		// Region s=regions[i];
		// cout<<"<taggedRectangle x=\""<<s.bbox_x1_<<"\" y=\""<<s.bbox_y1_
		// <<"\" width=\""<<s.bbox_x2_-s.bbox_x1_<<"\" height=\""<<s.bbox_y2_-s.bbox_y1_
		// <<"\" modelType=\"1\"/>"<<endl;
		for(int j=0;j<lines.size();j++){
			cv::Rect s=lines[j];
			cout<<"<taggedRectangle x=\""<<s.x<<"\" y=\""<<s.y
				<<"\" width=\""<<s.width<<"\" height=\""<<s.height
				<<"\" modelType=\"1\"/>"<<endl;
		}
		//if(step==2)
		cout<<"</taggedRectangles>"<<endl<<"</image>"<<endl;
	}
	//regions.clear();
	}
