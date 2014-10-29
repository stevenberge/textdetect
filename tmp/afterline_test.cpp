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
#include "test.pch"
#include <queue>
#define DECISION_THRESHOLD_EA 0.5
// #define DECISION_THRESHOLD_SF 0.3  //999999999

using namespace std;
using namespace cv;
bool g_Test[7]={false,false,false,false,false,false,false};
bool g_Opt[7]={false,false,false,false,false,false,false};
#define MCOUT(i) if(g_Test[i-1]) cout<<"【"<<i<<"】" 
#define IF(i) if(g_Test[i-1])
#define OPT(i) if(g_Opt[i-1])
float mean(vector<Region>&regions,vector<int>group){
	int n=group.size();
	if(n<2) return 0;
	vector<float>H(n,0),W(n,0);
	float aH=0,aW=0;
	rep(i,n){
		Region &s=regions[group[i]];
		H[i]=s.bbox_.height,W[i]=s.bbox_.width;
		aH+=H[i],aW+=W[i];
	}aH/=n,aW/=n;
	float mH=0,mW=0;
	rep(i,n){
		mH+=pow(H[i]/aH-1,2),mW+=pow(W[i]/aW-1,2);
	}
	mH/=n,mW/=n;
	return mH+mW/3;
}

int edges(Region &r){
	return 0;
}

int main( int argc, char** argv )
{
	//.conf
	//mser(%d,
	if(argc==1) return 0;
	if(argc==2){
		if(strcmp(argv[1],"test")==0){
			//test
			test();
		}
		return 0;
	}

	//threshold
	float DECISION_THRESHOLD_SF=0.5;
	sscanf(argv[2],"%f",&DECISION_THRESHOLD_SF);

	//inputs.....
	//for test
	int idx=3;
	if(argc>idx)
		if(strcmp(argv[3],"rltest")==0) {
			g_RegionLineTest=true;idx++;
		}
	while(argc>idx){
		if(strcmp(argv[idx],"opt")==0) {
			idx++;
			while(argc>idx){
				if(strcmp(argv[idx],"cout")!=0)
				{
					int c;sscanf(argv[idx],"%d",&c);if(c>=1&&c<=7) g_Opt[c-1]=true;
					idx++;
				}else break;
			}
		}
		else if(strcmp(argv[idx],"cout")==0) {
			idx++;
			while(argc>idx){
				if(strcmp(argv[idx],"opt")!=0)
				{
					int c;sscanf(argv[idx],"%d",&c);if(c>=1&&c<=7) g_Test[c-1]=true;
					idx++;
				}else break;
			}
		}
		else idx++;
	}

	RegionLine rl;vector<cv::Rect> lines,allRects; Mat img, grey, lab_img, gradient_magnitude, segmentation, all_segmentations; char str[1000]; vector<Region> regions;

	ifstream fin(".conf"); fin.getline(str,1000,'\n');
	int t;double t_min,t_max; sscanf(str,"mser:%d %lf %lf",&t,&t_min,&t_max);
	// cout<<"mser:"<<t<<","<<t_min<<","<<t_max<<endl;
	::MSER mser8(false,t,t_min,t_max,1,0.7);

	// ::MSER mser8(false,5,0.00001,0.45,1,0.7);

	RegionClassifier region_boost("boost_train/trained_boost_char.xml", 0); 
	GroupClassifier  group_boost("boost_train/trained_boost_groups.xml", &region_boost); 

	img = imread(argv[1]);

	cvtColor(img, grey, CV_BGR2GRAY);
	cvtColor(img, lab_img, CV_BGR2Lab);
	gradient_magnitude = Mat_<double>(img.size());
	get_gradient_magnitude( grey, gradient_magnitude);
	segmentation = Mat::zeros(img.size(),CV_8UC3);
	all_segmentations = Mat::zeros(240,320*11,CV_8UC3);

	vector<string> content;

	for (int step =1; step<=2; step++)
	{
		///////////////////////////////////////////////////////////////////////////////////////////
		///step 1 mser
		// cout<<"step "<<step<<endl;
		int stage=1;
		{
			regions.clear();

			if (step == 2)
				grey = 255-grey;

			mser8((uchar*)grey.data, grey.cols, grey.rows, regions);
			OPT(stage){
				Mat tmp= img.clone();//Mat::zeros(cvSize(img.cols, img.rows),CV_8UC1);
				drawRegions(tmp,regions);
				sprintf(str,"%s.%d.1.1.mser.jpg",argv[1],step);
				imwrite(str,tmp);
			}

			// RegionLine::unique(regions);
			MCOUT(stage)<<"unique "<<regions.size()<<" regions"<<endl;
			for (int i=regions.size()-1; i>=0; i--)
			{
				//@使用一些方法过滤mser,显然这样过滤过于武断,可以优化
				// int w=regions.at(i).bbox_.width,h=regions.at(i).bbox_.height;
				// if ( (regions.at(i).stroke_std_/regions.at(i).stroke_mean_ > 0.8) ||
				// (regions.at(i).num_holes_>3) || (w <=2) ||(h <5) || w>2.5*h ||
				// regions.at(i).area_>0.85*w*h&&w>0.6*h)
				if(eraseRegionRule1(regions[i])) 
				{
					// erases.push_back(regions.at(i));
					regions.erase(regions.begin()+i);
				}
			}
			// RegionLine::unique(regions);
			MCOUT(stage)<<"erfill "<<regions.size()<<" regions"<<endl;
			// mser8((uchar*)blue->data, grey.cols, grey.rows, tmp);
			//@填充region的pixels_[]
			sort(regions.begin(),regions.end(),cmper);
			for (int i=0; i<regions.size(); i++)
				regions[i].er_fill(grey);
			// regions.insert(regions.begin(),tmp.begin(),tmp.end());
		}

		OPT(stage){
			Mat tmp= img.clone();//Mat::zeros(cvSize(img.cols, img.rows),CV_8UC1);
			drawMSERs(tmp,&regions,false,NULL,false);
			sprintf(str,"%s.%d.1.2.rule1_erased_mser.jpg",argv[1],step);
			imwrite(str,tmp);
		}

		vector< vector<int> > final_clusters;
		//////////////////////////////////////////////////////////////////////////////////////////
		///step 2: \brief erases and i region
		stage=2;
		{
			MCOUT(stage)<<"extracting features..."<<endl;
			for (int i=regions.size()-1; i>=0; i--)
			{
				regions[i].extract_features(lab_img, grey, gradient_magnitude);
			}

			for (int i=regions.size()-1; i>=0; i--)
			{
				if(eraseRegionRule2(regions[i])) 
				{
					regions.erase(regions.begin()+i);
				}
			}

			// MCOUT(stage)<<"inflexion erasing...."<<endl;
			OPT(stage){
				Mat tmp= img.clone();//Mat::zeros(cvSize(img.cols, img.rows),CV_8UC1);
				// drawMSERs(tmp,&regions,false,NULL,false);//,true,&img,false);
				drawRegions(tmp,regions);
				sprintf(str,"%s.%d.2.1.erase.jpg",argv[1],step);
				imwrite(str,tmp);
			}
			//choose most likely regions
			// { // priority_queue<pair<float,int> > pq; // rep(i,regions.size()){ // float s=region_boost.get_votes(&regions[i]); // pq.push(pair<int,int>(s,i)); // }

			vector<vector<int> >groups;
			vector<float>scores;
			vector<int> nxt;
			{ 
				RegionLine::classify(regions,nxt,groups);
				MCOUT(stage)<<"line regions:"<<endl;
				// rep(i,regions.size()){
				// cout<<i<<"->"<<nxt[i]<<"  ";
				// }
				rep(i,groups.size()){
					scores.push_back(group_boost(&groups[i],&regions));
					float mn=0;
					if(scores[i]>=DECISION_THRESHOLD_EA||
							groups[i].size()>=6&&mean(regions,groups[i])<0.12){//&&(mn=mean(regions,groups[i]))<0.15) {
						MCOUT(stage)<<"find a new meaningful group"<<endl;
						final_clusters.push_back(groups[i]);
					}
					// else { MCOUT(stage)<<"pass a  group with mean:"<<mn<< " cnt:"<<groups[i].size()<<endl; }
				}
			}

			OPT(stage){
				Mat tmp= img.clone();//Mat::zeros(cvSize(img.cols, img.rows),CV_8UC1);
				// drawClusters(tmp,&regions,&groups);
				// drawRegions(tmp,regions);
				drawClusters(tmp,&regions,&groups);
				rep(i,groups.size())
					drawGroupScore(tmp,regions,groups[i],scores[i]);
				drawLines(tmp,regions,nxt);
				sprintf(str,"%s.%d.2.2.lined.jpg",argv[1],step);
				imwrite(str,tmp);
			}
		}

step3:
		//////////////////////////////////////////////////////////////////////////////////////////
		//step 3: \brief cluster 11 features
		//@定义一个聚类器
		stage=3;
		MCOUT(stage)<<"clustering "<<regions.size()<<" regions"<<endl;
		if(regions.size()==0) continue;
		MaxMeaningfulClustering 	mm_clustering(METHOD_METR_SINGLE, METRIC_SEUCLIDEAN);
		vector< vector<int> > meaningful_clusters;
		//@region之间的相似度
		Mat co_occurrence_matrix = Mat::zeros((int)regions.size(), (int)regions.size(), CV_64F);
		{
			// vector<vector<int> > twoDimCluster; map<int,int> maps[2]; int sizes[2]={0,0};
			/////////////////////////////////////////////////////////////////
			for (int f=0; f<NUM_FEATURES; f++)
			{
				meaningful_clusters.clear();
				// if(f!=2&&f!=5) continue;
				int mapTag=f==2?0:f==5?1:-1;
				int dim=-1;
				t_float * data=NULL;
				extractFeatureArr(img,regions,f,dim,&data);

				//@对刚刚记录的region属性作为feature进行聚类
				//@mm_clustering N个dim维向量聚类
				if(data!=NULL) {
					mm_clustering(data, regions.size(), dim, METHOD_METR_SINGLE, METRIC_SEUCLIDEAN, &meaningful_clusters);
					free(data);
				}
				else {
					rep(i,regions.size()) meaningful_clusters.push_back(vector<int>(1,i));
				}

				//accumunate occurence matrix
				for(int k=0;k<meaningful_clusters.size();k++){
					float score=0;
					if (meaningful_clusters[k].size()>0 && ((score=group_boost(&meaningful_clusters.at(k), &regions)) >= DECISION_THRESHOLD_SF) )
					{
						final_clusters.push_back(meaningful_clusters.at(k));
						accumulate_evidence(&meaningful_clusters.at(k), 1, &co_occurrence_matrix);
					}
				}
				//draw the meaningful_clusters generated by feature f
				OPT(stage){
					Mat tmp_segmentation = Mat::zeros(img.size(),CV_8UC3);
					Mat tmp_all_segmentations = Mat::zeros(240,320*11,CV_8UC3);
					drawClusters(tmp_segmentation, &regions, &meaningful_clusters);
					Mat tmp = Mat::zeros(240,320,CV_8UC3);
					resize(tmp_segmentation,tmp,tmp.size());
					tmp.copyTo(all_segmentations(Rect(320*f,0,320,240)));
				}
				meaningful_clusters.clear();
			}
			OPT(stage){
				//now all 11 pictures represent 11 features' meaningful-clusters in all_segmentation
				sprintf(str,"%s.%d.3.meaningful_cluster.jpg",argv[1],step);
				imwrite(str,all_segmentations);
			}
		}
		//my group method

		//////////////////////////////////////////////////////////////////////////////////////////
step4:
		//step 4: co-occurence matrix clustering
		stage=4;
		{
			//@@after all features has been evaluated and co_xxx has been calculated
			double minVal;
			double maxVal;
			minMaxLoc(co_occurrence_matrix, &minVal, &maxVal);

			maxVal = NUM_FEATURES - 1; //TODO this is true only if you are using "grow == 1" in accumulate_evidence function
			minVal=0;

			co_occurrence_matrix = maxVal - co_occurrence_matrix;
			co_occurrence_matrix = co_occurrence_matrix / maxVal;

			// fast clustering from the co-occurrence matrix
			t_float *D = (t_float*)malloc((regions.size()*regions.size()) * sizeof(t_float));
			genDimVector(co_occurrence_matrix,D);
			mm_clustering(D, regions.size(), METHOD_METR_AVERAGE, &meaningful_clusters); //  TODO try with METHOD_METR_COMPLETE
			free(D);

			///@KEY
			for (int i=meaningful_clusters.size()-1; i>=0; i--)
			{
				if ( (group_boost(&meaningful_clusters.at(i), &regions) >= DECISION_THRESHOLD_EA) )
				{
					final_clusters.push_back(meaningful_clusters.at(i));
				}
			}

			OPT(stage){
				Mat tmp=img.clone();
				drawClusters(tmp,&regions,&final_clusters);
				int cnt=0;rep(i,final_clusters.size()) cnt+=final_clusters[i].size();
				MCOUT(stage)<<"found "<<cnt<<" meaningful regions"<<endl;
				// drawRegions(tmp,regions);
				sprintf(str,"%s.%d.4.all_meaningful_cluster.jpg",argv[1],step);
				imwrite(str,tmp);
			}
		}

step5:
		//////////////////////////////////////////////////////////////////////////////////////////
		//step 5: gen lines
		stage=5;
		vector<Region> tmpRegions;
		{
			//@去重
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

			MCOUT(stage)<<"found "<<tmpR.size()<<" final meaningful regions"<<endl;
			for(int i=0;i<tmpR.size();i++) 
				tmpRegions.push_back(regions[tmpR[i]]);

			OPT(stage){
				Mat tmp=img.clone();
				// drawMSERs(tmp,&tmpRegions,false,NULL,false);
				fillRegions(tmp,tmpRegions);//,false,NULL,false);
				rep(i,tmpRegions.size()){
					sprintf(str,"%d",i);
					drawText(tmp,tmpRegions[i],str,1,0.5,0.5);
				}
				// drawRegions(tmp,tmpRegions);
				sprintf(str,"%s.%d.5.1.numbered_meaningful_regions.jpg",argv[1],step);
				imwrite(str,tmp);
			}
			MCOUT(stage)<<"unique regions"<<endl;
			RegionLine::unique(tmpRegions);
			MCOUT(stage)<<"found "<<tmpRegions.size()<<" meaningful regions after erasing"<<endl;
			OPT(stage){
				Mat tmp=img.clone();
				// drawMSERs(tmp,&tmpRegions,false,NULL,false);
				fillRegions(tmp,tmpRegions);//,false,NULL,false);
				rep(i,tmpRegions.size()){
					sprintf(str,"%d",i);
					drawText(tmp,tmpRegions[i],str,2,0.5,0.5);
				}
				// drawRegions(tmp,tmpRegions);
				sprintf(str,"%s.%d.5.2.unique_meaningful_regions.jpg",argv[1],step);
				imwrite(str,tmp);
			}
		}

		/////////////////////////////////////////////////////////
		//line regions
		stage=6;
		vector<vector<int> >groups;
		{
			vector<int> nxt;
			rl.classify(tmpRegions,nxt,groups,lines);
			MCOUT(stage)<<tmpRegions.size()<<" regions are classified into "<<groups.size()<<" groups "<<endl;
			assert(groups.size()==lines.size());
			vector<float> score;
			for(int i=0;i<groups.size();i++){//groups.size()-1;i>=0;i--){
				float f=group_boost(&groups.at(i), &tmpRegions);
				score.push_back(f);
			}

			//after regions lined
			MCOUT(stage)<<"meaningful region details:"<<endl;
			IF(stage){
				rep(i,tmpRegions.size()){
					RegionLine::display(i,tmpRegions[i]);
				}
			}
			MCOUT(stage)<<"regions in line group as:"<<endl;
			rep(i,nxt.size())
				MCOUT(stage)<<i<<"->"<<nxt[i]<<" ";
			MCOUT(stage)<<endl;

			OPT(stage){
				Mat tmp=img.clone();
				drawLines(tmp,tmpRegions,nxt);
				rep(i,groups.size()){
					fillRegions(tmp,tmpRegions,groups[i]);
					drawLines(tmp,tmpRegions,groups[i],nxt);
					drawGroupScore(tmp,tmpRegions,groups[i],score[i]);
				}
				rep(i,tmpRegions.size()){
					sprintf(str,"%d",i);
					drawText(tmp,tmpRegions[i],str,1,0.5,0.5,-32,0,0,255);
				}
				sprintf(str,"%s.%d.5.4.lined_regions.jpg",argv[1],step);
				imwrite(str,tmp);
			}

			MCOUT(stage)<<"erase some region lines from "<<groups.size()<<" regions"<<endl;
			for(int i=groups.size()-1;i>=0;i--){
				int n=groups[i].size();
				if(n<=3&&score[i]<1.0/groups[i].size()||
						n>3&&n<6&&score[i]<0.01/n) {
					MCOUT(stage)<<"erase region with score:"<<score[i]<<" cnt:"<<n
						<<" mean:"<<mean(tmpRegions,groups[i])<<endl;
					groups.erase(groups.begin()+i),lines.erase(lines.begin()+i),score.erase(score.begin()+i);
				}
			}
			MCOUT(stage)<<"[after erase] "<<groups.size()<<" groups remain after erasing"<<endl;
			OPT(stage){
				Mat tmp=img.clone();
				drawClusters(tmp,&tmpRegions,&groups);
				drawRects(tmp,lines);
				rep(i,groups.size())
				{
					drawLines(tmp,tmpRegions,groups[i],nxt);
					sprintf(str,"%d:%.3f:%d",i,score[i],groups[i].size());
					// drawGroupScore(tmp,tmpRegions,groups[i],score[i]);
					drawText(tmp,tmpRegions[groups[i][0]],str,2,0.6,0.6);
				}
				rep(i,tmpRegions.size()){
					Region &s=tmpRegions.at(i);
					int w1=s.bbox_.width,h1=s.bbox_.height;
					sprintf(str,"w:%d h:%d",w1,h1);
					// drawText(tmp,s,str,1,0.3,0.3);
					drawText(tmp,s,str,1,0.5,0.5,12,255,0,0);
				}
				sprintf(str,"%s.%d.5.5.erased_lined_regions.jpg",argv[1],step);
				imwrite(str,tmp);
			}


			MCOUT(stage)<<groups.size()<<" groups' properties:"<<endl;
			rep(i,groups.size())
			{
				IF(stage){
					MCOUT(stage)<<"region score of group "<<i<<":";
					rep(j,groups[i].size()){
						MCOUT(stage)<<tmpRegions[groups[i][j]].classifier_votes_<<" ";
					}
					cout<<"properties of group "<<i<<":"<<endl;
					rep(j,groups[i].size()){
						int k=groups[i][j];
						RegionLine::display(k,tmpRegions[k]);
					}
				}
				MCOUT(stage)<<"clean group "<<i<<":"<<endl; 
				RegionLine::cleanGroup(tmpRegions,groups[i]);
			}
			rep(i,groups.size())
				lines[i]=(RegionLine::dump(tmpRegions,groups[i]));

			OPT(stage){
				Mat tmp=img.clone();
				drawClusters(tmp,&tmpRegions,&groups);
				drawRects(tmp,lines);
				rep(i,groups.size())
				{
					drawLines(tmp,tmpRegions,groups[i],nxt);
					sprintf(str,"%d:%.3f:%d",i,score[i],groups[i].size());
					// drawGroupScore(tmp,tmpRegions,groups[i],score[i]);
					drawText(tmp,tmpRegions[groups[i][0]],str,2,0.6,0.6);
				}
				// drawRegionScores(tmp,tmpRegions);
				sprintf(str,"%s.%d.5.6.cleand_lined_regions.jpg",argv[1],step);
				imwrite(str,tmp);
			}
			}

step7:
			//////////////////////////////////////////////////////////////////////////////////////////
			//final gen results
			stage=7;
			{
				RegionLine::unique(lines);
				// lines.clear();
				// rep(i,groups.size()){
					// rep(j,groups[i].size()){
						// int k=groups[i][j];
						// lines.push_back(tmpRegions[k].bbox_);
					// }
				// }

				OPT(stage){
					Mat tmp=img.clone();
					drawRects(tmp,lines,Scalar(0,0,255),3);
					sprintf(str,"%s.%d.6.final.jpg",argv[1],step);
					imwrite(str,tmp);
				}

				if(step==1)
					cout<<"<image>"<<endl<<"<imageName>"<<argv[1]<<"</imageName>"<<endl
						<<"<taggedRectangles>"<<endl;
				for(int j=0;j<lines.size();j++){
					cv::Rect s=lines[j];
					cout<<"<taggedRectangle x=\""<<s.x<<"\" y=\""<<s.y
						<<"\" width=\""<<s.width<<"\" height=\""<<s.height
						<<"\" modelType=\"1\"/>"<<endl;
				}
				if(step==2)
					cout<<"</taggedRectangles>"<<endl<<"</image>"<<endl;
			}
		}
		}
