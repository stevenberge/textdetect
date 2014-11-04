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

#define NUM_FEATURES 11 

#define DECISION_THRESHOLD_EA 0.5
#define DECISION_THRESHOLD_SF 0.999999999

using namespace std;
using namespace cv;

#include "utils.h"
#include "extend.h"
inline float dist(Region &a, Region &b){
  float f1 = (float)a.stroke_mean_/b.stroke_mean_;
  if( f1 > 2 || f1 < 0.5 ) return 0;

  f1 = (float)a.stroke_mean_*a.bbox_.height/(b.stroke_mean_*b.bbox_.height);
  if( f1 > 4 || f1 < 0.25 ) return 0;
  f1=f1>1?f1:1/f1;

  cv::Rect s = a.bbox_, t = b.bbox_;
  float sw = s.width, tw = t.width, sh = s.height, th = t.height;
  float sx = s.x, tx = t.x, sy = s.y, ty = t.y;

  float dx = ( tx + tw/2 )-( sx + sw/2 ), dy = (ty + th/2 )-( sy + sh/2 );
  float d = pow( dx*dx + dy*dy, 0.5)/(sh + th);

  if(d>5) return 0;
  // cout<<f1<<":"<<d<<endl;
  return  100/(d*f1+1);
}


int main( int argc, char** argv )
{

  Mat img, grey, lab_img, gradient_magnitude, segmentation, all_segmentations;

  vector<Region> regions;
  //::MSER mser8(true,20,0.00008,0.03,1,0.7);
  ::MSER mser8(true,15,0.00008,0.05,1,0.7);

  RegionClassifier region_boost("boost_train/trained_boost_char.xml", 0); 
  GroupClassifier  group_boost("boost_train/trained_boost_groups.xml", &region_boost); 

  img = imread(argv[1]);
  cvtColor(img, grey, CV_BGR2GRAY);
  cvtColor(img, lab_img, CV_BGR2Lab);
  gradient_magnitude = Mat_<double>(img.size());
  get_gradient_magnitude( grey, gradient_magnitude);

  segmentation = Mat::zeros(img.size(),CV_8UC3);
  all_segmentations = Mat::zeros(240,320*11,CV_8UC3);

  vector<Region> final_regions;
  int mid = -1;
  for (int step =1; step<3; step++)
  {

    if (step == 2)
      grey = 255-grey;

    //double t_tot = (double)cvGetTickCount();

    //double t = (double)cvGetTickCount();
    cout<<"mser"<<endl;
    mser8((uchar*)grey.data, grey.cols, grey.rows, regions);
    cout<<regions.size()<<" regions extracted"<<endl;

    //t = cvGetTickCount() - t;
    //cout << "Detected " << regions.size() << " regions" << " in " << t/((double)cvGetTickFrequency()*1000.) << " ms." << endl;
    //t = (double)cvGetTickCount();

    for (int i=0; i<regions.size(); i++)
      regions[i].er_fill(grey);
    {
      Mat tmp = Mat::zeros(img.size(), CV_8UC3);
      drawMSERs(tmp, &regions, true, &img, true);
      char buf[100]; sprintf(buf, "out1/%s.%d.mser0.jpg", argv[1], step);
      imwrite(buf, tmp);
    }
    //t = cvGetTickCount() - t;
    //cout << "Regions filled in " << t/((double)cvGetTickFrequency()*1000.) << " ms." << endl;
    //t = (double)cvGetTickCount();

    double max_stroke = 0;
    for (int i=regions.size()-1; i>=0; i--)
    {
      regions[i].extract_features(lab_img, grey, gradient_magnitude);  //|| (regions.at(i).bbox_.width <=1)
      if ( (regions.at(i).stroke_std_/regions.at(i).stroke_mean_ > 0.8) || (regions.at(i).num_holes_>2)  || (regions.at(i).bbox_.height <=2)
           || (regions.at(i).bbox_.width > 3.5*regions.at(i).bbox_.height))
        regions.erase(regions.begin()+i);
      else 
        max_stroke = max(max_stroke, regions[i].stroke_mean_);
    }

 /////////
    cout<<"calculate graph"<<endl;
    int N = regions.size();
    vector<vector<float> > graph(N, vector<float>(N, 0));
    {
      for (int i=0; i<N; i++){
        graph[i][i]=100;
        for(int j=i+1; j<N; j++){
          graph[i][j]=graph[j][i]=dist(regions[i], regions[j]);
        }
      }
    }
    {
      Mat tmp = Mat::zeros(img.size(), CV_8UC3);
      drawMSERs(tmp, &regions, true, &img, true);
      char buf[100]; sprintf(buf, "out1/%s.%d.mser.jpg", argv[1], step);
      imwrite(buf, tmp);
    }
    cout<<"graph calculated"<<endl;

    //t = cvGetTickCount() - t;
    //cout << "Features extracted in " << t/((double)cvGetTickFrequency()*1000.) << " ms." << endl;
    //t = (double)cvGetTickCount();


    MaxMeaningfulClustering 	mm_clustering(METHOD_METR_SINGLE, METRIC_SEUCLIDEAN);

    vector< vector<int> > meaningful_clusters;
    vector< vector<int> > final_clusters;
    Mat co_occurrence_matrix = Mat::zeros((int)regions.size(), (int)regions.size(), CV_64F);

    int dims[NUM_FEATURES] = {3,3,3,3,3,3,3,3,3,5,5};

    for (int f=0; f<NUM_FEATURES; f++)
    {
      unsigned int N = regions.size();
      if (N<3) break;
      int dim = dims[f];
      t_float *data = (t_float*)malloc(dim*N * sizeof(t_float));
      int count = 0;

      for (int i=0; i<regions.size(); i++)
      {
        data[count] = (t_float)(regions.at(i).bbox_.x+regions.at(i).bbox_.width/2)/img.cols;
        data[count+1] = (t_float)(regions.at(i).bbox_.y+regions.at(i).bbox_.height/2)/img.rows;
        switch (f)
        {
          case 0:
            data[count+2] = (t_float)regions.at(i).intensity_mean_/255;
            break;	
          case 1:
            data[count+2] = (t_float)regions.at(i).boundary_intensity_mean_/255;	
            break;	
          case 2:
            data[count+2] = (t_float)regions.at(i).bbox_.y/img.rows;	
            break;	
          case 3:
            data[count+2] = (t_float)(regions.at(i).bbox_.y+regions.at(i).bbox_.height)/img.rows;	
            break;	
          case 4:
            data[count+2] = (t_float)max(regions.at(i).bbox_.height, regions.at(i).bbox_.width)/max(img.rows,img.cols);	
            break;	
          case 5:
            data[count+2] = (t_float)regions.at(i).stroke_mean_/max_stroke;	
            break;	
          case 6:
            data[count+2] = (t_float)regions.at(i).area_/(img.rows*img.cols);	
            break;	
          case 7:
            data[count+2] = (t_float)(regions.at(i).bbox_.height*regions.at(i).bbox_.width)/(img.rows*img.cols);	
            break;	
          case 8:
            data[count+2] = (t_float)regions.at(i).gradient_mean_/255;	
            break;	
          case 9:
            data[count+2] = (t_float)regions.at(i).color_mean_.at(0)/255;	
            data[count+3] = (t_float)regions.at(i).color_mean_.at(1)/255;	
            data[count+4] = (t_float)regions.at(i).color_mean_.at(2)/255;	
            break;	
          case 10:
            data[count+2] = (t_float)regions.at(i).boundary_color_mean_.at(0)/255;	
            data[count+3] = (t_float)regions.at(i).boundary_color_mean_.at(1)/255;	
            data[count+4] = (t_float)regions.at(i).boundary_color_mean_.at(2)/255;	
            break;	
        }
        count = count+dim;
      }

      mm_clustering(data, N, dim, METHOD_METR_SINGLE, METRIC_SEUCLIDEAN, &meaningful_clusters); // TODO try accumulating more evidence by using different methods and metrics

      for (int k=0; k<meaningful_clusters.size(); k++)
      {
        //if ( group_boost(&meaningful_clusters.at(k), &regions)) // TODO try is it's betetr to accumulate only the most probable text groups
        accumulate_evidence(&meaningful_clusters.at(k), 1, &co_occurrence_matrix);

        if ( (group_boost(&meaningful_clusters.at(k), &regions) >= DECISION_THRESHOLD_SF) )
        {
          final_clusters.push_back(meaningful_clusters.at(k));
        }
      }

      Mat tmp_segmentation = Mat::zeros(img.size(),CV_8UC3);
      Mat tmp_all_segmentations = Mat::zeros(240,320*11,CV_8UC3);
      drawClusters(tmp_segmentation, &regions, &meaningful_clusters);
      Mat tmp = Mat::zeros(240,320,CV_8UC3);
      resize(tmp_segmentation,tmp,tmp.size());
      Mat _t=tmp_all_segmentations(Rect(320*f,0,320,240));
      tmp.copyTo(_t);
      all_segmentations = all_segmentations + tmp_all_segmentations;

      free(data);
      meaningful_clusters.clear();
    }



    {
        vector<vector<int> > srt;
        int dim=1;
        t_float *data = (t_float*)malloc(dim*N * sizeof(t_float));
        for (int i=0, count=0; i<regions.size(); i++, count+=dim)
        {
            Region &r = regions.at(i);
            //data[count+0] = (t_float)r.bbox_.height*r.stroke_mean_;
            //data[count+1] = (t_float)regions.at(i).stroke_mean_;
            //data[count+2] = (t_float)regions.at(i).stroke_std_;
            //data[count+1] = (t_float)regions.at(i).bbox_.x/pow(img.cols, 0.5);
            data[count] = (t_float)regions.at(i).bbox_.y/pow(img.rows, 0.5);
        }
        mm_clustering(data, N, dim, METHOD_METR_SINGLE, METRIC_SEUCLIDEAN, &meaningful_clusters); // TODO try accumulating more evidence by using different methods and metrics
        for(int i=0; i<meaningful_clusters.size(); i++){
            vector<float> scores;
            vector<int> &s=meaningful_clusters.at(i);
            for(int j=0; j<s.size(); j++){
                scores.push_back(regions.at(s[j]).classifier_votes_);
            }
            int j=random()%1000;
            cout<<"group "<<j<<":"<<endl;
            float f = groupScore(graph, scores, s);
            if(f>400){
                srt.push_back(s);
            }
//            Mat tmp = Mat::zeros(img.size(), CV_8UC3);
//            fillRegions(tmp, regions, s);
//            char buf[100]; sprintf(buf, "out1/%d.group.jpg",  j);
//            imwrite(buf, tmp);
        }

        set<int> rs;
        set<int> ss;
        vector<int> crt;
        for(int i=0; i<final_clusters.size(); i++){
          for(int j=0; j<final_clusters[i].size(); j++){
            rs.insert(final_clusters[i][j]);
          }
        }
        for(int i=0; i<srt.size(); i++){
          for(int j=0; j<srt[i].size(); j++){
            ss.insert(srt[i][j]);
          }
        }
        // cout<<"after insert"<<endl;
        for(set<int>::iterator it=ss.begin(); it!=ss.end(); it++){
            int t=*it;
            if(rs.find(t)!=rs.end()) continue;
            crt.push_back(*it);
        }
        Mat tmp = Mat::zeros(img.size(), CV_8UC3);
        fillRegions(tmp, regions, crt);
        char buf[100]; sprintf(buf, "out1/%s.%d.correct0.jpg", argv[1], step);
        imwrite(buf, tmp);
        final_clusters.push_back(crt);
    }

    //t = cvGetTickCount() - t;
    //cout << "Clusterings (" << NUM_FEATURES << ") done in " << t/((double)cvGetTickFrequency()*1000.) << " ms." << endl;
    //t = (double)cvGetTickCount();

    /**/
    double minVal;
    double maxVal;
    minMaxLoc(co_occurrence_matrix, &minVal, &maxVal);

    maxVal = NUM_FEATURES - 1; //TODO this is true only if you are using "grow == 1" in accumulate_evidence function
    minVal=0;

    co_occurrence_matrix = maxVal - co_occurrence_matrix;
    co_occurrence_matrix = co_occurrence_matrix / maxVal;

    //we want a sparse matrix

    t_float *D = (t_float*)malloc((regions.size()*regions.size()) * sizeof(t_float)); 
    int pos = 0;
    for (int i = 0; i<co_occurrence_matrix.rows; i++)
    {
      for (int j = i+1; j<co_occurrence_matrix.cols; j++)
      {
        D[pos] = (t_float)co_occurrence_matrix.at<double>(i, j);
        pos++;
      }
    }

    // fast clustering from the co-occurrence matrix
    mm_clustering(D, regions.size(), METHOD_METR_AVERAGE, &meaningful_clusters); //  TODO try with METHOD_METR_COMPLETE
    free(D);

    //t = cvGetTickCount() - t;
    //cout << "Evidence Accumulation Clustering done in " << t/((double)cvGetTickFrequency()*1000.) << " ms. Got " << meaningful_clusters.size() << " clusters." << endl;
    //t = (double)cvGetTickCount();


    for (int i=meaningful_clusters.size()-1; i>=0; i--)
    {
      //if ( (! group_boost(&meaningful_clusters.at(i), &regions)) || (meaningful_clusters.at(i).size()<3) )
      if ( (group_boost(&meaningful_clusters.at(i), &regions) >= DECISION_THRESHOLD_EA) )
      {
        final_clusters.push_back(meaningful_clusters.at(i));
      }
    }
//////////
    {
      set<int> rs;
      for(int i=0; i<final_clusters.size(); i++){
        for(int j=0; j<final_clusters[i].size(); j++){
          rs.insert(final_clusters[i][j]);
        }
      }
      // cout<<"after insert"<<endl;
      vector<Region> bs;
      vector<int> crt;
      for(int j=0; j<regions.size(); j++){
        if(rs.end()==rs.find(j)){
          float score=0;
          for(set<int>::iterator it=rs.begin(); it!=rs.end(); it++){
            //bs.push_back(regions[rs[i]]);
            {
              if(graph[*it][j] > 40)
                score+=graph[*it][j];
            }
          }
          // cout<<score<<endl;
          if(score>100) bs.push_back(regions[j]), crt.push_back(j);
        }
      }
      ////
      final_clusters.push_back(crt);

      Mat tmp = Mat::zeros(img.size(), CV_8UC3);
      drawMSERs(tmp, &bs, true, &img, true);
      char buf[100]; sprintf(buf, "out1/%s.%d.correct.jpg", argv[1], step);
      imwrite(buf, tmp);
    }


    drawClusters(segmentation, &regions, &final_clusters);
    {
        set<int> res;
        for(int i=0; i<final_clusters.size(); i++){
            for(int j=0; j<final_clusters[i].size(); j++){
                res.insert(final_clusters[i][j]);
            }
        }
        for(set<int>::iterator it=res.begin(); it!=res.end(); it++){
            int i=*it;
            final_regions.push_back(regions[i]);
        }
        if( step == 1) mid=final_regions.size()-1;
    }

    if (step == 2)
    {
      vector<int> res;
      unique(final_regions, mid, res);
      Mat tmp = Mat::zeros(img.size(), CV_8UC3);
      fillRegions(tmp, final_regions, res);
      cvtColor(tmp, tmp, CV_BGR2GRAY);
      threshold(tmp,tmp,1,255,CV_THRESH_BINARY);
      char buf[100];
      sprintf(buf, "out1/%s.out.png", argv[1]);
      imwrite(buf, tmp);

    }


    regions.clear();
    //t_tot = cvGetTickCount() - t_tot;
    //cout << " Total processing for one frame " << t_tot/((double)cvGetTickFrequency()*1000.) << " ms." << endl;

  }

}

