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
#define cout if(debug)cout

using namespace std;
using namespace cv;

#include "utils.h"
#include "extend.h"
inline bool inSameLine(Region &a, Region &b){
    cv::Rect s = a.bbox_, t = b.bbox_;
    float sh = s.height, th = t.height;
    float sx = s.x, tx = t.x, sy = s.y, ty = t.y;
    return ty + th >= sy + sh/3 && ty <= sy + sh*2/3
            || sy + sh >= ty + th/3 && sy <= ty + th*2/3;
}
inline float dist(Region &a, Region &b){
    float f1 = (float)a.stroke_mean_/b.stroke_mean_;
    if( f1 > 2 || f1 < 0.5 ) return 0;

    f1 = a.bbox_.height/(float)b.bbox_.height;
    if( f1 > 2 || f1 < 0.5 ) return 0;

    f1 = (float)a.stroke_mean_*a.bbox_.height/(b.stroke_mean_*b.bbox_.height);
    if( f1 > 4 || f1 < 0.25 ) return 0;
    f1=f1>1?f1:1/f1;

    float f2=a.gradient_mean_/b.gradient_mean_;
    f2=f2>1?f2:1/f2;

    cv::Rect s = a.bbox_, t = b.bbox_;
    float sw = s.width, tw = t.width, sh = s.height, th = t.height;
    float sx = s.x, tx = t.x, sy = s.y, ty = t.y;

    float dx = ( tx + tw/2 )-( sx + sw/2 ), dy = (ty + th/2 )-( sy + sh/2 );
    float d = pow( dx*dx + dy*dy, 0.5)/(sh + th)+1; //1~3

    if(d>5) return 0;
    // cout<<f1<<":"<<d<<endl;
    return  100/(d*f1*f2+1);
}
void dfs0(const vector<vector<bool> > &sameline, const set<int> &block, vector<int> &line, int v, vector<bool> & vis){
    int N=vis.size();
    line.push_back(v);
    vis[v]=true;
    for(int i=0; i<N; i++){
        if(!vis[i] && block.count(i) && sameline[v][i] ){
            dfs0(sameline, block, line, i, vis);
        }
    }
}
void dfs(const vector<vector<float> > &graph, const float thres, vector<int> &tmp, int v, vector<bool> & vis){
    int N=vis.size();
    tmp.push_back(v);
    vis[v]=true;
    for(int i=0; i<N; i++){
        if(!vis[i] && graph[v][i]>thres){
            dfs(graph, thres, tmp, i, vis);
        }
    }
}
void cutToLines(const vector<vector<bool> > &sameline, const vector<int> &block, vector<vector<int> > &lines){
    int N = block.size(), M=sameline.size();
    set<int> tt;
    for(int i=0; i<N; i++){
        tt.insert(block[i]);
    }
    vector<bool> vis(M, false);
    for(int i=0; i<block.size(); i++){
        if(!vis[block[i]]){
            vector<int> line;
            dfs0(sameline, tt, line, block[i], vis);
            lines.push_back(line);
        }
    }
}
struct cmp{
    bool operator ()(const Rect &a, const Rect &b){
        return a.x < b.x;
    }
}cmper;
void getLineRect(const vector<Region> &regions, const vector<int> &line, vector<Rect> &res){
    int n = line.size();
    if(n==0) return ;

    vector<Rect> rects;
    float width = 0;
    for(int i=0; i<n; i++){
        const Rect &r = regions[line[i]].bbox_;
        rects.push_back(r);
        width += r.width;
    }
    width /= n;

    sort(rects.begin(), rects.end(), cmper);
    int x1=-1, x2=-1, y1=-1, y2=-1;
    //int x = 0, y = 0;
    for(int i=0; i<rects.size(); i++){
        Rect &r=rects[i];
        if( x2<0 || r.x-x2 < width/2){
//        const Region &r=regions[line[i]];
            x1 = x1 >= 0 ? min(x1, r.x) : r.x;
            y1 = y1 >= 0 ? min(y1, r.y) : r.y;
            x2 = x2 >= 0 ? max(x2, r.x + r.width) : r.x + r.width;
            y2 = y2 >= 0 ? max(y2, r.y + r.height) : r.y + r.height;
        }
        else{
            res.push_back(Rect(x1, y1, x2-x1, y2-y1));
            x1 = r.x, y1 = r.y, x2 = r.x + r.width, y2 = r.y + r.height;
        }
    }
    if(x1>=0) res.push_back(Rect(x1, y1, x2-x1, y2-y1));
    return ;
}

int main( int argc, char** argv )
{

    Mat img, grey, lab_img, gradient_magnitude, segmentation, all_segmentations;

    vector<Region> regions;
    vector<Rect> rects;
    //::MSER mser8(true,20,0.00008,0.03,1,0.7);
    ::MSER mser8(true,15,0.00008,0.05,1,0.7);
    bool debug=argc>2;
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
        vector<Region> erased;
        for (int i=regions.size()-1; i>=0; i--)
        {
            Region &r=regions.at(i);
            r.extract_features(lab_img, grey, gradient_magnitude);  //|| (r.bbox_.width <=1)
            if ( (r.stroke_std_/r.stroke_mean_ > 0.8) || (r.num_holes_>2)  || (r.bbox_.height <=2)
                 || (r.bbox_.width > 3.5*r.bbox_.height) || r.stroke_std_>0.1*r.bbox_.height ){
                if(r.stroke_std_>0.1*r.bbox_.height)   {
                    erased.push_back(r);
                    //cout<<"mean width:"<<r.stroke_mean_<<" std width:"<<r.stroke_std_<<" width var"<<r.stroke_var_<<" region height:"<<r.bbox_.height<<endl;
                }
                regions.erase(regions.begin()+i);
            }
            else
                max_stroke = max(max_stroke, regions[i].stroke_mean_);
        }
        {
            Mat tmp = Mat::zeros(img.size(), CV_8UC3);
            drawMSERs(tmp, &erased, true, &img, true);
            char buf[100]; sprintf(buf, "out1/%s.%d.erased.jpg", argv[1], step);
            imwrite(buf, tmp);
        }
        {
            Mat tmp = Mat::zeros(img.size(), CV_8UC3);
            drawMSERs(tmp, &regions, true, &img, true);
            char buf[100]; sprintf(buf, "out1/%s.%d.mser.jpg", argv[1], step);
            imwrite(buf, tmp);
        }
        /////////
        cout<<"calculate graph"<<endl;
        int N = regions.size();
        vector<vector<float> > graph(N, vector<float>(N, 0));
        vector<vector<bool> > sameline(N, vector<bool>(N, false));
        {
            for (int i=0; i<N; i++){
                graph[i][i]=100;
                for(int j=i+1; j<N; j++){
                    graph[i][j]=graph[j][i]=dist(regions[i], regions[j]);
                    sameline[i][j]=sameline[j][i]=inSameLine(regions[i], regions[j]);
                }
            }
        }

        cout<<"graph calculated"<<endl;

        MaxMeaningfulClustering 	mm_clustering(METHOD_METR_SINGLE, METRIC_SEUCLIDEAN);
        vector<vector<int> > final_clusters;
        vector<vector<int> > blocks;
        vector<bool> blockTag;
        {
            //divide into connected tree
            float thres=20;
            vector<bool> vis(N, false);
            for(int i=0; i<N; i++){
                if(!vis[i]){
                    vector<int> tmp;
                    dfs(graph, thres, tmp, i, vis);
                    blocks.push_back(tmp);
                }
            }

            Mat tmp = Mat::zeros(img.size(), CV_8UC3);
            drawClusters(tmp, &regions, &blocks);
            char buf[100];
            sprintf(buf, "out1/%s.%d.block.jpg", argv[1], step);
            imwrite(buf, tmp);

            blockTag.resize(blocks.size(), false);
            for(int j=0; j<blocks.size(); j++){
                if ( (group_boost(&blocks.at(j), &regions) >= DECISION_THRESHOLD_SF) )
                {
                    final_clusters.push_back(blocks[j]);
                    blockTag[j]=true;
                }
            }

            tmp = Mat::zeros(img.size(), CV_8UC3);
            drawClusters(tmp, &regions, &final_clusters);
            sprintf(buf, "out1/%s.%d.block1.jpg", argv[1], step);
            imwrite(buf, tmp);
        }

        vector<vector<int> > lines;
        vector<int> lineNum(N, -1);
        vector<bool> lineTag(N, false);
        {
            int dim=2;
            t_float *data = (t_float*)malloc(dim*N * sizeof(t_float));

            for(int i=0; i< blocks.size(); i++){
                //if(blockTag[i]) continue;
                cutToLines(sameline, blocks[i], lines);
            }

            Mat tmp = Mat::zeros(img.size(), CV_8UC3);
            drawClusters(tmp, &regions, &lines);
            char buf[100]; sprintf(buf, "out1/%s.%d.lines0.jpg", argv[1], step);
            imwrite(buf, tmp);

            vector<vector<int> > srt;
            vector<vector<int> > rrt;
            for(int i=0; i<lines.size(); i++){
                vector<int> &line = lines.at(i);
                for(int j = 0; j < line.size(); j++){
                    lineNum[line[j]] = i;
                }
                int j=random()%1000;
                //cout<<"group "<<j<<":"<<endl;
                if(line.size()>1)
                {

                    float f = groupScore(graph,  line);
                    float g = groupVar(regions, line);
                    float h= group_boost(&line, &regions)/ DECISION_THRESHOLD_SF;

                    if(f>=200 && g<=0.015 && h>=0.1 || f>700 && g<=0.1 || h>DECISION_THRESHOLD_SF){
                        srt.push_back(line);
                        final_clusters.push_back(line);
                    }
                    else{
                        rrt.push_back(line);
                    }
                    if(debug){
                        cout<<"group"<<j<<":"<<f<<endl;
                        cout<<"var:"<<g<<endl;
                        cout<<"boost:"<<h<<endl;
                        Mat tmp = Mat::zeros(img.size(), CV_8UC1);
                        fillRegions(tmp, regions, line);
                        char buf[100]; sprintf(buf, "out1/%d.group.jpg",  j);
                        imwrite(buf, tmp);
                    }
                }
            }
            tmp = Mat::zeros(img.size(), CV_8UC3);
            drawClusters(tmp, &regions, &rrt);
            sprintf(buf, "out1/%s.%d.lines-1.jpg", argv[1], step);
            imwrite(buf, tmp);
        }


        //t = cvGetTickCount() - t;
        //cout << "Features extracted in " << t/((double)cvGetTickFrequency()*1000.) << " ms." << endl;
        //t = (double)cvGetTickCount();

        vector< vector<int> > meaningful_clusters;
        //vector< vector<int> > final_clusters;
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
            for(int j = 0; j < N; j++){
                if(!rs.count(j)){
                    float score=0;
                    for(set<int>::iterator it=rs.begin(); it!=rs.end(); it++){
                        {
                            if(graph[*it][j] > 40)
                                score += graph[*it][j];
                        }
                    }
                    if(score>600) bs.push_back(regions[j]), crt.push_back(j), rs.insert(j);
                }
            }
            ////
            final_clusters.push_back(crt);

            /////////
            final_clusters.clear();
            for(set<int>::iterator it = rs.begin(); it != rs.end(); it++){
                int t = *it, s = lineNum[t];
                if(s >= 0 && !lineTag[s])
                    lineTag[s] = true,  getLineRect(regions, lines[s], rects),
                            final_clusters.push_back(lines[s]);
            }

            Mat tmp = Mat::zeros(img.size(), CV_8UC3);
            drawMSERs(tmp, &bs, true, &img, true);
            char buf[100]; sprintf(buf, "out1/%s.%d.correct.jpg", argv[1], step);
            imwrite(buf, tmp);
        }

        // drawClusters(segmentation, &regions, &final_clusters);
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
            Mat tmp = Mat::zeros(img.size(), CV_8UC1);
            fillRegions(tmp, final_regions, res);
            //cvtColor(tmp, tmp, CV_BGR2GRAY);
            //threshold(tmp,tmp,1,255,CV_THRESH_BINARY);
            char buf[100];
            sprintf(buf, "out1/%s.out.png", argv[1]);
            imwrite(buf, tmp);

            for(int i=0; i<rects.size(); i++){
                Rect &r = rects[i];
                printf("%d, %d, %d, %d\n", r.x, r.y, r.x + r.width, r.y + r.height);
            }

        }
        regions.clear();
        //t_tot = cvGetTickCount() - t_tot;
        //cout << " Total processing for one frame " << t_tot/((double)cvGetTickFrequency()*1000.) << " ms." << endl;
    }

}

