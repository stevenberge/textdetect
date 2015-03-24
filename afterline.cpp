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
#include <list>
#define NUM_FEATURES 11 
#define DECISION_THRESHOLD_EA 0.5
#define DECISION_THRESHOLD_SF 0.999999999

#define APPENDVEC(a, b) a.insert(a.end(), b.begin(), b.end())
using namespace std;
using namespace cv;

#include "utils.h"
#include "groups.h"
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

Rect getLineRect(const vector<Region> &regions, const vector<int> &line){
    int n = line.size();
    if(n==0) return Rect(-1, -1, -1, -1);
    int x1=-1, x2=-1, y1=-1, y2=-1;
    for(int i=0; i<n; i++){
        Rect r=regions[line[i]].bbox_;
        x1 = min(x1, r.x);
        y1 = min(y1, r.y);
        x2 = max(x2, r.x + r.width);
        y2 = max(y2, r.y + r.height);
    }
    return Rect(x1, y1, x2-x1, y2-y1);
}
void getLineRects(const vector<Region> &regions, const vector<int> &line, vector<Rect> &res){
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
        assert(r.x>=x2);
        if( x2<0 || r.x-x2 < width*3){
            const Rect &r=regions[line[i]].bbox_;
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
int cntOverlap(Region &r,  vector<vector<int> > &ps, vector<bool> &tmp, int &mi, int &mn){
    map<int, int> ss;
    float n = r.pixels_.size(), cnt = 0;
    for(int i = 0; i< n; i++){
        for(int j = 0; j<ps[i].size(); j++){
            if(tmp[ps[i][j]])
                ss[ps[i][j]] ++;
        }
    }
    mn = 0, mi = -1;
    for(map<int, int>::iterator it = ss.begin(); it!=ss.end(); it++){
        if(it->second>1) cnt++;
        if(it->second>mn) mi = it->first, mn = it->second;
    }
    return cnt;
}

bool overlaps(Region &r, vector<int> &ps, vector<bool> &ns, int &t, int &cnt){
    float n = r.pixels_.size();
     t = -1, cnt = 0;
    for(int i = 0; i< n; i++){
        int c = r.pixels_[i];
        if(ps[c]>=0 && ns[ps[c]]) {
            if(t<0) t=ps[c], cnt=1;
            else if(t==ps[c]) cnt++;
            else return false;
        }
    }
    return true;
}
bool overlaps(Region &r, set<int>  &ps, float thr=0.6){
    float n = r.pixels_.size(), cnt = 0;
    for(int i = 0; i< n; i++){
        if(ps.count(r.pixels_[i])) cnt++;
    }
    return cnt >= thr*n;
}
bool overlaps(Region &r, vector<bool> &ps, float thr=0.6){
    float n = r.pixels_.size(), cnt = 0;
    for(int i = 0; i< n; i++){
        if(ps[r.pixels_[i]]) cnt++;
    }
    return cnt >= thr*n;
}

void postUnique(vector<Region> &regions){
    int n = regions.size();
    vector<set<int> > sets(n);
    for(int i = 0; i<n; i++){
        for(int j = 0; j<regions[i].pixels_.size(); j++){
            sets[i].insert(regions[i].pixels_[j]);
        }
    }
    vector<Region> res;
    for(int i = 0; i<n; i++){
        int j; for(j = i+1; j<n; j++){
            if(overlaps(regions[i], sets[j], 0.7)) break;// has son
        }
        if(j==n) res.push_back(regions[i]);
    }
    regions=res;
}
typedef vector<Region>::iterator IT;
struct RegionCmp{
    bool operator()(const IT &a, const IT &b){
        return   a->area_<=b->area_;
    }
}regionCmp;
int main( int argc, char** argv )
{
    char buf[100];
    Mat img, grey, lab_img, gradient_magnitude, segmentation, all_segmentations;
    //::MSER mser8(true,20,0.00008,0.03,1,0.7);
    bool debug = (argc>2 && strcmp(argv[2], "true")==0);

    char *filename = argv[1];
    #define IFOPT   if(debug)
    //::MSER mser8(true, thr,0.00008,0.005,1,0.7);

    RegionClassifier region_boost("boost_train/trained_boost_char.xml", 0);
    GroupClassifier  group_boost("boost_train/trained_boost_groups.xml", &region_boost);



    img = imread(filename);


    cvtColor(img, grey, CV_BGR2GRAY);
    cvtColor(img, lab_img, CV_BGR2Lab);
    gradient_magnitude = Mat_<double>(img.size());
    get_gradient_magnitude( grey, gradient_magnitude);

    segmentation = Mat::zeros(img.size(),CV_8UC3);
    all_segmentations = Mat::zeros(240,320*11,CV_8UC3);

   // int thrs[6] = {75, 34, 30, 25, 16, 10};
     // int thrs[5] = {50, 34, 30, 18, 13};

     int thrs[5] = { 80, 30, 18};


    vector<Region> ff_regions;
    vector<Region> ff_dots;
    list<Region> dots[2];
    vector<vector<int> > pixels(2, vector<int>(img.cols*img.rows, -1));
    vector<Region> valid_regions[2];
    vector<bool> valid_num[2];
    vector<int> valid_lcs[2];
    ///////////////////////////
    {
        ::MSER mser8(true, 10,0.000001,0.008,0.5,0.33);

        ////// dots
        for (int step =1; step<3; step++)
        {
            if (step == 2)
                grey = 255-grey;
            vector<Region> regions;
            mser8((uchar*)grey.data, grey.cols, grey.rows, regions);
            for (int i=0; i<regions.size(); i++){
                regions[i].er_fill(grey);
                regions[i].extract_features(lab_img, grey, gradient_magnitude);
            }

            IFOPT{
                Mat tmp = Mat::zeros(img.size(), CV_8UC1);
                fillRegions(tmp, regions);
                char buf[100]; sprintf(buf, "out1/%s.%d.0dots.png", filename, step);


                imwrite(buf, tmp);
            }
            for(int i= 0; i<regions.size(); i++){
                if(isDotStroke(regions[i])) {
                    dots[step-1].push_back(regions[i]);
                }
            }

            IFOPT{
                Mat tmp = Mat::zeros(img.size(), CV_8UC1);
                fillRegions(tmp, dots[step-1]);
                char buf[100]; sprintf(buf, "out1/%s.%d.dots.png", filename, step);


                imwrite(buf, tmp);
            }
        }
    }
    grey = 255-grey;

    /////////////////////////////////////////

    int mid = -1;
    for (int step =1; step<3; step++)
    {
        if (step == 2)
            grey = 255-grey;


        for(int iii = 0; iii<3; iii++){


            ::MSER mser8(true, thrs[iii],0.00008,0.08,0.5,0.33);

            vector<Region> regions;
            vector<Region> final_regions;
            vector<int> line_sizes;
            //    vector<vector<int> > final_lines;
            vector<Region> line_regions;

            cout<<"step:"<<step<<"."<<iii<<endl;
            cout<<"mser"<<endl;

            mser8((uchar*)grey.data, grey.cols, grey.rows, regions);
            cout<<regions.size()<<" regions extracted"<<endl;


            for (int i=0; i<regions.size(); i++){
                regions[i].er_fill(grey);
                regions[i].extract_features(lab_img, grey, gradient_magnitude);
            }

            cout<<"after inited msers"<<endl;

            IFOPT{
                Mat tmp = Mat::zeros(img.size(), CV_8UC3);
                drawMSERs(tmp, &regions, true, &img, true);
                char buf[100]; sprintf(buf, "out1/%s.%d.%d.mser0.jpg", filename, step, iii);


                imwrite(buf, tmp);
            }

            {
                // regions are sorted in ids<int, int>

                // regions -> ids[]


                vector<bool > ps(img.cols*img.rows, false);
                vector<pair<int, int> > ids;
                for(int i = 0; i < regions.size(); i++){
                    ids.push_back(pair<int, int>(regions[i].area_, i));
                }
                sort(ids.begin(), ids.end());
                cout<<"after  sort"<<endl;
                // merged regions are kicked off
                vector<bool> tmp(regions.size(), false);
                for(int i = 0; i<ids.size(); i++){
                    if(overlaps(regions[ids[i].second], ps, 0.6)) continue;
                    tmp[ids[i].second] =true;
//                    int mi, mn;
//                    if(cntOverlap( regions[ids[i].second], ps, tmp, mi, mn)>1) continue;
//                    if(mi>=0){
//                        tmp[mi] = false;
//                    }
//                    tmp[ids[i].second]= true;
                    for(int j = 0; j<regions[ids[i].second].pixels_.size(); j++)
                        ps[regions[ids[i].second].pixels_[j]] = true;//.push_back(ids[i].second);
                }
                vector<Region> sr;
                for(int i = 0; i<ids.size(); i++)
                    if(tmp[ids[i].second]) sr.push_back(regions[ids[i].second]);

                    //else cout<<"regions "<<ids[i].second<<" is kicked off"<<endl;


                regions = sr;
            }

            IFOPT{
                Mat tmp = Mat::zeros(img.size(), CV_8UC1);
                fillRegions(tmp, regions);

                char buf[100]; sprintf(buf, "out1/%s.%d.%d.mser--.jpg", filename, step, iii);


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
                //|| (r.bbox_.width <=1)
                if ( (r.stroke_std_/r.stroke_mean_ > 0.8) || (r.num_holes_>3)  || (r.bbox_.height <=2) || r.area_ <=4
                     || (r.bbox_.width > 8.5*r.bbox_.height)  ){//|| r.stroke_mean_>0.3*r.bbox_.height
                    erased.push_back(r);
                    regions.erase(regions.begin()+i);
                }
                else
                    max_stroke = max(max_stroke, regions[i].stroke_mean_);
            }
            IFOPT{
                Mat tmp = Mat::zeros(img.size(), CV_8UC3);
                drawMSERs(tmp, &erased, true, &img, true);

                char buf[100]; sprintf(buf, "out1/%s.%d.erased.jpg", filename, step);


                imwrite(buf, tmp);
            }
            IFOPT{
                Mat tmp = Mat::zeros(img.size(), CV_8UC3);
                drawMSERs(tmp, &regions, true, &img, true);

                char buf[100]; sprintf(buf, "out1/%s.%d.mser.jpg", filename, step);
                imwrite(buf, tmp);
            }
            /////////
            // regions -> graph[][], sameline[][]


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


            ////////
            // graph, vis -> blocks


            vector<vector<int> > final_clusters;
            vector<vector<int> > blocks;
            vector<bool> blockTag;
            {
                //divide into connected tree
                float thres=25;
                vector<bool> vis(N, false);
                for(int i=0; i<N; i++){
                    if(!vis[i]){
                        vector<int> tmp;
                        dfs(graph, thres, tmp, i, vis);
                        blocks.push_back(tmp);
                    }
                }
                IFOPT{
                    Mat tmp = Mat::zeros(img.size(), CV_8UC3);
                    drawClusters(tmp, &regions, &blocks);
                    char buf[100];

                    sprintf(buf, "out1/%s.%d.block.jpg", filename, step);


                    imwrite(buf, tmp);
                }

                blockTag.resize(blocks.size(), false);
            }

            ////////
            // blocks[][], sameline[] -> lines[][], lineNum[], rects[]


            vector<vector<int> > lines;
            vector<Rect> rects;
            vector<bool> lineTags;
            vector<int> lineNum(N, -1);
            {
                int dim=2;
                t_float *data = (t_float*)malloc(dim*N * sizeof(t_float));

                for(int i=0; i< blocks.size(); i++){
                    //if(blockTag[i]) continue;
                    cutToLines(sameline, blocks[i], lines);
                }

                IFOPT{
                    Mat tmp = Mat::zeros(img.size(), CV_8UC3);
                    drawClusters(tmp, &regions, &lines);

                    char buf[100]; sprintf(buf, "out1/%s.%d.lines0.jpg", filename, step);


                    imwrite(buf, tmp);
                }

                rects.resize(lines.size());
                lineTags= vector<bool>(lines.size(), false);

                vector<vector<int> > srt;
                vector<vector<int> > rrt;
                for(int i=0; i<lines.size(); i++){
                    vector<int> &line = lines.at(i);
                    for(int j = 0; j < line.size(); j++){
                        lineNum[line[j]] = i;
                    }
                    rects[i] = getLineRect(regions, lines[i]);

                    int n = line.size();
                    // cout<<"line"<<i<<" : n="<< line.size()<<endl ;

                    /////////
                    // line, graph, regions -> final_clusters(valid line added), srt&rrt(invalid lines)


                    if(n>1)
                    {
                        IFOPT{
                            Mat tmp = Mat::zeros(img.size(), CV_8UC1);
                            fillRegions(tmp, regions, line);

                            char buf[100]; 
                            sprintf(buf, "out1/%s.%d.%d.%d.line.png",  filename, step, iii, i);


                            imwrite(buf, tmp);
                        }

                        float f = groupSim(graph,  line);
                        float g = groupVar(regions, line);
                        float h= group_boost(&line, &regions);
                        bool k= groupSco(region_boost, regions, line);
                        LineFeature lf;
                        if(!groupLs(regions, line, i, lf)) continue;
                        pair<float, float> p = groupAlias(regions, line);
                        if((p.first>0.18 && p.second>0.3 || p.first>0.25 || p.second>0.5) && n<5 && lf.sw_var>0.01 ||
                                (n==2 && lf.sw_var>0.01 && p.first>0.15)) {
                            cout<<"line"<<i<<" : n="<< line.size() <<", oy="<<p.first<<", ox="<<p.second<<endl;
                            continue;
                        }
                        g/=n;
                        if(n<2 && g>0.005) continue;
                        if(f>=200 && g<=0.01 && h>=0.1 || f>120 && g<=0.01
                                || f>150 && g<=0.02 || f>400 && g<0.01 || h>DECISION_THRESHOLD_SF*2/3 || g<0.02 && k){
                            cout<<"#line"<<i<<" : n="<< line.size() <<", f="<<f<<", g="<<g<<", h="<<h<<", k="<<k<<endl;
                            srt.push_back(line);
                            final_clusters.push_back(line);
                            lineTags[i] = true;
                        }
                        else{
                            cout<<"line"<<i<<" : n="<< line.size() <<", f="<<f<<", g="<<g<<", h="<<h<<", k="<<k<<endl;
                            rrt.push_back(line);
                        }
                    }


                    ////////
                    // line, regions, rects[], dots -> ff_dots(valid dots added)


                    if(lineTags[i]){
                        float sw = groupSW(regions, line);
                        Rect &r = rects[i];
                        Rect  q(r.x, r.y+r.height/2, r.x+r.width, r.y+r.height);
                        for(list<Region>::iterator it = dots[step-1].begin(); it!=dots[step-1].end(); ){
                            Region & dot = *it;
                            int x = (dot.bbox_x1_+ dot.bbox_x2_)/2, y = (dot.bbox_y1_+dot.bbox_y2_)/2;
                            Point p(x, y);
                            if(dot.stroke_mean_<1.5*sw && dot.stroke_mean_>0.3*sw && within(p, q)){
                                int left =0, right =0;
                                int j; for( j = 0; j<line.size(); j++){
                                    Region & t = regions[line[j]];
                                    int d = min(abs(t.bbox_x1_-x), abs(x-t.bbox_x2_)), e = t.bbox_y2_ -y;
                                    bool l = abs(t.bbox_x1_-x) < abs(x-t.bbox_x2_);
                                    if(d<t.bbox_.width && e<t.bbox_.height/3 && e>0) {
                                        if(l) left ++;
                                        else right ++;
                                    }
                                }
                                if(left  && right)
                                    ff_dots.push_back(dot), it = dots[step-1].erase(it);
                                else it++;
                            }
                            else it++;
                        }
                    }
                }
                IFOPT{
                    Mat tmp = Mat::zeros(img.size(), CV_8UC3);
                    drawClusters(tmp, &regions, &rrt);

                    sprintf(buf, "out1/%s.%d.lines-1.jpg", filename, step);


                    imwrite(buf, tmp);
                }
            }

            ///////////////////////////////////////////////////////////////
//            {
//                set<int> rs;
//                for(int i=0; i<final_clusters.size(); i++){
//                    for(int j=0; j<final_clusters[i].size(); j++){
//                        rs.insert(final_clusters[i][j]);
//                    }
//                }
//                // cout<<"after insert"<<endl;
//                vector<Region> bs;
//                vector<int> crt;
//                for(int j = 0; j < N; j++){
//                    if(!rs.count(j)){
//                        float score=0;
//                        for(set<int>::iterator it=rs.begin(); it!=rs.end(); it++){
//                            {
//                                if(graph[*it][j] > 40)
//                                    score += graph[*it][j];
//                            }
//                        }
//                        if(score>600) bs.push_back(regions[j]), crt.push_back(j), rs.insert(j);
//                    }
//                }
//                ////
//                final_clusters.push_back(crt);
//                IFOPT{
//                    cout<<"as crt is chosen..."<<endl;
//                    Mat tmp = Mat::zeros(img.size(), CV_8UC1);
//                    fillRegions(tmp, regions, crt);
//                    sprintf(buf, "out1/crt.png");
//                    imwrite(buf, tmp);
//                }

//                /////////
//                //final_clusters.clear();
//                for(set<int>::iterator it = rs.begin(); it != rs.end(); it++){
//                    int t = *it, s = lineNum[t];
//                    if(s >= 0 && !lineTag[s])
//                        lineTag[s] = true;
//                }
//            }

            ///////////////////////////////////////////////////////////////
            // drawClusters(segmentation, &regions, &final_clusters);

            // final_clusters, regions -> final_regions


            cout<<"store final regions"<<endl;
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
                    line_sizes.push_back(lines[lineNum[i]].size());
                }
            }


            // overlay check
            // final_regions, pixels -> valid_num
            


            int offset= valid_regions[step-1].size();
            //valid_regions[step-1].append(final_regions);
            APPENDVEC(valid_regions[step-1], final_regions);
            vector<bool> tmp(final_regions.size(), false);
            APPENDVEC(valid_num[step-1], tmp);
            APPENDVEC(valid_lcs[step-1], line_sizes);


            for(int i = 0; i< final_regions.size(); i++){
                int k ,c;
                if(!overlaps(final_regions[i], pixels[step-1], valid_num[step-1], k, c)) {
                    //cout<<"the region collapse with previous ones"<<endl;


                    continue;
                }
                if(k>=0) {
                    float s= c/(float)valid_regions[step-1][k].area_,
                            t=final_regions[i].area_/(float)valid_regions[step-1][k].area_;
                    if(t<1.5) continue;  // not a parent of k
                    valid_num[step-1][k] = false;
                }

               // cout<<"the region is valid"<<endl;


                valid_num[step-1][i+offset]=true;
                for(int j = 0; j<final_regions[i].pixels_.size(); j++)
                    pixels[step-1][final_regions[i].pixels_[j]]=i + offset;
            }


            {
//                vector<int> tv;
//                for(int i = 0; i<valid_regions[step-1].size(); i++)
//                    if(valid_num[step-1][i]) tv.push_back(i);
                Mat tmp = Mat::zeros(img.size(), CV_8UC1);
                fillRegions(tmp, final_regions);//valid_regions[step-1], tv);
                char buf[100];

                sprintf(buf, "out1/%s.%d.%d.out.png", filename, step, iii);


                imwrite(buf, tmp);
            }
        }

        //////////////////////////////////////////
        {
            vector<int> tv;
            for(int i = 0; i<valid_regions[step-1].size(); i++)
                if(valid_num[step-1][i]) tv.push_back(i);
            Mat tmp = Mat::zeros(img.size(), CV_8UC1);
            fillRegions(tmp, valid_regions[step-1], tv);
            char buf[100];

            sprintf(buf, "out1/%s.%d.out.png", filename, step);


            imwrite(buf, tmp);
        }
    }
    ////////////////////////////////////////////
    {

        ///// recognize i and j from dots[]
        // dots + valid_regions => valid_regions(dots merged), ff_dots, is, 


        vector<Region> is;

        for(int step =1; step<3; step++){
            for(int i = 0; i<valid_regions[step-1].size(); i++){
                if(!valid_num[step-1][i]) continue;
                if(isHorizonStroke(valid_regions[step-1][i], region_boost)){
                    for(list<Region>::iterator it = dots[step-1].begin(); it!=dots[step-1].end(); ){
                        if(isI(*it, valid_regions[step-1][i])){
                            ff_dots.push_back(*it);

                            //is.push_back(*it),
                            valid_regions[step-1][i].merge(&(*it));


                            is.push_back(valid_regions[step-1][i]);
                            it = dots[step-1].erase(it);
                        }
                        else it++;
                    }
                }
            }
        }

        IFOPT{
            Mat tmp = Mat::zeros(img.size(), CV_8UC1);
            fillRegions(tmp, is);
            sprintf(buf, "out1/%s.is.png", filename);
            imwrite(buf, tmp);
        }
        IFOPT{
            Mat tmp = Mat::zeros(img.size(), CV_8UC1);
            fillRegions(tmp, ff_dots);
            sprintf(buf, "out1/%s.ds.png", filename);


            imwrite(buf, tmp);
        }
    }
    //postUnique(ff_regions[step-1]);


    ///////
    // valid_regions -> lines
    for(int step = 0; step<2; step++){
      int N = valid_regions[step].size();
      vector<int> block;
      vector<vector<int> > lines;
      vector<vector<bool> > sameline(N, vector<bool>(N, false));
      for(int i = 0; i<N; i++) {
        block.push_back(i);
        for(int j = 0; j<N; j++){
          sameline[i][j]=sameline[j][i]=inSameLine(valid_regions[step][i], valid_regions[step][j]);
        }
      }
      cutToLines(sameline, block, lines);
      for(int i = 0; i<lines.size(); i++) {
        Mat tmp = Mat::zeros(img.size(), CV_8UC1);
        fillRegions(tmp, valid_regions[step], lines[i]);
        char buf[100];
        sprintf(buf, "rect/%s.%d.%d.rect.png", filename, step, i);
        imwrite(buf, tmp);
      }
    }


    ///////////////////////////////////
    //merge valid_regions[0 and 1] -> ff_regions
   for(int i = 0; i<valid_num[0].size(); i++){


        if(!valid_num[0][i]) continue;
        Region &s = valid_regions[0][i];
        int j = 0; for( j = 0; j<valid_num[1].size(); j++){
            if(!valid_num[1][j]) continue;
            Region & t=valid_regions[1][j];
            //if(overlay(s, t) && boxArea(s)<boxArea(t)) break;//valid_lcs[0][i] <valid_lcs[1][j]) break;
            if(contains(t, s) && boxArea(s)<boxArea(t)) break;
        }
        if(j==valid_num[1].size()){
            ff_regions.push_back(valid_regions[0][i]);
        }
    }
    for(int i = 0; i<valid_num[1].size(); i++){
        if(!valid_num[1][i]) continue;
        Region &s = valid_regions[1][i];
        int j = 0; for( j = 0; j<valid_num[0].size(); j++){
            if(!valid_num[0][j]) continue;
            Region & t=valid_regions[0][j];
            //if(overlay(s, t) && valid_lcs[1][i] <valid_lcs[0][j]) break;
            //if(overlay(s, t) && boxArea(s)<boxArea(t)) break;
            if(contains(t, s) && boxArea(s)<boxArea(t)) break;
        }
        if(j==valid_num[0].size()){
            ff_regions.push_back(valid_regions[1][i]);
        }
    }

    for(int i = 0; i<ff_dots.size(); i++){
        int j = 0; for(; j<ff_regions.size(); j++){
            if(overlay(ff_regions[j], ff_dots[i])) break;
        }
        if(j==ff_regions.size()) ff_regions.push_back(ff_dots[i]);
    }




    //APPENDVEC(ff_regions, ff_dots);
    {
        Mat tmp = Mat::zeros(img.size(), CV_8UC1);
        fillRegions(tmp, ff_regions);
        char buf[100];

        sprintf(buf, "out1/%s.out.png", filename);


        imwrite(buf, tmp);
    }

}
