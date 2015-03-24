#ifndef REGIONLINE_H_
#define REGIONLINE_H
#include "region.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include "utils.h"
using std::pair;
using std::sort;
using std::vector;
using std::cout;
using std::endl;
bool g_RegionLineTest=false;
#define FCOUT if(g_RegionLineTest) cout
void print(Region &r,ostream &out){
  Rect s=r.bbox_;
  out<<s.x<<","<<s.y<<"->"<<s.width+s.x<<","<<s.height+s.y;
}
void print(Rect &s){
  cout<<s.x<<","<<s.y<<"->"<<s.width+s.x<<","<<s.height+s.y;
}
void print(Region &r){
  print(r,cout);
}
static void regionTest(Region &r){
  // cout<<"test regions:"<<endl;
  // rep(i,std::min((size_t)5,tmpRegions.size()))
  // regionTest(tmpRegions[i]);
  rep(i,r.pixels_.size())
      cout<<r.pixels_[i]<<" ";
  cout<<endl;
}

struct RegionSort{
  bool	operator () (const Region &s,const Region &t){
    return  s.bbox_x1_<t.bbox_x1_;
  }
}cmper;

inline float area(Region &region){
  Rect &r=region.bbox_;
  return r.width*r.height;
}

class RegionLine{
public:
  static float inlineScore(Region &a,Region &b){
    // stroke
    float f = (float)a.stroke_mean_/b.stroke_mean_;
    if( f > 2 || f < 0.5 ) return 0;
    cv::Rect s = a.bbox_, t = b.bbox_;
    return inlineScore(s, t);
  }

  static float inlineScore(Rect &s, Rect &t)
  {
    float sw = s.width, tw = t.width, sh = s.height, th = t.height;
    float sx = s.x, tx = t.x, sy = s.y, ty = t.y;
    float sx1 = s.x+s.width, tx1 = t.x+t.width, sy1 = s.y+s.height, ty1 = t.y+t.height;

    //width height
//    {
//      float x = sw/tw, y = sh/th;
//      FCOUT<<"x:"<<x<<" y:"<<y<<endl;
//      if( x>2.2 && y>2.5|| (1/x)>2.2 && (1/y)>2.5 ) return 0;
//      if( y>3 || y<0.33 ) return 0;
//    }

    //y position
    {
      int hy=max(sy, ty),ly=min(sy1, ty1);
      if( ly < hy || ly-hy < min(sh, th)*2/7 ) return 0;
    }

    // too close or too far
    {
      if(tx < sx + sw/4) return 0;
      float x = t.x - sx1;
      if( x > max(sh, th)*3.5 ) return 0;
    }

    float dx = ( tx + tw/2 )-( sx + sw/2 );
    // float x=min(deltaX/s.width,deltaX/t.width);
    float x = dx/s.width;
    return 1/( x + 0.5 );
  }

  static inline	int xDis(Region &x,Region &y){
    // return y.bbox_.x+y.bbox_.width/2-(x.bbox_.x+x.bbox_.width/2);
    return y.bbox_.x-(x.bbox_.x+x.bbox_.width);
  }

  void dump(vector<Region> &regions,vector<Rect>& lines){
    rep(i,(regions.size())){
      Region &r=regions.at(i);
      lines.push_back(r.bbox_);
    }
  }

  static void classify(vector<Region>& regions,vector<int> &nxt, vector<vector<int> > &groups){
    int N = regions.size();
    groups.clear();
    nxt.resize(N, -1);

    vector<vector<float> > scores(N, vector<float>(N, 0));
    rep(i, N){
      for(int j=i+1;j<N;j++){
          scores[i][j] = scores[j][i] = inlineScore(regions[i], regions[j]);
      }
    }
    vector<int> num(N, -1);
    vector<vector<int> > idx(N, vector<int>());
    int numi = 0;
    rep(i, N){
      float s = 0; int si = -1;
      for(int j = 0; j<i; j++){
          if(scores[i][j] >0 && scores[i][j] >= s) s = scores[i][j], si = j;
      }
      if( si != -1)  num[i] = num[si], groups[num[i]].push_back(i), nxt[si]=i;
      else num[i] = numi++, groups.push_back(vector<int>(1, i));
    }
  }

  void classify(vector<Region>& regions,vector<int> &nxt,vector<vector<int> > &groups,vector<Rect>& textlines){
    int N=regions.size();
    textlines.clear();
    groups.clear();
    vector<int> pre(N,-1);
    vector<int> nw(N,-1);
    // vector<int> nxt(N,-1);
    nxt.resize(N,-1);
    vector<float>score(N,-1);

    FCOUT<<"classify regions into lines:"<<endl;
    rep(k,N){
      if(nxt[k]!=-1) continue;
      rep(j,N) score[j]=0;
      int i=k;
      while(i>=0){
          float m=0;
          int mi=-1;
          float mw=0;
          for(int j=i+1;j<N;j++){
              if(pre[j]!=-1) continue;
              // float threshold_width=(pre[i]>0?nw[i]:-1);
              score[j]=inlineScore(regions[i],regions[j]);//threshold_width);//tmp is the width of j and nxt[j], while
              FCOUT<<i<<","<<j<<":"<<score[j]<<"   ";
              if(score[j]>m){m=score[j],mi=j;}
            }
          if(mi>=0) {
              pre[mi]=i;
              nxt[i]=mi;
              //间隔
              nw[i]=xDis(regions[i],regions[mi]);//mi has pre as i, and prew[mi] is the width between i and mi
              FCOUT<<i<<"->"<<mi<<"    ";
            }
          i=mi;
        }
    }

    //divide into words

    //merge
    rep(i,N){
      if(pre[i]>=0) continue;//find the head of textlines
      int j=i;
      bool isInValid=true;
      while(j>=0){
          if(!isIStroke(regions[j])&&regions[j].area_<0.7*area(regions[j])) {
              isInValid=false;break;
            }
          j=nxt[j];
        }
      if(isInValid)  continue;

      vector<int> group;
      j=i;
      while(j>=0){
          group.push_back(j);
          j=nxt[j];
        }
      groups.push_back(group);
      textlines.push_back(dump(regions,group));
    }
  }

  inline bool isIStroke(Region &s){
    float w1=s.bbox_.width,h1=s.bbox_.height;
    //shape
    if(h1/w1<2) return false;
    if(h1/w1>6) return true;
    if(s.num_holes_>0) return false;
    return true;
  }

  static bool containedBy(Region &l,Region &r){//l contained by r
    int n=l.pixels_.size(),t=r.pixels_.size();
    if(n>=t) return false;
    int m=(n>=0&&n<3)?n:(n<20)?n/3:n/10;
    rep(i,m){
      int s=rand()%n;
      if(!binary_search(r.pixels_.begin(),r.pixels_.end(),l.pixels_[s])){
          return false;
        }
    }
    // FCOUT<<"region l contained by r"<<endl;
    return l.classifier_votes_<r.classifier_votes_;
  }

  //test if rect s and t can be merged into one linegroup
  //the score would be high if possible
  static float ilGroupScore(Rect &s,Rect &t){
    {
      float y=max(s.y,t.y),y1=min((s.y+s.height),(t.y+t.height));
      if(y1-y<0.7*max(s.height,t.height)) return 0;
    }
    {
      int x=min(s.x,t.x),y=min(s.y,t.y);
      int x1=max(s.x+s.width,t.x+t.width),y1=max(s.y+s.height,t.y+t.height);
      float m=(x1-x)*(y1-y)
          ,n=(s.width*s.height+t.width*t.height);
      if(m>1.2*n) return 0;
      else return n/m;
    }
  }

  //test if s and t can be merged into one linegroup
  static bool overlaps(Rect &s,Rect &t){//l overlaps r,then the
    float f=ilGroupScore(s,t);
    return f>0.7&&1/f>0.7;
  }

  //cal the possibility that rect can be in group with average ah,aw,ad,as
  static inline float ff(float ah,float aw,float ad,float as,float h,float w,float d,float s){
    FCOUT<<ah<<","<<aw<<","<<ad<<","<<as<<":"<<h<<","<<w<<","<<d<<","<<s<<endl;
    assert(ah>0&&aw>0&&as>0&&h>0&&w>0&&s>0);
    float s1=(ah>h?(float)ah/h:(float)h/ah);
    float s2=(aw>w?1:pow(w/aw,0.4));
    float s3=1;
    if(ad>0&&d>0)
      s3=(ad>d?(float)(ad+aw)/(d+aw):(float)(d+aw)/(ad+aw));
    float s4=(as>s?(float)as/s:s/as);
    float ss=s1*s2*s4;
    return ss;
  }

  //display
  static void display(int i,Region &region){
    Rect &r=region.bbox_;
    cout<<"region "<<i<<"->w:"<<r.width<<" h:"<<r.height<<" stroke:"<<region.stroke_mean_
       <<" x1,y1:"<<region.bbox_x1_<<","<<region.bbox_y1_
      <<" x2,y2:"<<region.bbox_x2_<<","<<region.bbox_y2_<<endl;
  }

  static Rect dump(vector<Region>&regions,vector<int> group){
    assert(!group.empty());
    int miny=0x7fffffff,maxy=0,maxx=0;
    int minx=regions[group[0]].bbox_x1_;
    rep(i,group.size()){
      int j=group[i];
      cv::Rect rect=regions[j].bbox_;
      miny=min(miny,rect.y);
      maxy=max(maxy,rect.y+rect.height);
      maxx=max(maxx,rect.x+rect.width);
    }
    return Rect(minx,miny,maxx-minx,maxy-miny);
  }

  //if false,group will be erased
  //group's regions are examined if it is character, in the order from the left and right
  //if not, it will be erased
  static bool cleanGroup(vector<Region>&regions,vector<int>&group){
    int n=group.size();
    FCOUT<<"before cleaned, group size:"<<group.size()<<endl;
    if(n<2) return true;
    float avgH=0,avgW=0,avgD=0,avgS=0;
    rep(i,n){
      int j=group[i];
      avgH+=regions[j].bbox_.height,avgW+=regions[j].bbox_.width,avgS+=regions[j].stroke_mean_;
      if(i>0) {
          int d=regions[group[i]].bbox_x1_-regions[group[i-1]].bbox_x2_;
          if(d>0) avgD+=d;
        }
    }
    // avgH/=n,avgW/=n,avgS/=n,avgD/=(n-1);
    while(group.size()>1){
        int n=group.size();
        Region &r=regions[group[0]];
        int h=r.bbox_.height,w=r.bbox_.width,s=r.stroke_mean_;
        int d=regions[group[1]].bbox_x1_-regions[group[0]].bbox_x2_;
        float ss=ff(avgH/n,avgW/n,avgD/(n-1),avgS/n,h,w,d,s);
        FCOUT<<"0:"<<ss<<endl;
        if(ss>3) {
            avgH-=h,avgW-=w,avgS-=s,avgD-=d;
            group.erase(group.begin());
          }
        else break;
      }
    while(group.size()>1){
        int n=group.size();
        Region &r=regions[group[n-1]];
        int h=r.bbox_.height,w=r.bbox_.width,s=r.stroke_mean_;
        int d=regions[group[n-1]].bbox_x1_-regions[group[n-2]].bbox_x2_;
        float ss=ff(avgH/n,avgW/n,avgD/(n-1),avgS/n,h,w,d,s);
        FCOUT<<"0:"<<ss<<endl;
        if(ss>3) {
            avgH-=h,avgW-=w,avgS-=s,avgD-=d;
            group.erase(group.end()-1);
          }
        else break;
      }
    FCOUT<<"cleaned group size:"<<group.size()<<endl;
    return true;
  }
  static void unique(vector<Region>& regions){
    // FCOUT<<"unique..."<<endl;
    int n=regions.size();
    vector<vector<int> > containArr(n,vector<int>());
    for(int i=regions.size()-1;i>=0;i--){
        rep(j,regions.size()){
          if(j!=i&&containedBy(regions[i],regions[j])){//i is contained by j?
              // Rect &l=regions[i].bbox_,&r=regions[j].bbox_;
              // print(l);FCOUT<<endl;print(r);FCOUT<<endl;
              // FCOUT<<"x,y overlaps"<<endl;
              containArr[j].push_back(i);
              break;
            }
        }
      }
    vector<int> eraseRs;
    rep(i,n){
      int m=containArr[i].size();
      if(m>0&&m<2){
          rep(j,containArr[i].size())
              eraseRs.push_back(containArr[i][j]);
        }
    }
    sort(eraseRs.begin(),eraseRs.end(),greater<int>());
    vector<int>::iterator it=std::unique(eraseRs.begin(),eraseRs.end());
    eraseRs.erase(it,eraseRs.end());
    rep(i,eraseRs.size())
        regions.erase(regions.begin()+eraseRs[i]);
  }
  static void unique(vector<Rect>& rects){
    // FCOUT<<"unique..."<<endl;
    for(int i=rects.size()-1;i>=0;i--){
        rep(j,rects.size()){
          if(j!=i){//&&overlaps(rects[i],rects[j])){//i is contained by j?
              Rect &s=rects[i],&t=rects[j];
              int x=min(s.x,t.x),y=min(s.y,t.y);
              int x1=max(s.x+s.width,t.x+t.width),y1=max(s.y+s.height,t.y+t.height);
              if(overlaps(s,t)){
                  rects[j]=Rect(x,y,x1-x,y1-y);
                  rects.erase(rects.begin()+i);
                  break;
                }
            }
        }
      }
  }
};
#endif
