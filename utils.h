#ifndef UTILS_H
#define UTILS_H
#include "region.h"
#include <set>
#include <list>
#define rep(i,n) for(int i=0;i<(int)n;i++)
void accumulate_evidence(vector<int> *meaningful_cluster, int grow, Mat *co_occurrence)
{
  //for (int k=0; k<meaningful_clusters->size(); k++)
  for (int i=0; i<meaningful_cluster->size(); i++)
    for (int j=i; j<meaningful_cluster->size(); j++)
      if (meaningful_cluster->at(i) != meaningful_cluster->at(j))
        {
          co_occurrence->at<double>(meaningful_cluster->at(i), meaningful_cluster->at(j)) += grow;
          co_occurrence->at<double>(meaningful_cluster->at(j), meaningful_cluster->at(i)) += grow;
        }
}

inline void show(const Mat &img){
  imshow("..",img);
  waitKey(0);
}

void get_gradient_magnitude(Mat& _grey_img, Mat& _gradient_magnitude)
{
  cv::Mat C = cv::Mat_<double>(_grey_img);

  cv::Mat kernel = (cv::Mat_<double>(1,3) << -1,0,1);
  cv::Mat grad_x;
  filter2D(C, grad_x, -1, kernel, cv::Point(-1,-1), 0, cv::BORDER_DEFAULT);

  cv::Mat kernel2 = (cv::Mat_<double>(3,1) << -1,0,1);
  cv::Mat grad_y;
  filter2D(C, grad_y, -1, kernel2, cv::Point(-1,-1), 0, cv::BORDER_DEFAULT);

  for(int i=0; i<grad_x.rows; i++)
    for(int j=0; j<grad_x.cols; j++)
      _gradient_magnitude.at<double>(i,j) = sqrt(pow(grad_x.at<double>(i,j),2)+pow(grad_y.at<double>(i,j),2));

}

static uchar bcolors[][3] = 
{
  {0,0,255},
  {0,128,255},
  {0,255,255},
  {0,255,0},
  {255,128,0},
  {255,255,0},
  {255,0,0},
  {255,0,255},
  {255,255,255}
};

//
void drawText(Mat &img,Region &region,char* str,int width,double hScale,double vScale,int h,int r, int g, int b){
  CvFont font;
  int lineW=width;
  // double hScale=0.6,vScale=0.6;
  cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,hScale,vScale,0,lineW);
  IplImage src=img;
  cvPutText(&src,str,cvPoint(region.bbox_x1_,region.bbox_y1_+h),&font,CV_RGB(r,g,b));
}
void drawText(Mat &img,Region &region,char* str,int width,double hScale,double vScale){
  drawText(img,region,str,width,hScale,vScale,-5,255,0,0);
}
inline int rdm(){
  return rand()%256;
}
//将region通过红线连起来
void drawLines(Mat &img,vector<Region> &regions,vector<int> &nxt){
  IplImage src=img;
  for(int i=0;i<regions.size();i++){
      int j=nxt[i];
      if(j<0) continue;
      Rect& r=regions[i].bbox_,&t=regions[j].bbox_;
      Point p1(r.x+r.width/2,r.y+r.height/1),p2(t.x+t.width/2,t.y+t.height/1);
      // cvLine(&src,p1,p2,CV_RGB(rdm(),rdm(),rdm()),2,0,0);
      cvLine(&src,p1,p2,CV_RGB(255,0,0),2,0,0);
    }
}
void drawLines(Mat &img,vector<Region> &regions,vector<int> &group,vector<int> &nxt){
  IplImage src=img;
  // cout<<"draw region lines:";rep(i,group.size()) cout<<group[i]<<" ";cout<<endl;
  int a=rdm(),b=rdm(),c=rdm();
  rep(t_i,group.size()){
    int i=group[t_i];
    int j=nxt[i];
    if(j<0||group.end()==find(group.begin(),group.end(),j)) continue;
    Rect& r=regions[i].bbox_,&t=regions[j].bbox_;
    Point p1(r.x+r.width/2,r.y+r.height/1),p2(t.x+t.width/2,t.y+t.height/1);
    cvLine(&src,p1,p2,CV_RGB(a,b,c),3,0,0);
  }
}

void drawRegionScores(Mat &img,vector<Region> &regions){
  char str[10];
  for(int i=0;i<regions.size();i++){
      sprintf(str,"%.1f",regions[i].classifier_votes_);
      drawText(img,regions[i],str,1,0.5,0.5);
    }
}
void drawGroupScore(Mat &img,vector<Region> &regions,vector<int> &group,float score){
  if(group.size()<=0) return;
  Region &region=regions[group[0]];
  char str[10];
  sprintf(str,"%.3f:%d",score,group.size());
  drawText(img,region,str,1,0.6,0.6);
}

//@画出填充之后的region
void drawMSERs(Mat& img, vector<Region> *regions, bool fill, Mat *orImg, bool singleColor)
{
  //img = img*0;
  int w=img.cols,h=img.rows;
  uchar* rsptr = (uchar*)img.data;
  if(fill)
  for (int i=0; i<regions->size(); i++)
    {
      Region & r = regions->at(i);
      set<int> area;
      for (int p=0; p<regions->at(i).pixels_.size(); p++){
          int j=regions->at(i).pixels_.at(p);
          area.insert(j);
        }
      // cout<<"region "<<i<<" size "<<regions->at(i).pixels_.size()<<endl;
      for (int p=0; p<regions->at(i).pixels_.size(); p++)
        {
          int j=regions->at(i).pixels_.at(p);
          if(singleColor){
              rsptr[j*3] = 255; // (j*30%255);//255;//bcolors[i%9][2];
              rsptr[j*3+1] = 255;//bcolors[i%9][1];
              rsptr[j*3+2] = 255;//(j*20%255);//bcolors[i%9][0];
            }else{
              uchar* ssptr = (uchar*)orImg->data;
              rsptr[j*3] = ssptr[j*3] ;
              rsptr[j*3+1] = ssptr[j*3+1] ;
              rsptr[j*3+2] =  ssptr[j*3+2] ;
            }
        }
    }
  if(!fill){
      int cols = img.cols;
      vector<vector<Point> > contours;
      for (int i=0; i<regions->size(); i++)
      {
            Region & r = regions->at(i);
            //vector<Point> ps;
            for(int k = 0; k < r.contour_.size(); k++){
                Point & p = r.contour_[k];
                int j = p.x * cols + p.y;
                rsptr[j*3] = 255;
                rsptr[j*3+1] = 255 ;
                rsptr[j*3+2] =  255 ;
            }
      }
      for (int i=0; i<regions->size(); i++)
        cv::drawContours(img, contours, i, Scalar(255, 255, 255));
    }
}
void drawRects(Mat &canvas,vector<cv::Rect> &rects,Scalar scalar,int thickness);
//@画出rect的 boundbox
void drawRects(Mat &canvas,vector<cv::Rect> &rects){
  drawRects(canvas,rects,Scalar(0,255,0),2);
}
void drawRects(Mat &canvas,vector<cv::Rect> &rects,Scalar scalar,int thickness){
  vector<vector<Point> > contours;
  for (int i=rects.size()-1; i>=0; i--)
    {
      vector<Point> contour;
      Rect s=rects.at(i);
      // cout<<"region "<<i<<" level:"<<s.level_<<" bound:"<<
      // s.bbox_x1_<< s.bbox_y1_<< s.bbox_x2_<< s.bbox_y2_<< endl;
      //@@画出region的边框
      contour.push_back(Point(s.x,s.y));
      contour.push_back(Point(s.x+s.width,s.y));
      contour.push_back(Point(s.x+s.width,s.y+s.height));
      contour.push_back(Point(s.x,s.y+s.height));
      contours.push_back(contour);
    }
  drawContours(canvas, contours, -1, scalar, thickness);//CV_FILLED, 8);
}
//@画出region的 boundbox
void drawRegions(Mat &canvas,vector<Region> &regions){
  vector<vector<Point> > contours;
  for (int i=regions.size()-1; i>=0; i--)
    {
      vector<Point> contour;
      Region s=regions.at(i);
      // cout<<"region "<<i<<" level:"<<s.level_<<" bound:"<<
      // s.bbox_x1_<< s.bbox_y1_<< s.bbox_x2_<< s.bbox_y2_<< endl;
      //@@画出region的边框
      contour.push_back(Point(s.bbox_x1_,s.bbox_y1_));
      contour.push_back(Point(s.bbox_x1_,s.bbox_y2_));
      contour.push_back(Point(s.bbox_x2_,s.bbox_y2_));
      contour.push_back(Point(s.bbox_x2_,s.bbox_y1_));
      contours.push_back(contour);
    }
  drawContours(canvas, contours, -1, Scalar(1,1,255), 2);//CV_FILLED, 8);
}
//画出不同cluster，每种颜色不一
void drawClusters(Mat& img, vector<Region> *regions, vector<vector<int> > *meaningful_clusters)
{
  //img = img*0;
  uchar* rsptr = (uchar*)img.data;
  for (int i=0; i<meaningful_clusters->size(); i++)
    {

      for (int c=0; c<meaningful_clusters->at(i).size(); c++)
        {
          float area=(float)(regions->at(meaningful_clusters->at(i)[c]).area_)/(img.cols*img.rows);
          // if(area>0.1) cout<<"draw cluster, area size:"<<area<<endl;
          for (int p=0; p<regions->at(meaningful_clusters->at(i).at(c)).pixels_.size(); p++)
            {
              rsptr[regions->at(meaningful_clusters->at(i).at(c)).pixels_.at(p)*3] = bcolors[i%9][2];
              rsptr[regions->at(meaningful_clusters->at(i).at(c)).pixels_.at(p)*3+1] = bcolors[i%9][1];
              rsptr[regions->at(meaningful_clusters->at(i).at(c)).pixels_.at(p)*3+2] = bcolors[i%9][0];
            }
        }
    }
}
void fillRegions(Mat& img, vector<Region> &regions,vector<int> &group)
{
  //img = img*0;
  uchar* rsptr = (uchar*)img.data;
  for (int l=0; l<group.size(); l++)
    {
      int i=group[l];
      for (int p=0; p<regions[i].pixels_.size(); p++)
        {
          rsptr[regions[i].pixels_.at(p)] = 255;
      //    rsptr[regions[i].pixels_.at(p)*3+1] = 255;//bcolors[i%9][2];
      //    rsptr[regions[i].pixels_.at(p)*3+2] =255;// bcolors[i%9][2];
        }
    }
}
void fillRegions(Mat& img, vector<Region> &regions)
{
  //img = img*0;
  uchar* rsptr = (uchar*)img.data;
  for (int i=0; i<regions.size(); i++)
    {
      for (int p=0; p<regions.at(i).pixels_.size(); p++)
        {
          rsptr[regions[i].pixels_.at(p)] = 255;
        }
    }
}

void fillRegions(Mat& img, list<Region> &regions)
{
    //img = img*0;
    uchar* rsptr = (uchar*)img.data;
    for (list<Region>::iterator it = regions.begin(); it!=regions.end(); it++)
    {
      Region & r = *it;
      for (int p=0; p<r.pixels_.size(); p++)
        {
          rsptr[r.pixels_.at(p)] = 255;
        }
    }
}

#define NUM_FEATURES 11 
//@提取regions的某种feature到data[]
void extractFeatureArr(Mat &img,vector<Region>&regions,int f,int &dim,t_float **p_data){
  static int dims[NUM_FEATURES] = {3,3,3,3,3,3,3,3,3,5,5};
  unsigned int N = regions.size();
  if (N<3) {
      *p_data=NULL;
      return;
    }

  double max_stroke = -1;
  if(f==5){//stroke_mean_/max_stroke
      for(int i=0;i<N;i++)
        max_stroke=max(max_stroke, regions[i].stroke_mean_);
    }

  dim = dims[f];
  t_float* data=*p_data = (t_float*)malloc(dim*N * sizeof(t_float));
  int count = 0;
  for (int i=0; i<regions.size(); i++)
    {
      //@记录region的中心点坐标
      data[count] = (t_float)(regions.at(i).bbox_.x+regions.at(i).bbox_.width/2)/img.cols;
      data[count+1] = (t_float)(regions.at(i).bbox_.y+regions.at(i).bbox_.height/2)/img.rows;
      switch (f)
        {
        //记录region的各项属性
        case 0://important
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
        case 5://important
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
          //平均颜色值
          data[count+2] = (t_float)regions.at(i).color_mean_.at(0)/255;
          data[count+3] = (t_float)regions.at(i).color_mean_.at(1)/255;
          data[count+4] = (t_float)regions.at(i).color_mean_.at(2)/255;
          break;
        case 10:
          //边界像素平均颜色值
          data[count+2] = (t_float)regions.at(i).boundary_color_mean_.at(0)/255;
          data[count+3] = (t_float)regions.at(i).boundary_color_mean_.at(1)/255;
          data[count+4] = (t_float)regions.at(i).boundary_color_mean_.at(2)/255;
          break;
        }
      count = count+dim;
    }
}

//@提取regions的某种feature到data[]
void extractFeatureArr(Mat &img, vector<Region>&regions, int &dim, t_float **p_data){
  unsigned int N = regions.size();
  dim = 2;
  t_float* data= *p_data = (t_float*)malloc(dim*N * sizeof(t_float));
  int count = 0;
  for (int i=0; i<regions.size(); i++)
    {

      data[count] = (t_float)regions.at(i).bbox_.y/img.rows;
      data[count+1] = (t_float)(regions[i].stroke_mean_);
      count = count+dim;
    }
}

bool eraseRegionRule1(Region &region){
  int w=region.bbox_.width,h=region.bbox_.height;
  // cout<<"region w:"<<w<<" h:"<<h<<" holes:"<<region.num_holes_<<" ";
  // if((w <=2) || // region.area_<8 || // //	region.area_>0.85*w*h&&w>0.6*h|| // region.num_holes_>=3)
  if ( (w <=2 && h <=3 ) || w>2.5*h )//||
    // region.area_>0.75*w*h&&w>0.5*h)
    {
      // printf("rule1:area:%d w:%d h:%d stroke:%f strokestd:%f num_holes:%d\n",region.area_,w,h,region.stroke_mean_,region.stroke_std_,region.num_holes_);
      // cout<<"erased"<<endl;
      return true;
    }
  // cout<<"remained"<<endl;
  return false;
}

bool eraseRegionRule2(Region &region){
  return false;
  int w=region.bbox_.width,h=region.bbox_.height;
  if ( (region.stroke_std_/region.stroke_mean_ > 0.8)||
       region.num_holes_>3||
       region.area_>0.85*w*h&&w>0.6*h)
    {
      // printf("rule2:area:%d w:%d h:%d stroke:%f strokestd:%f num_holes:%d\n",region.area_,w,h,region.stroke_mean_,region.stroke_std_,region.num_holes_);
      return true;
    }
  return false;
}

/*
 */
// bool eraseRule3(){
/*for (int i=regions.size()-1; i>=0; i--)
	{
	regions[i].inflexion();
// fout<<"inflex:"<<regions[i].inflexion_num_<<endl;
int h=regions[i].bbox_.height,w=regions[i].bbox_.width; int n=regions[i].inflexion_num_; int m=regions[i].perimeter_;
if((float)n/m>0.5&&(float)(m/(h+w))>0.6||n>20)//(float)n/(h+w)>0.4&&n/) 
{ sprintf(str,"per:%d flex:%d h:%d w:%d",m,n,h,w); fout<<str<<endl; regions.erase(regions.begin()+i); }
}*/
/*
// fout<<"max_area:"<<max_area<<" areas:"<<endl;
for (int i=regions.size()-1; i>=0; i--)
{
// if(regions.at(i).area_<max_area/100){
// regions.erase(regions.begin()+i);
// }
}
*/



inline float perimeterR(Region &s){
  float r=(float)s.area_/s.perimeter_*2;
  return s.perimeter_/r;
}
inline float perimeterH(Region &s){
  // cout<<s.perimeter_<<":"<<s.bbox_.height<<endl;
  return (float)s.perimeter_/s.bbox_.height;
}

bool isI(Region &s,Region t){//s is dot
  //if(!isDotStroke(s)) return false;
  //if(!isDot(s)) return false;
  float w1=s.bbox_.width,h1=s.bbox_.height;
  float w2=t.bbox_.width,h2=t.bbox_.height;
  // cout<<"is I?"<<endl;
  // cout<<(float)t.area_/s.area_<<":"<<w1/w2<<":"<<w2/w1<<":"<<h2/h1<<":...."<<endl;
  if((float)t.area_/s.area_<1.5) return false;
  // cout<<"isI:"<<w1<<","<<h1<<"   "<<w2<<","<<h2<<endl;
  //stroke width ratio
  if(w1/w2>1.6||w2/w1>10) return false;
  //height ratio
  if(h2/h1<1.5) return false;
  //distance
  if(s.bbox_x1_>t.bbox_x2_ + w2 || s.bbox_x2_ <t.bbox_x1_ - w2
     ||s.bbox_y2_- t.bbox_y1_ > h1/3.0
     ||t.bbox_y1_ -s.bbox_y2_>  h2)
    return false;
  // cout<<"yes. is I"<<endl;
  return true;
}


void genDimVector(const Mat &co_occurrence_matrix,t_float* D){
  int pos = 0;
  for (int i = 0; i<co_occurrence_matrix.rows; i++)
    {
      for (int j = i+1; j<co_occurrence_matrix.cols; j++)
        {
          D[pos] = (t_float)co_occurrence_matrix.at<double>(i, j);
          pos++;
        }
    }
}

//judge for I
bool isDotStroke(Region &s){
  if(s.num_holes_>0) return false;
  if((s.bbox_.width<=3 && s.bbox_.height <=3) || s.area_ <=6 ) return true;
  float w1=s.bbox_.width,h1=s.bbox_.height;
  //shape
  if(w1/h1>3||h1/w1>3) return false;
  //area
  float xl = pow(w1*w1+h1*h1, 0.5);
  if(s.area_<0.4*3.14/4*xl*xl) return false;
  //周长<4tr >=2tr
  // if(perimeterR(s)>13) return false;
  // cout<<"dotstroke:w:"<<w1<<" h:"<<h1<<" area:"<<s.area_<<endl;
  return true;
}

//judge for lower
bool isHorizonStroke(Region &s,RegionClassifier & classifier){
  if(s.num_holes_>0) return false;
  float w1=s.bbox_.width,h1=s.bbox_.height;
  //shape
  if(h1/w1<1.7||s.area_<0.66*w1*h1) {
      // cout<<"not horizontalstroke:w:"<<w1<<" h:"<<h1<<" area:"<<s.area_<<endl;
      return false;
    }
  // if(!classifier(&s)) return false;
  //perimeter
  // cout<<"perimeterH:"<<perimeterH(s)<<endl;
  // if(perimeterH(s)>4) return false;
  // if(s.perimeter_>h1) return false;
  // cout<<"horizontalstroke:w:"<<w1<<" h:"<<h1<<" area:"<<s.area_<<endl;
  return true;
}


//search i and j
void searchIJ(Mat &img,vector<Region> &regions, RegionClassifier &region_boost ){
  vector<Region> is;
  // MCOUT(2)<<"searching i and j connecting.."<<endl;
  //@connect character i
  for(int j=regions.size()-1;j>=0;j--){
      if(isDotStroke(regions.at(j))){
          for (int i=0; i<regions.size(); i++)
            {
              // if(regions.at(i).bbox_.width*2<regions.at(i).bbox_.height)
              if(isHorizonStroke(regions.at(i),region_boost)){
                  // is.push_back(regions[i]);
                  // fout<<"test region as horizontal stroke:";print(regions[i]);fout<<endl;
                  // fout<<"test region as dot:";print(regions[j]);fout<<endl;
                  if(isI(regions.at(j),regions.at(i))){
                      // fout<<"merging .."<<endl;print(regions[i]);fout<<endl;print(regions[j]);fout<<endl;
                      regions[i].iTag_=true;
                      regions[i].merge(&regions[j]);
                      // regions[i].extract_features(lab_img, grey, gradient_magnitude);
                      is.push_back(regions[i]);
                      regions.erase(regions.begin()+j);
                      break;
                    }
                }
            }
        }
    }
}

inline int boxArea(Region &a){
    return a.bbox_.width * a.bbox_.height;
}

bool overlay(Region &a, Region &b){
    int x1 = max(a.bbox_x1_, b.bbox_x1_), x2 = min(a.bbox_x2_, b.bbox_x2_),
            y1= max(a.bbox_y1_, b.bbox_y1_), y2 = min(a.bbox_y2_, b.bbox_y2_);
    float mw = min(a.bbox_.width, b.bbox_.width), mh = min(a.bbox_.height, b.bbox_.height);
    if(x1<x2-mw/3.0 && y1<y2-mh/3.0) return true;
    return false;
//    //assume a.pixels[] and b.pixels[] are sorted before
//    int l = 0, r= 0;
//    int cnt = 0;
//    int m = 0.5*min(a.area_, b.area_);
//    while(l<a.pixels_.size() && r<b.pixels_.size()){
//        if(a.pixels_[l]==b.pixels_[r]){
//            cnt++, l++, r++;
//            if(cnt>m) return true;
//        }
//        else if(a.pixels_[l]>b.pixels_[r]) r++;
//        else l++;
//    }

}

bool contains(Region &a, Region &b){
    Rect &s=a.bbox_, &t=b.bbox_;
    return s.x<=t.x+t.width/2.0 && s.y<=t.y+t.height/2.0 && s.x+s.width>=t.x+t.width/2.0 &&
            s.y+s.height>=t.y+t.height/2.0;
}

inline bool within(Point &p, Rect &r){
    return p.x >=r.x && p.y >= r.y
           && p.x<=r.x+r.width && p.y <=r.y+r.height;
}

void unique(vector<Region>& regions, int mid){
    int n=regions.size();
    vector<Region> tmp;
    for(int i=0; i<n; i++){
        int j=0; for(; j<n; j++){
            if(i>mid&&j>mid || i<=mid&&j<=mid) continue;
            if(contains(regions[j], regions[i])) break;
        }
        if(j==n) tmp.push_back(regions[i]);
    }
    regions = tmp;
}


#endif
