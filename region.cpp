#include "region.h"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <limits>
#include "extend.h"

using namespace std;
using namespace cv;

Region::Region(int level, int pixel) : level_(level), pixel_(pixel), area_(0),
    variation_(numeric_limits<double>::infinity()), stable_(false), parent_(0), child_(0), next_(0), bbox_x1_(10000), bbox_y1_(10000), bbox_x2_(0), bbox_y2_(0)
{
    fill_n(moments_, 5, 0.0);
}
struct PointCMP{
    bool operator()(const Point& a, const Point &b){
        return a.x<b.x || a.x==b.x && a.y<b.y;
    }
}pcmp;

inline bool isOnEdge(set<int> &area,int s,int width){
  int x=s/width,y=s%width;
  int t;
  t=(x-1)*width+y;
  if(area.find(t)==area.end()) return true;

  t=x*width+(y+1);
  if(area.find(t)==area.end()) return true;

  t=(x+1)*width+y;
  if(area.find(t)==area.end()) return true;

  t=x*width+(y-1);
  if(area.find(t)==area.end()) return true;
  return false;
}
//@laoxu
void Region::inflexion(int cols){
    set<int> area;
    Region & r = *this;
    for (int p=0; p<pixels_.size(); p++){
        int j= r.pixels_.at(p);
        area.insert(j);
      }
    // cout<<"region "<<i<<" size "<<regions->at(i).pixels_.size()<<endl;
    for (int p=0; p<r.pixels_.size(); p++)
    {
        int j=r.pixels_.at(p);
        if(isOnEdge(area,j,cols)){
            contour_.push_back(Point(j/cols, j%cols));
        }
    }

    vector<Point>& contour_poly=contour_;
    //sort(contour_.begin(), contour_.end(), pcmp);
    vector<Point> ss;
//    for(int i = 0; i<contour_.size(); i++){
//        Point & p = contour_.at(i);
//        cout<<" ("<<p.x<<","<<p.y<<") ";
//    }cout<<endl;
    bool was_convex=false;
    int num_inflexion_points=0;
    // cout<<"inflex with "<<contour_.size()<<" points"<<endl;
    for (int p = 0 ; p<(int)contour_poly.size(); p++)
    {
        int p_prev = p-1;
        int p_next = p+1;
        if (p_prev == -1)
            p_prev = contour_poly.size()-1;
        if (p_next == (int)contour_poly.size())
            p_next = 0;

        double angle_next = atan2((contour_poly[p_next].y-contour_poly[p].y),(contour_poly[p_next].x-contour_poly[p].x));
        double angle_prev = atan2((contour_poly[p_prev].y-contour_poly[p].y),(contour_poly[p_prev].x-contour_poly[p].x));
        if ( angle_next < 0 )
            angle_next = 2.*CV_PI + angle_next;

        double angle = (angle_next - angle_prev);
        if (angle > 2.*CV_PI)
            angle = angle - 2.*CV_PI;
        else if (angle < 0)
            angle = 2.*CV_PI + std::abs(angle);

        if (p>0)
        {
            if ( ((angle > CV_PI)&&(!was_convex)) || ((angle < CV_PI)&&(was_convex)) )
                num_inflexion_points++;
        }
        was_convex = (angle > CV_PI);
    }
    inflexion_num_=num_inflexion_points;
}
//inline void Region::accumulate(int x, int y)
void Region::accumulate(int x, int y)
{
    ++area_;
    moments_[0] += x;
    moments_[1] += y;
    moments_[2] += x * x;
    moments_[3] += x * y;
    moments_[4] += y * y;

    bbox_x1_ = min(bbox_x1_, x);
    bbox_y1_ = min(bbox_y1_, y);
    bbox_x2_ = max(bbox_x2_, x);
    bbox_y2_ = max(bbox_y2_, y);
}

void Region::merge(Region * child)
{
    assert(!child->parent_);
    assert(!child->next_);

    // Add the moments together
    area_ += child->area_;
    moments_[0] += child->moments_[0];
    moments_[1] += child->moments_[1];
    moments_[2] += child->moments_[2];
    moments_[3] += child->moments_[3];
    moments_[4] += child->moments_[4];

    // Rebuild bounding box
    bbox_x1_ = min(bbox_x1_, child->bbox_x1_);
    bbox_y1_ = min(bbox_y1_, child->bbox_y1_);
    bbox_x2_ = max(bbox_x2_, child->bbox_x2_);
    bbox_y2_ = max(bbox_y2_, child->bbox_y2_);

    child->next_ = child_;
    child_ = child;
    child->parent_ = this;

    //@add by xuzhigang
    bbox_=Rect(bbox_x1_,bbox_y1_,bbox_x2_-bbox_x1_,bbox_y2_-bbox_y1_);
}

void Region::process(int delta, int minArea, int maxArea, double maxVariation)
{
    // Find the last parent with level not higher than level + delta
    const Region * parent = this;

    while (parent->parent_ && (parent->parent_->level_ <= (level_ + delta)))
        parent = parent->parent_;

    // Calculate variation
    variation_ = static_cast<double>(parent->area_ - area_) / area_;

    // Whether or not the region *could* be stable
    const bool stable = (!parent_ || (variation_ <= parent_->variation_)) &&
            (area_ >= minArea) && (area_ <= maxArea) && (variation_ <= maxVariation);

    // Process all the children
    for (Region * child = child_; child; child = child->next_) {
        child->process(delta, minArea, maxArea, maxVariation);

        if (stable && (variation_ < child->variation_))
            stable_ = true;
    }

    // The region can be stable even without any children
    if (!child_ && stable)
        stable_ = true;
}

bool Region::check(double variation, int area) const
{
    if (area_ <= area)
        return true;

    if (stable_ && (variation_ < variation))
        return false;

    for (Region * child = child_; child; child = child->next_)
        if (!child->check(variation, area))
            return false;

    return true;
}

void Region::save(double minDiversity, vector<Region> & regions)
{
    if (stable_) {
        const int minParentArea = area_ / (1.0 - minDiversity) + 0.5;

        const Region * parent = this;

        while (parent->parent_ && (parent->parent_->area_ < minParentArea)) {
            parent = parent->parent_;

            if (parent->stable_ && (parent->variation_ <= variation_)) {
                stable_ = false;
                break;
            }
        }

        if (stable_) {
            const int maxChildArea = area_ * (1.0 - minDiversity) + 0.5;

            if (!check(variation_, maxChildArea))
                stable_ = false;
        }

        if (stable_) {
            regions.push_back(*this);
            regions.back().parent_ = 0;
            regions.back().child_ = 0;
            regions.back().next_ = 0;
        }
    }

    for (Region * child = child_; child; child = child->next_)
        child->save(minDiversity, regions);
}

void Region::detect(int delta, int minArea, int maxArea, double maxVariation,
                    double minDiversity, vector<Region> & regions)
{
    process(delta, minArea, maxArea, maxVariation);
    save(minDiversity, regions);
}

/* function:    er_fill is borowed from vlfeat-0.9.14/toolbox/mser/vl_erfill.c
 ** description: Extremal Regions filling
 ** author:      Andrea Vedaldi
 **/

/*
   Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
   All rights reserved.

   The function is part of the VLFeat library and is made available under
   the terms of the BSD license (see the COPYING file).
   */
void Region::er_fill(Mat& _grey_img)
{
    const uint8_t *src = (uint8_t*)_grey_img.data;


    double er = pixel_;
    int ndims = 2;
    int dims [2];
    dims[0] = _grey_img.cols;
    dims[1] = _grey_img.rows;
    int last = 0 ;
    int last_expanded = 0 ;
    uint8_t value = 0 ;

    double const * er_pt ;

    int*   subs_pt ;       /* N-dimensional subscript                 */
    int*   nsubs_pt ;      /* diff-subscript to point to neigh.       */
    int*   strides_pt ;    /* strides to move in image array          */
    uint8_t*  visited_pt ;    /* flag                                    */
    int*   members_pt ;    /* region members                          */
    bool   invert = false;

    /* get dimensions */
    int nel   = dims[0]*dims[1];
    uint8_t *I_pt  = (uint8_t *)src;

    /* allocate stuff */
    subs_pt    = (int*) malloc( sizeof(int)      * ndims ) ;
    nsubs_pt   = (int*) malloc( sizeof(int)      * ndims ) ;
    strides_pt = (int*) malloc( sizeof(int)      * ndims ) ;
    visited_pt = (uint8_t*)malloc( sizeof(uint8_t)     * nel   ) ;
    members_pt = (int*) malloc( sizeof(int)      * nel   ) ;

    er_pt = &er;

    /* compute strides to move into the N-dimensional image array */
    strides_pt [0] = 1 ;
    int k;
    for(k = 1 ; k < ndims ; ++k) {
        strides_pt [k] = strides_pt [k-1] * dims [k-1] ;
    }

    //fprintf(stderr,"strides_pt %d %d \n",strides_pt [0],strides_pt [1]);

    /* load first pixel */
    memset(visited_pt, 0, sizeof(uint8_t) * nel) ;
    {
        int idx = (int) *er_pt ;
        if (idx < 0) {
            idx = -idx;
            invert = true ;
        }
        if( idx < 0 || idx > nel+1 ) {
            fprintf(stderr,"ER=%d out of range [1,%d]",idx,nel) ;
            return;
        }
        members_pt [last++] = idx ;
    }
    value = I_pt[ members_pt[0] ]  ;

    /* -----------------------------------------------------------------
   *                                                       Fill region
   * -------------------------------------------------------------- */
    while(last_expanded < last) {

        /* pop next node xi */
        int index = members_pt[last_expanded++] ;

        /* convert index into a subscript sub; also initialize nsubs
       to (-1,-1,...,-1) */
        {
            int temp = index ;
            for(k = ndims-1 ; k >=0 ; --k) {
                nsubs_pt [k] = -1 ;
                subs_pt  [k] = temp / strides_pt [k] ;
                temp         = temp % strides_pt [k] ;
            }
        }

        /* process neighbors of xi */
        while(true) {
            int good = true ;
            int nindex = 0 ;

            /* compute NSUBS+SUB, the correspoinding neighbor index NINDEX
         and check that the pixel is within image boundaries. */
            for(k = 0 ; k < ndims && good ; ++k) {
                int temp = nsubs_pt [k] + subs_pt [k] ;
                good &= 0 <= temp && temp < (signed) dims[k] ;
                nindex += temp * strides_pt [k] ;
            }

            /* process neighbor
         1 - the pixel is within image boundaries;
         2 - the pixel is indeed different from the current node
         (this happens when nsub=(0,0,...,0));
         3 - the pixel has value not greather than val
         is a pixel older than xi
         4 - the pixel has not been visited yet
         */
            if(good
                    && nindex != index
                    && ((!invert && I_pt [nindex] <= value) ||
                        ( invert && I_pt [nindex] >= value))
                    && ! visited_pt [nindex] ) {

                //fprintf(stderr,"nvalue %d  value %d",(int)(I_pt [nindex]),(int)(I_pt [index]));
                //fprintf(stderr,"           index %d\n",index);
                //fprintf(stderr,"neightbour index %d\n",nindex);

                /* mark as visited */
                visited_pt [nindex] = 1 ;

                /* add to list */
                members_pt [last++] = nindex ;
            }

            /* move to next neighbor */
            k = 0 ;
            while(++ nsubs_pt [k] > 1) {
                nsubs_pt [k++] = -1 ;
                if(k == ndims) goto done_all_neighbors ;
            }
        } /* next neighbor */
done_all_neighbors : ;
    } /* goto pop next member */

    /*
   * Save results
   */
    {
        for (int i = 0 ; i < last ; ++i) {
            pixels_.push_back(members_pt[i]);
            //fprintf(stderr,"	pixel inserted %d: %d\n",i,members_pt[i]);
        }
    }
    //@xzg
    sort(pixels_.begin(),pixels_.end());

    free( members_pt ) ;
    free( visited_pt ) ;
    free( strides_pt ) ;
    free( nsubs_pt   ) ;
    free( subs_pt    ) ;

    return;
}
void Region::grow(cv::Mat &img,  int threshold){
    static int dr[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, -1}, {1, 1}, {-1, -1}, {-1, 1}};
    assert (img.channels() == 3);
    int w = img.cols, h = img.rows;

    std::queue<cv::Point> que;
    std::vector<int> my_vec;
    uchar *data = img.data;
    bool *visit = new bool[w*h];
    memset(visit, 0, sizeof(bool)*w*h);

    for(int i=0; i<this->pixels_.size(); i+=5){
        int p = this->pixels_[i];
        int x = p%w, y=p/w, z = p;
        my_vec.push_back(z), que.push(cv::Point(x, y)), visit[z] = true;
    }

    while(!que.empty()){
        cv::Point p=que.front(); que.pop();
        int x = p.x, y=p.y, z = y*w + x;
        uchar a = data[z*3], b = data[z*3+1], c = data[z*3+2];
        uchar *data = img.data;

        for(int i=0; i<8; i++){
            int x1 = x+dr[i][0], y1 = y+dr[i][1], z1 = y1*w + x1;
            if(x1>=0 && x1<w && y1>=0 && y1<h && !visit[y1*w+x1]){
                uchar a1 = data[z1*3], b1 = data[z1*3+1], c1 = data[z1*3+2];
                if(abs(a1-a)+abs(b1-b)+abs(c1-c) < threshold){
                    my_vec.push_back(z1), que.push(cv::Point(x1, y1));
                    visit[z1] = true;
                }
            }
        }
    }
    this->pixels_ = my_vec;
    delete []visit;
}

void Region::extract_features(Mat& _lab_img, Mat& _grey_img, Mat& _gradient_magnitude)
{

    bbox_x2_++;
    bbox_y2_++;

    center_.x = bbox_x2_-bbox_x1_ / 2;
    center_.y = bbox_y2_-bbox_y1_ / 2;

    bbox_ = cvRect(bbox_x1_,bbox_y1_,bbox_x2_-bbox_x1_,bbox_y2_-bbox_y1_);

    Mat canvas = Mat::zeros(_lab_img.size(),CV_8UC1);
    uchar* rsptr = (uchar*)canvas.data;
    for (int p=0; p<pixels_.size(); p++)
    {
        rsptr[pixels_[p]] = 255;
    }

    int x = bbox_x1_ - min (10, bbox_x1_);
    int y = bbox_y1_ - min (10, bbox_y1_);
    int width  = bbox_x2_ - x + min(10, _lab_img.cols-bbox_x2_);
    int height = bbox_y2_ - y + min(10, _lab_img.rows-bbox_y2_);

    CvRect rect = cvRect(x, y, width, height);
    Mat bw = canvas(rect);

    Scalar mean,std;
    meanStdDev(_grey_img(rect),mean,std,bw);
    intensity_mean_ = mean[0];
    intensity_std_  = std[0];

    meanStdDev(_lab_img(rect),mean,std,bw);
    color_mean_.push_back(mean[0]);
    color_mean_.push_back(mean[1]);
    color_mean_.push_back(mean[2]);
    color_std_.push_back(std[0]);
    color_std_.push_back(std[1]);
    color_std_.push_back(std[2]);

    int i=random()%1000; char buf[100];

    Mat tmp; //, tmp1=bw.clone(), tmp2_=bw.clone(); //, tmp1 = bw.clone();
    distanceTransform(bw, tmp, CV_DIST_L1, 3); //L1 gives distance in round integers while L2 floats
    distanceTransform(bw, tmp, CV_DIST_L2, 3);
   // xihua(tmp, xh);


//    Mat xh=bw.clone();
//    IplImage s=xh;
//    scantable(&s);
//    sprintf(buf, "out1/%d.xihua.png", i);
//    imwrite(buf, xh);

//    thin(bw, xh, 10);
//    sprintf(buf, "out1/%d.xihua1.png", i);
//    imwrite(buf, xh);

//    mask(tmp, xh, 40);
//    sprintf(buf, "out1/%d.xihuamsk.png", i);
//    imwrite(buf, xh);
//    cout<<i<<":"<<endl;
   // cout<<"original:"<<endl;
   // printImg(bw, false);
   // cout<<"ds:"<<endl;
   // printImg(tmp, false);
    meanStdDev(tmp,mean,std,bw);

    stroke_mean_ = mean[0];
    stroke_std_  = std[0];

//    cout<<"stroke_mean:"<<stroke_mean_<<" stroke_std:"<<stroke_std_<<endl;

    Mat element = getStructuringElement( MORPH_RECT, Size(5, 5), Point(2, 2) );
    dilate(bw, tmp, element);
    absdiff(tmp, bw, tmp);

    meanStdDev(_grey_img(rect), mean, std, tmp);
    boundary_intensity_mean_ = mean[0];
    boundary_intensity_std_  = std[0];

    meanStdDev(_lab_img(rect), mean, std, tmp);
    boundary_color_mean_.push_back(mean[0]);
    boundary_color_mean_.push_back(mean[1]);
    boundary_color_mean_.push_back(mean[2]);
    boundary_color_std_.push_back(std[0]);
    boundary_color_std_.push_back(std[1]);
    boundary_color_std_.push_back(std[2]);

    Mat tmp2;
    dilate(bw, tmp, element);
    erode(bw, tmp2, element);
    absdiff(tmp, tmp2, tmp);

    meanStdDev(_gradient_magnitude(rect), mean, std, tmp);
    gradient_mean_ = mean[0];
    gradient_std_  = std[0];


    copyMakeBorder(bw, bw, 5, 5, 5, 5, BORDER_CONSTANT, Scalar(0));

    num_holes_ = 0;
    holes_area_ = 0;
    vector<vector<Point> > contours0;
    vector<Vec4i> hierarchy;
    findContours( bw, contours0, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    for (int k=0; k<hierarchy.size();k++)
    {
        //TODO check this thresholds, are they coherent? is there a faster way?
        //num of holes
        if ((hierarchy[k][3]==0)&&
                ((((float)contourArea(Mat(contours0.at(k)))/contourArea(Mat(contours0.at(0))))>0.01)||(contourArea(Mat(contours0.at(k)))>31)))
        {
            num_holes_++;
            holes_area_ += (int)contourArea(Mat(contours0.at(k)));
        }
    }
    perimeter_ = (int)contours0.at(0).size();
    //@..
    approxPolyDP( Mat(contours0[0]), contour_, max(bbox_.width,bbox_.height)/25, true );


    // if(perimeter_<10){
    // cout<<"------------"<<endl;
    // for(int i=0;i<contours0.at(0).size();i++){
    // Point p=contours0.at(0)[i];
    // cout<<p.x<<","<<p.y<<endl;
    // }
    // }
    for (int k=0; k<hierarchy.size();k++)
        rect_ = minAreaRect(contours0.at(0));
}
