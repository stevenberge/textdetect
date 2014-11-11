#ifndef REGION_H
#define REGION_H

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <queue>
#include <stdint.h>

/// A Maximally Stable Extremal Region.
class Region
{
public:
  bool iTag_=false;// if is character i
  bool isolate=false;
  int inflexion_num_;
  int convex_holl;
  int level_; ///< Level at which the region is processed.
  int pixel_; ///< Index of the initial pixel (y * width + x).
  int area_; ///< Area of the region (moment zero).
  double moments_[5]; ///< First and second moments of the region (x, y, x^2, xy, y^2).
  double variation_; ///< MSER variation.

  /// Axis oriented bounding box of the region
  int bbox_x1_;
  int bbox_y1_;
  int bbox_x2_;
  int bbox_y2_;

  /// Constructor.
  /// @param[in] level Level at which the region is processed.
  /// @param[in] pixel Index of the initial pixel (y * width + x).
  Region(int level = 256, int pixel = 0);

  /// Fills an Extremal Region (ER) by region growing from the Index of the initial pixel(pixel_).
  /// @param[in] grey_img Grey level image
  void er_fill(cv::Mat& _grey_img);

  std::vector<int> pixels_; ///< list pf all pixels indexes (y * width + x) of the region

  /// Extract_features.
  /// @param[in] lab_img L*a*b* color image to extract color information
  /// @param[in] grey_img Grey level version of the original image
  /// @param[in] gradient_magnitude of the original image
  void extract_features(cv::Mat& _lab_img, cv::Mat& _grey_img, cv::Mat& _gradient_magnitude);

  cv::Point center_;	///< Center coordinates of the region
  cv::Rect bbox_;		///< Axis aligned bounding box
  cv::RotatedRect rect_;		///< Axis aligned bounding box
  int perimeter_;		///< Perimeter of the region
  int num_holes_;		///< Number of holes of the region
  int holes_area_;	///< Total area filled by all holes of this regions

  float intensity_mean_;	///< mean intensity of the whole region
  float intensity_std_;	///< intensity standard deviation of the whole region
  std::vector<float> color_mean_;	///< mean color (L*a*b*)  of the whole region
  std::vector<float> color_std_;	///< color (L*a*b*) standard deviation of the whole region
  float boundary_intensity_mean_;	///< mean intensity of the boundary of the region
  float boundary_intensity_std_;	///< intensity standard deviation of the boundary of the region
  std::vector<float> boundary_color_mean_; ///< mean color (L*a*b*)  of the boundary of the region
  std::vector<float> boundary_color_std_;	 ///< color (L*a*b*) standard deviation of the boundary of the region
  //@laoxu
  std::vector<cv::Point>  contour_;

  double stroke_mean_;	///< mean stroke of the whole region
  double stroke_std_;	///< stroke standard deviation of the whole region
  double stroke_var_;///variance

  double gradient_mean_;	///< mean gradient magnitude of the whole region
  double gradient_std_;	///< gradient magnitude standard deviation of the whole region

  float classifier_votes_; ///< Votes of the Region_Classifier for this region
  void inflexion();
  void merge(Region * child);

  void grow(cv::Mat &img, int p,  int threshold=20){
      static int dr[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1} };
      assert (img.channels() == 3);
      int w = img.cols, h = img.rows;

      std::queue<cv::Point> que;
      std::vector<int> my_vec;
      uchar *data = img.data;
      bool *visit = new bool[w*h];
      memset(visit, 0, sizeof(bool)*w*h);

      int x = p/w, y=p%w, z = p;
      my_vec.push_back(z), que.push(cv::Point(x, y)), visit[z] = true;

      while(!que.empty()){
          cv::Point p=que.front(); que.pop();
          int x = p.x, y=p.y, z = x*w + y;
          uchar a = data[z*3], b = data[z*3+1], c = data[z*3+2];
          uchar *data = img.data;

          for(int i=0; i<4; i++){
              int x1 = x+dr[i][0], y1 = y+dr[i][1], z1 = x1*w + y1;
              if(x1>=0 && x1<w && y1>=0 && y1<h && !visit[x1*w+y1]){
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

private:
  bool stable_; // Flag indicating if the region is stable
  Region * parent_; // Pointer to the parent region
  Region * child_; // Pointer to the first child
  Region * next_; // Pointer to the next (sister) region

  void accumulate(int x, int y);
  void detect(int delta, int minArea, int maxArea, double maxVariation, double minDiversity,
              std::vector<Region> & regions);
  void process(int delta, int minArea, int maxArea, double maxVariation);
  bool check(double variation, int area) const;
  void save(double minDiversity, std::vector<Region> & regions);

  friend class MSER;
};

#endif
