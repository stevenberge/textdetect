#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>
using namespace std;
using namespace cv;
void gaussianKernelGenerator(Mat &result, int besarKernel, double delta);
void gaussianKernelDerivativeGenerator(Mat &resultX, Mat &resultY, int besarKernel, double delta);
void harris(const Mat src, Mat& result, Mat Gx, Mat Gy, Mat Gxy, int thres, double k);
void rgb2gray(const Mat src, Mat &result);
void circleMidpoint(Mat &img, int x0, int y0, int radius, int val);
/*harris*/
void harris(const Mat src, Mat& result, Mat Gx, Mat Gy, Mat Gxy, int thres, double k)
{
  int centerKernelGyGx = Gy.cols / 2;
  int centerKernelGxy = Gxy.cols /2;
  Mat Ix2 = (Mat_<double>(src.rows, src.cols));
  Mat Iy2 = (Mat_<double>(src.rows, src.cols));
  Mat Ixy = (Mat_<double>(src.rows, src.cols));
  Mat IR = (Mat_<double>(src.rows, src.cols));
  result = src.clone();
  double sX;
  double sY;
  int ii,jj;
  for(int i = 0; i < src.cols; ++i){
    for(int j = 0; j < src.rows; ++j){
      sX = 0;
      sY = 0;
      for(int ik = -centerKernelGyGx; ik <= centerKernelGyGx; ++ik ){
        ii = i + ik;
        for(int jk = -centerKernelGyGx; jk <= centerKernelGyGx; ++jk ){
          jj = j + jk;
          if(ii >= 0 && ii < src.cols && jj >= 0 && jj < src.rows){
            sX += src.at<uchar>(jj, ii) * Gx.at<double>(centerKernelGyGx + jk, centerKernelGyGx + ik);
            sY += src.at<uchar>(jj, ii) * Gy.at<double>(centerKernelGyGx + jk, centerKernelGyGx + ik);
          }
        }
      }
      Ix2.at<double>(j, i) = sX * sX;
      Iy2.at<double>(j, i) = sY * sY;
      Ixy.at<double>(j, i) = sX * sY;
    }
  }
  double sX2;
  double sY2;
  double sXY;
  double R;
  for(int i = 0; i < src.cols; ++i){
    for(int j = 0; j < src.rows; ++j){
      sX2 = 0;
      sY2 = 0;
      sXY = 0;
      for(int ik = -centerKernelGxy; ik <= centerKernelGxy; ++ik ){
        ii = i + ik;
        for(int jk = -centerKernelGxy; jk <= centerKernelGxy; ++jk ){
          jj = j + jk;
          if(ii >= 0 && ii < src.cols && jj >= 0 && jj < src.rows){
            sX2 += Ix2.at<double>(jj, ii) * Gxy.at<double>(centerKernelGxy + jk, centerKernelGxy + ik);
            sY2 += Iy2.at<double>(jj, ii) * Gxy.at<double>(centerKernelGxy + jk, centerKernelGxy + ik);
            sXY += Ixy.at<double>(jj, ii) * Gxy.at<double>(centerKernelGxy + jk, centerKernelGxy + ik);
          }
        }
      }
      //H = [Sx2(x, y) Sxy(x, y); Sxy(x, y) Sy2(x, y)];
      //R = det(H) - k * (trace(H) ^ 2);
      R =
        ((sX2 * sY2) - (sXY * sXY)) //det(H)
        -
        pow( (sX2 + sY2),2) //(trace(H) ^ 2)
        *
        k
        ;
      if(R > thres){
        IR.at<double>(j, i) = R;
      }else{
        IR.at<double>(j, i) = 0;
      }
    }
  }
  for(int y = 1; y < IR.rows - 1; ++y){
    for(int x = 6; x < IR.cols - 6; ++x){
      //non-maximal suppression
      if(
          IR.at<double>(y,x) > IR.at<double>(y + 1,x) &&
          IR.at<double>(y,x) > IR.at<double>(y - 1,x) &&
          IR.at<double>(y,x) > IR.at<double>(y, x + 1) &&
          IR.at<double>(y,x) > IR.at<double>(y, x - 1) &&
          IR.at<double>(y,x) > IR.at<double>(y + 1,x + 1) &&
          IR.at<double>(y,x) > IR.at<double>(y + 1,x - 1) &&
          IR.at<double>(y,x) > IR.at<double>(y - 1, x + 1) &&
          IR.at<double>(y,x) > IR.at<double>(y - 1, x - 1)
        )
      {
        circleMidpoint(result, x, y, 5, 255);
        result.at<uchar>(y,x) = 255;
      }
    }
  }
}
void circleMidpoint(Mat &img, int x0, int y0, int radius, int val)
{
  int x = radius, y = 0;
  int radiusError = 1-x;
  while(x > y)
  {
    img.at<uchar>(y+ y0,x + x0) = val;
    img.at<uchar>(x+ y0,y + x0) = val;
    img.at<uchar>(-y+ y0,x + x0) = val;
    img.at<uchar>(-x+ y0,y + x0) = val;
    img.at<uchar>(-y+ y0,-x + x0) = val;
    img.at<uchar>(-x+ y0,-y + x0) = val;
    img.at<uchar>(y+ y0,-x + x0) = val;
    img.at<uchar>(x+ y0,-y + x0) = val;
    y++;
    if(radiusError < 0){
      radiusError += 2 * y + 1;
    }else{
      x--;
      radiusError += 2 * ( y - x + 1);
    }
  }
}
//Gaussian
void gaussianKernelGenerator(Mat &result, int besarKernel, double delta)
{
  int kernelRadius = besarKernel / 2;
  result = Mat_<double>(besarKernel, besarKernel);
  double pengali = 1 / ( 2 * (22 / 7) * delta * delta) ;
  for(int filterX = - kernelRadius; filterX <= kernelRadius; filterX++){
    for(int filterY = - kernelRadius; filterY <= kernelRadius; filterY++){
      result.at<double>(filterY + kernelRadius, filterX + kernelRadius) =
        exp(-( sqrt( pow(filterY, 2) + pow(filterX, 2) ) / ( pow(delta, 2) * 2) ))
        * pengali;
    }
  }
}
//Gaussian
void gaussianKernelDerivativeGenerator(Mat &resultX, Mat &resultY, int besarKernel, double delta)
{
  int kernelRadius = besarKernel / 2;
  resultX = Mat_<double>(besarKernel, besarKernel);
  resultY = Mat_<double>(besarKernel, besarKernel);
  double pengali = -1 / ( 2 * (22 / 7) * pow(delta, 4) ) ;
  for(int filterX = - kernelRadius; filterX <= kernelRadius; filterX++){
    for(int filterY = - kernelRadius; filterY <= kernelRadius; filterY++){
      resultX.at<double>(filterY + kernelRadius, filterX + kernelRadius) =
        exp(-( ( pow(filterX, 2) ) / ( pow(delta, 2) * 2) ))
        * pengali * filterX;
      resultY.at<double>(filterY + kernelRadius, filterX + kernelRadius) =
        exp(-( ( pow(filterY, 2) ) / ( pow(delta, 2) * 2) ))
        * pengali * filterY;
    }
  }
  //cout<< result << endl;
  //cout<< resultY << endl;
}
void rgb2gray(const Mat src, Mat &result)
{
  CV_Assert(src.depth() != sizeof(uchar)); //harus 8 bit
  result = Mat::zeros(src.rows, src.cols, CV_8UC1); //buat matrik 1 chanel
  uchar data;
  if(src.channels() == 3){
    for( int i = 0; i < src.rows; ++i)
      for( int j = 0; j < src.cols; ++j )
      {
        data = (uchar)(((Mat_<Vec3b>) src)(i,j)[0] * 0.0722 + ((Mat_<Vec3b>) src)(i,j)[1] * 0.7152 + ((Mat_<Vec3b>) src)(i,j)[2] * 0.2126);
        result.at<uchar>(i,j) = data;
      }
  }else{
    result = src;
  }
}
int main(int argc, char *argv[])
{
  Mat src = imread(argv[1]);
  rgb2gray(src, src);
  Mat dGx, dGy, Gxy;
  Mat corner;
  gaussianKernelGenerator(Gxy, 7, 3.9);
  cout<<"Gxy :"<<endl<<Gxy<<endl<<endl;
  gaussianKernelDerivativeGenerator(dGx, dGy, 7, 1.3);
  cout<<"dGx :"<<endl<<dGx<<endl<<endl;
  cout<<"dGy :"<<endl<<dGy<<endl<<endl;
  harris(src, corner, dGx, dGy, Gxy, 5000, 0.04);
  namedWindow("asli");
  imshow("asli", src);
  namedWindow("harris");
  imshow("harris", corner);
  waitKey(0);
}
