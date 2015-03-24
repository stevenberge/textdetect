#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
void maximumValue(Mat src , int radius, Mat& dst)
{
  for(int row = 0; row< src.rows; row++)
  {
    for (int col = 0; col< src.cols; col++)
    {
      float maximum = 0;
      for (int win_x = -radius; win_x <= radius; win_x++)
      {
        int newRow = win_x + row;
        if(newRow >= 0 && newRow < src.rows)
        {
          for (int win_y = -radius; win_y<= radius; win_y++)
          {
            int newCol = win_y + col;
            if(newCol >=0 && newCol < src.cols)
            {
              //printf("%f ",src.at<float>(newRow,newCol));
              if (maximum < src.at<float>(newRow,newCol))
              {
                maximum = src.at<float>(newRow,newCol);
              }
            }
          }
        }
      }
      dst.at<float>(row,col) = maximum;
    }
  }
}
void calculateKernel(Mat src,float** kernel,Mat dst, int winSize)
{
  for (int row = 0; row < src.rows; row++)
  {
    for (int col = 0 ; col< src.cols; col++)
    {
      int sum = 0;
      for (int win_x = -winSize; win_x<=winSize ;win_x++)
      {
        int newRow = row+win_x;
        if(newRow>=0 &&newRow<src.rows)
        {
          for (int win_y = -winSize; win_y<=winSize; win_y++)
          {
            int newCol = col+win_y;
            if ( newCol >=0 && newCol<src.cols)
            {
              sum += src.at<uchar>(newRow,newCol)*kernel[win_x+winSize][win_y+winSize];
            }
          }
        }
      }
      dst.at<float>(row,col) = sum;
    }
  }
}
void cons(Mat src1, Mat src2, Mat& dst, float** gaussianKernel, int winSize)
{
  for (int row = 0; row < src1.rows; row++)
  {
    for (int col = 0 ; col< src1.cols; col++)
    {
      float sum = 0;
      for (int win_x = -winSize; win_x<=winSize ;win_x++)
      {
        int newRow = row+win_x;
        if(newRow>=0 &&newRow<src1.rows)
        {
          for (int win_y = -winSize; win_y<=winSize; win_y++)
          {
            int newCol = col+win_y;
            if ( newCol >=0 && newCol<src1.cols)
            {
              sum += src1.at<float>(newRow,newCol)*src2.at<float>(newRow,newCol)*gaussianKernel[win_x+winSize][win_y+winSize];
            }
          }
        }
      }
      dst.at<float>(row,col) = sum;
    }
  }
}
int gcorner(Mat &src)
{
  Mat dst, cdst;
  namedWindow("hello1");
  imshow("hello1",src);
  int width = src.cols;
  int height = src.rows;
  src.copyTo(dst);
  cvtColor(src,src,COLOR_BGR2GRAY);
  Mat Ix, Iy, Ixx, Iyy, Ixy, Cim,Mx;
  Ix.create(Size(src.cols,src.rows),CV_32FC1 );
  Iy.create(Size(src.cols,src.rows),CV_32FC1);
  Ixx.create(Size(src.cols,src.rows),CV_32FC1);
  Iyy.create(Size(src.cols,src.rows),CV_32FC1);
  Ixy.create(Size(src.cols,src.rows),CV_32FC1);
  Cim.create(Size(src.cols,src.rows),CV_32FC1);
  Mx.create(Size(src.cols,src.rows),CV_32FC1);
  //float kernelX[3][3] = {{1,0,-1},{1,0,-1},{1,0,-1}};
  //float kernelY[3][3] = {{1,1,1},{0,0,0},{-1,-1,-1}};
  float **kernelX,**kernelY;
  kernelX = new float*[3];
  kernelY = new float*[3];
  for (int i = 0;i< 3;i++)
  {
    kernelX[i] = new float[3];
    kernelY[i] = new float[3];
    for (int j = 0 ;j< 3;j++)
    {
      kernelX[i][j] = 1-j;
      kernelY[i][j] = 1-i;
    }
  }
  calculateKernel(src,kernelX,Ix, 1);
  calculateKernel(src,kernelY,Iy, 1);
  //float gaussianKernel[][6] = {{0.0003,0.0023,0.0062,0.0062,0.0023,0.0003},{0.0023,0.0168,0.0458,0.0458,0.0168,0.0023},{0.0062,0.0458,0.1244,0.1244,0.0458,0.0062},{ 0.0062,0.0458,0.1244,0.1244,0.0458,0.0062},{0.0023,0.0168,0.0458,0.0458,0.0168,0.0023},{0.0003,0.0023,0.0062,0.0062,0.0023,0.0003}};
  float ** gaussianKernel;
  gaussianKernel = new float *[7];
  for (int i = 0 ;i<7 ;i++)
  {
    gaussianKernel[i] = new float[7];
  }
  float arrSum = 0;
  float sigma = 1;
  int win_size = 3;
  for (int i = -win_size ; i <= win_size; i++)
  {
    for (int j = -win_size ; j<= win_size; j++)
    {
      int r = i*i + j*j;
      float k = -r/(2*sigma*sigma);
      gaussianKernel[i+win_size][j+win_size] = exp(k)/sqrt(2*3.1415926*sigma*sigma);
      arrSum += gaussianKernel[i+win_size][j+win_size];
      //printf("%f ",gaussianKernel[i+win_size][j+win_size]);
    }
    //printf("\n");
  }
  for (int i = -win_size ; i <= win_size; i++)
  {
    for (int j = -win_size ; j<= win_size; j++)
    {
      gaussianKernel[i+win_size][j+win_size] /= arrSum;
      //printf("%f ",gaussianKernel[i+win_size][j+win_size]);
    }
    //printf("\n");
  }
  cons(Ix, Ix, Ixx, gaussianKernel,3);
  cons(Iy, Iy, Iyy, gaussianKernel,3);
  cons(Ix, Iy, Ixy, gaussianKernel,3);
  for (int row = 0; row < src.rows; row++)
  {
    for (int col = 0 ; col< src.cols; col++)
    {
      Cim.at<float>(row,col) = ( Ixx.at<float>(row,col)*
          Iyy.at<float>(row,col) - Ixy.at<float>(row,col)*
          Ixy.at<float>(row,col) ) / ( Ixx.at<float>(row,col)
          + Iyy.at<float>(row,col) + 1e-4 );
      //printf("%6.2f ",Cim.at<float>(row,col));
    }
    //printf("\n");
  }
  int radius = 2;
  int thresh = 1000;
  int sze = 2*radius +1;
  maximumValue(Cim, radius, Mx);
  int cnt = 0;
  for (int i = 0 ;i< Cim.rows ;i++)
  {
    for (int j = 0; j< Cim.cols; j++)
    {
      if (Cim.at<float>(i,j) == Mx.at<float>(i,j) && Cim.at<float>(i,j)> thresh)
      {
        cnt ++;
        // circle(dst,Point(j,i),2,Scalar(0,0,255),3);
      }
    }
  }
  //sze = 2*radius+1;,,,, % Size of mask.
  // mx = ordfilt2(cim,sze^2,ones(sze)); % Grey-scale dilate. maximum filter
  // cim = (cim==mx)&(cim>thresh);, % Find maxima.
  // [r,c] = find(cim);,,,, % Find row,col coords.
  //cout<<Mx<<endl;
  return cnt;
}
