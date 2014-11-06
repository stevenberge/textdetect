#include <iostream>
#include <fstream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
using namespace std;
using namespace cv;
void fillRegions(Mat& img, vector<Rect> &rects)
{
  uchar* rsptr = (uchar*)img.data;
  vector<vector<Point> > contours;
  for (int l=0; l<rects.size(); l++)
  {
    Rect &s = rects[l];
    vector<Point> contour;
    contour.push_back(Point(s.x,s.y));
    contour.push_back(Point(s.x+s.width,s.y));
    contour.push_back(Point(s.x+s.width,s.y+s.height));
    contour.push_back(Point(s.x,s.y+s.height));
    contours.push_back(contour);
  }
  drawContours(img, contours, -1, Scalar(0,255,0), 2);//CV_FILLED, 8);
}
int main(int n, char **args){
  cout<<"cmd oriImg markTxt outImg"<<endl;
  char* img = args[1], *txt = args[2], *oImg = args[3];
  ifstream out;
  out.open(txt, ios::in);
  Mat tmp = imread(img);
  vector<Rect> rects;
  char buf[100];
  int x1, y1, x2, y2;
  while(true){
    out>>(buf);
    sscanf(buf, "%d, %d, %d, %d", &x1, &y1, &x2, &y2);
    rects.push_back(Rect(x1, y1, x2-x1, y2-y1));
  }
  fillRegions(tmp, rects);
  imwrite(oImg, tmp);
  return 0;
}
