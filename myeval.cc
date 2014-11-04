#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>
using namespace std;
using namespace cv;

int main(int argc, char **argv){
  char *of=argv[1], *gt=argv[2];
  Mat a=imread(of), b=imread(gt);
  cvtColor(a, a, CV_BGR2GRAY);
  cvtColor(b, b, CV_BGR2GRAY);
  // cout<<a.channels()<<":"<<b.channels()<<endl;
  Mat c = a & b;
  cout<<of<<endl;
  cout<<countNonZero(c)<<" "<<countNonZero(a)<<" "<<countNonZero(b)<<endl;
  return 0;
}
