#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>
using namespace std;
using namespace cv;

int main(int argc, char **argv){
  char *of=argv[0], *gt=argv[1];
  Mat a=imread(of), b=imread(gt);
  Mat c = a & b;
  cout<<of<<endl;
  cout<<countNonZero(c)<<" "<<countNonZero(a)<<" "<<countNonZero(b)<<endl;
  return 0;
}
