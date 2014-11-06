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
  float cc = countNonZero(c), ca = countNonZero(a), cb = countNonZero(b);
  ca = ca > 0 ? ca : 1;
  cb = cb > 0 ? cb : 1;
  cout << cc << " " << cb << " " << ca <<
    " " << (cc/cb) << " " << (cc/ca) << endl;
  return 0;
}
