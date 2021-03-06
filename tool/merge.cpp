#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
using namespace std;
using namespace cv;

int main(int argc, char**args){
  cout<<"cmd a b +/-/& c"<<endl;
  cout<<"a#"<<args[1]<<":b#"<<args[2]<<endl;
  char *s=args[1], *t=args[2], *op=args[3], *u=args[4];
  Mat a=imread(s), b=imread(t);

  if(a.channels()==3){
    cvtColor(a, a, CV_BGR2GRAY);
    threshold(a, a, 25, 255, CV_THRESH_BINARY);
  }
  if(b.channels()==3){
    cvtColor(b, b, CV_BGR2GRAY);
    threshold(b, b, 25, 255, CV_THRESH_BINARY);
  }

  Mat c=a.clone();
  if(strcmp(op,"&")==0){
    c=a&b;
  }
  else if(strcmp(op,"+")==0){
    c=a|b;
  }
  else if(strcmp(op,"-")==0){
    c=a-b;
  }
  imwrite(u, c);
  return 0;
}
