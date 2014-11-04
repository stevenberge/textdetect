#include <iostream>
using namespace std;

int main(){
  char buf[1000];
  int a, b, c;
  int a_=0, b_=0, c_=0;
  while(cin>>buf){
    if(buf[0]==0) break;
    cin>>a>>b>>c;//c is the gt pixel cnt
    a_+=a, b_+=b, c_+=c;
  }
  float r=(float)a_/c_, p=(float)a_/b_;
  cout<<"recall:"<<r<<endl;
  cout<<"precision:"<<p<<endl;
  cout<<"f:"<<(2*r*p)/(r+p)<<endl;
  return 0;
}
