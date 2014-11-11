#include <iostream>
using namespace std;

int main(){
	int a, b, c;
	int a_ = 0, b_ = 0, c_ = 0;
  float r, p;
	char buf[100];
	while(cin >> buf){
		if ( !buf[0] ) break;
		// cout << " add " << buf << endl;
		cin >> a >> b >> c >> r >> p;
		a_ += a, b_ += b, c_ += c;
	}
	cout << a_ << ":"<< b_ << ":" << c_ << endl;
	cout << " recall: " << (float)a_ / b_ << endl;
	cout << " precision: " << (float)a_ / c_ << endl;
	return 0;
}
