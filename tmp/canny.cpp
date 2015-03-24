#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv/highgui.h"
#include <opencv2/opencv.hpp>
using namespace cv;
int main( int argc, char** argv )
{
	//声明IplImage指针
	IplImage* img = NULL;
	IplImage* cannyImg = NULL;
	char *filename;
	if(argc<2) return 0;
	filename=argv[1];
	img=cvLoadImage(filename,1);
	//载入图像，强制转化为Gray
	if((img = cvLoadImage(filename, 0)) != 0 )
	{
		//为canny边缘图像申请空间
		cannyImg = cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
		//canny边缘检测
		cvCanny(img, cannyImg, 50, 150, 3);
		char str[100];
		sprintf(str,"%s.canny.jpg",argv[1]);
		cvSaveImage(str,cannyImg);
		//创建窗口
		cvNamedWindow("src", 1);
		cvNamedWindow("canny",1);
		//显示图像
		cvShowImage( "src", img );
		cvShowImage( "canny", cannyImg );
		cvWaitKey(0); //等待按键
		//销毁窗口
		cvDestroyWindow( "src" );
		cvDestroyWindow( "canny" );
		
		//释放图像
		cvReleaseImage( &img );
		cvReleaseImage( &cannyImg );
		return 0;
	}
	return -1;
}
