#include "extend.h"

void printImg(Mat& img, bool bin){
  int c = img.channels(), nl = img.cols * c, nr = img.rows;
  //cout<<"Mat chanel:"<<c<<" widthstep:"<<nl<<" height:"<<nr<<endl;
  int num=0;
  for (int i=0; i<nr; i+=1){
    const uchar* srcData=img.ptr<uchar>(i);
    for (int j=0; j<nl; j+=1)
    {
      num += srcData[ j ];//Blue通道
    }
  }
  if(num==0) {
      cout<<"black image"<<endl;
      return;
  }
  ///////////////////////
  for (int i=0; i<nr; i+=1){
    const uchar* srcData=img.ptr<uchar>(i);
    for (int j=0; j<nl; j+=c)
    {
      int t = srcData[ j ];//Blue通道
      if(bin) cout<<(t>0?1:0)<<" ";
      else  cout<<t<<" ";
    }
    cout<<endl;
  }
}

void mask(Mat&src, Mat&dst){

  int c = dst.channels();
  int nl = dst.cols * c;
  int nr = dst.rows;
  assert(src.cols==dst.cols&&
      src.rows==dst.rows&&
      src.channels()==dst.channels());
  for (int i=0; i<nr; i+=1){
    const uchar* srcData=src.ptr<uchar>(i);
    uchar* dstData=dst.ptr<uchar>(i);
    for (int j=0; j<nl; j+=1)
    {
      int t = dstData[ j ];//Blue通道
      int s = srcData[ j ];
      if(t>0) dstData[ j ] = s;
    }
  }

}


void printImg(IplImage *out, bool bin){
  uchar* srcData = (uchar*)out->imageData;//数据的起始地址
  int srcStep = out->widthStep;//图像的行长度，即多少个字节
  cout<<"Ipl chanel:"<<out->nChannels<<" widthstep:"<<srcStep<<" width:"<<out->width<<" height:"<<out->height<<endl;
  for (int row = 0; row < out->height; row++)
  {
    for (int col = 0; col < out->width; col++)
    {
      int t=srcData[ row*srcStep + col*out->nChannels + 0];//Blue通道
      if(bin) cout<<(t?1:0)<<" ";
      else cout<<t<<" ";
    }
    cout<<endl;
  }
}


//将 DEPTH_8U型二值图像进行细化  经典的Zhang并行快速细化算法
void thin(const Mat &src, Mat &dst, const int iterations)
{
  const int height =src.rows -1;
  const int width  =src.cols -1;

  //拷贝一个数组给另一个数组
  if(src.data != dst.data)
  {
    src.copyTo(dst);
  }


  int n = 0,i = 0,j = 0;
  Mat tmpImg;
  uchar *pU, *pC, *pD;
  bool isFinished = false;

  for(n=0; n<iterations; n++)
  {
    dst.copyTo(tmpImg); 
    isFinished = false;   //一次 先行后列扫描 开始
    //扫描过程一 开始
    for(i=1; i<height;  i++) 
    {
      pU = tmpImg.ptr<uchar>(i-1);
      pC = tmpImg.ptr<uchar>(i);
      pD = tmpImg.ptr<uchar>(i+1);
      for(int j=1; j<width; j++)
      {
        if(pC[j] > 0)
        {
          int ap=0;
          int p2 = (pU[j] >0);
          int p3 = (pU[j+1] >0);
          if (p2==0 && p3==1)
          {
            ap++;
          }
          int p4 = (pC[j+1] >0);
          if(p3==0 && p4==1)
          {
            ap++;
          }
          int p5 = (pD[j+1] >0);
          if(p4==0 && p5==1)
          {
            ap++;
          }
          int p6 = (pD[j] >0);
          if(p5==0 && p6==1)
          {
            ap++;
          }
          int p7 = (pD[j-1] >0);
          if(p6==0 && p7==1)
          {
            ap++;
          }
          int p8 = (pC[j-1] >0);
          if(p7==0 && p8==1)
          {
            ap++;
          }
          int p9 = (pU[j-1] >0);
          if(p8==0 && p9==1)
          {
            ap++;
          }
          if(p9==0 && p2==1)
          {
            ap++;
          }
          if((p2+p3+p4+p5+p6+p7+p8+p9)>1 && (p2+p3+p4+p5+p6+p7+p8+p9)<7)
          {
            if(ap==1)
            {
              if((p2*p4*p6==0)&&(p4*p6*p8==0))
              {                           
                dst.ptr<uchar>(i)[j]=0;
                isFinished =TRUE;                            
              }

              //   if((p2*p4*p8==0)&&(p2*p6*p8==0))
              //    {                           
              //         dst.ptr<uchar>(i)[j]=0;
              //         isFinished =TRUE;                            
              //    }

            }
          }                    
        }

      } //扫描过程一 结束


      dst.copyTo(tmpImg); 
      //扫描过程二 开始
      for(i=1; i<height;  i++)  //一次 先行后列扫描 开始
      {
        pU = tmpImg.ptr<uchar>(i-1);
        pC = tmpImg.ptr<uchar>(i);
        pD = tmpImg.ptr<uchar>(i+1);
        for(int j=1; j<width; j++)
        {
          if(pC[j] > 0)
          {
            int ap=0;
            int p2 = (pU[j] >0);
            int p3 = (pU[j+1] >0);
            if (p2==0 && p3==1)
            {
              ap++;
            }
            int p4 = (pC[j+1] >0);
            if(p3==0 && p4==1)
            {
              ap++;
            }
            int p5 = (pD[j+1] >0);
            if(p4==0 && p5==1)
            {
              ap++;
            }
            int p6 = (pD[j] >0);
            if(p5==0 && p6==1)
            {
              ap++;
            }
            int p7 = (pD[j-1] >0);
            if(p6==0 && p7==1)
            {
              ap++;
            }
            int p8 = (pC[j-1] >0);
            if(p7==0 && p8==1)
            {
              ap++;
            }
            int p9 = (pU[j-1] >0);
            if(p8==0 && p9==1)
            {
              ap++;
            }
            if(p9==0 && p2==1)
            {
              ap++;
            }
            if((p2+p3+p4+p5+p6+p7+p8+p9)>1 && (p2+p3+p4+p5+p6+p7+p8+p9)<7)
            {
              if(ap==1)
              {
                //   if((p2*p4*p6==0)&&(p4*p6*p8==0))
                //   {                           
                //         dst.ptr<uchar>(i)[j]=0;
                //         isFinished =TRUE;                            
                //    }

                if((p2*p4*p8==0)&&(p2*p6*p8==0))
                {                           
                  dst.ptr<uchar>(i)[j]=0;
                  isFinished =TRUE;                            
                }

              }
            }                    
          }

        }

      } //一次 先行后列扫描完成          
      //如果在扫描过程中没有删除点，则提前退出
      if(isFinished == false)
      {
        break; 
      }
    }

  }
}

void strenth(Mat &mat, Mat &mergeImg)
{
  //用来存储各通道图片的向量
  vector<Mat> splitBGR(mat.channels());
  //分割通道，存储到splitBGR中
  split(mat,splitBGR);
  //对各个通道分别进行直方图均衡化
  for(int i=0; i<mat.channels(); i++)
    equalizeHist(splitBGR[i],splitBGR[i]);
  //合并通道
  merge(splitBGR,mergeImg);
}

bool isContourPoint(const int x, const int y, const Mat& bwImg)
{
  bool p[10] ={0}; //记录当前点的8邻域的有无情况

  const uchar *pU= bwImg.ptr(y-1, x);  //上一行
  const uchar *pC= bwImg.ptr(y, x);    //当前行
  const uchar *pD= bwImg.ptr(y+1, x);  //下一行


   p[2]=*(pU) ? true:false;
   p[3]=*(pU+1) ? true:false;
   p[4]=*(pC+1) ? true:false;
   p[5]=*(pD+1) ? true:false;
   p[6]=*(pD) ? true:false;
   p[7]=*(pD-1) ? true:false;
   p[8]=*(pC-1) ? true:false;
   p[9]=*(pU-1) ? true:false;


  int Np=0;//邻域不为零节点总数
  int Tp=0;//邻域节点由0变成1的次数
  for (int i=2; i<10; i++)
  {
  Np += p[i];
  int k= (i<9) ? (i+1) : 2;
  if ( p[k] -p[i]>0)
  {
  Tp++;
  }
  }
  int p246= p[2] && p[4] && p[6];
  int p468= p[4] && p[6] && p[8];

  int p24= p[2] && !p[3] && p[4] && !p[5] && !p[6] && !p[7] && !p[8] && !p[9];
  int p46= !p[2] && !p[3] && p[4] && !p[5] && p[6] && !p[7] && !p[8] && !p[9];
  int p68= !p[2] && !p[3] && !p[4] && !p[5] && p[6] && !p[7] && p[8] && !p[9];
  int p82= p[2] && !p[3] && !p[4] && !p[5] && !p[6] && !p[7] && p[8] && !p[9];

  int p782= p[2] && !p[3] && !p[4] && !p[5] && !p[6] && p[7] && p[8] && !p[9];
  int p924= p[2] && !p[3] && p[4] && !p[5] && !p[6] && !p[7] && !p[8] && p[9];
  int p346= !p[2] && p[3] && p[4] && !p[5] && p[6] && !p[7] && !p[8] && !p[9];
  int p568= !p[2] && !p[3] && !p[4] && p[5] && p[6] && !p[7] && p[8] && !p[9];

  int p689= !p[2] && !p[3] && !p[4] && !p[5] && p[6] && !p[7] && p[8] && p[9];
  int p823= p[2] && p[3] && !p[4] && !p[5] && !p[6] && !p[7] && p[8] && !p[9];
  int p245= p[2] && !p[3] && p[4] && p[5] && !p[6] && !p[7] && !p[8] && !p[9];
  int p467= !p[2] && !p[3] && p[4] && !p[5] && p[6] && p[7] && !p[8] && !p[9];

  int p2468= p24 || p46 || p68 || p82;
  int p3333= p782 || p924 || p346 || p568 || p689 || p823 || p245 || p467;

  //判定条件第一个由数字图像处理上得到，由于结果不够满意，又加上两个条件
  return ( !p246 && !p468 && (Np<7) && (Np>1) && (Tp==1) ) || p2468 || p3333; 
}


//细化二值图像，得到单像素连通域
void thin2(Mat& bwImg)
{

  const int imgRows=bwImg.rows -1;
  const int imgCols=bwImg.cols -1;

  int Remove_Num;
  int i, j;
  do //循环调用，直至没有可以去掉的点
  {
    Remove_Num=0;
    for (j = 1; j < imgRows; j++)
    {
      for(i = 1; i < imgCols; i++)
      {

        if ( *bwImg.ptr(j, i) && isContourPoint( i, j, bwImg))//符合条件，去掉
        {
          *bwImg.ptr(j, i)=0;
          Remove_Num++;
        }  //if
      }  //for
    }  //for
  } while( Remove_Num);
  cout<<"thin2 end"<<endl;
}


float widthVar(Region &r){

    return 0;
}
