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

//void mask(Mat&src, Mat&dst, int n){

//    int c = dst.channels();
//    int nl = dst.cols * c;
//    int nr = dst.rows;
//    assert(src.cols==dst.cols&&
//           src.rows==dst.rows&&
//           src.channels()==dst.channels());
//    for (int i=0; i<nr; i+=1){
//        const uchar* srcData=src.ptr<uchar>(i);
//        uchar* dstData=dst.ptr<uchar>(i);
//        for (int j=0; j<nl; j+=1)
//        {
//            int t = dstData[ j ];
//            int s = srcData[ j ];
//            if(t>0) dstData[ j ] = s* n %255;
//        }
//    }

//}


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


//////////////////////////////////////////////////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////////////////////////////////////////////////////
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

///////////////////////////////////////////////////////////////////////////////////////////////////////
void xihua(const Mat &o, Mat &tmp){
    tmp = Mat::zeros(o.size(), CV_8UC1);
    assert(o.channels()==1);
    const int c = o.cols, r = o.rows;
    for(int i=0; i<r; i++){
        int mi = -1, md = -1;
        for(int j=1; j<c-1; j++){
            int pre = *(o.data + i*c + j-1);
            int t =   *(o.data + i*c + j);
            int pst = *(o.data + i*c + j+1);
            //o.at(point(i, j)) = 0;
            if( t>pre && t>pst ){
                // tmp.at(point(i, mi)) = md;
                *(tmp.data + i * c + j) = 255 ; //md;
            }
        }
    }
    //    for(int i=0; i<c; i++){
    //        int mi = -1, md = -1;
    //        for(int j=0; j<r; j++){
    //          int t = *(o.data + j * c + i);
    //            //o.at(point(j, i)) = 0;
    //            if( t < md ){
    //                //tmp.at(point(mi, i)) = md;
    //                *(tmp.data + mi * c + i) = 255; //md;
    //             }
    //            mi = j, md = t;
    //        }
    //    }
}



//////////////////////////////////////////////////////////////////////////////////////////////////////
//http://blog.csdn.net/qinjianganying/article/details/6756575
static int erasetable[256] = {
    0,0,1,1,0,0,1,1,     1,1,0,1,1,1,0,1,
    1,1,0,0,1,1,1,1,     0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,1,1,     1,1,0,1,1,1,0,1,
    1,1,0,0,1,1,1,1,     0,0,0,0,0,0,0,1,
    1,1,0,0,1,1,0,0,     0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,     0,0,0,0,0,0,0,0,
    1,1,0,0,1,1,0,0,     1,1,0,1,1,1,0,1,
    0,0,0,0,0,0,0,0,     0,0,0,0,0,0,0,0,
    0,0,1,1,0,0,1,1,     1,1,0,1,1,1,0,1,
    1,1,0,0,1,1,1,1,     0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,1,1,     1,1,0,1,1,1,0,1,
    1,1,0,0,1,1,1,1,     0,0,0,0,0,0,0,0,
    1,1,0,0,1,1,0,0,     0,0,0,0,0,0,0,0,
    1,1,0,0,1,1,1,1,     0,0,0,0,0,0,0,0,
    1,1,0,0,1,1,0,0,     1,1,0,1,1,1,0,0,
    1,1,0,0,1,1,1,0,     1,1,0,0,1,0,0,0
};//这个表是检测是用来细化黑色点边缘的，若为0则保留，否则删除，如果用来细化白色点边缘的话就取fan

/**********************************************************88
函数名：npow
参数类型：int n
功能：求2的n次方并返回结果
返回值类型：int
***************************************************************/
int npow(int n)
{
    int mul = 1;
    for(int i=0;i<n;i++)
    {
        mul *= 2;
    }
    return mul;
}

/*****************************************************************************
函数名：scantable
参数类型：IplImage *src
功能：扫描像素8领域周围的八个像素值（只检测白色点得周围），像素值为0置1，否则为0，并保存（如果是检测黑色的周围的话就相反），
再根据得到的数，将得到的数看做一个二进制数化为十进制的数，这个值即为查询索引，如果查到的值为0则保留，否则删除
***************************************************************************************/
void scantable(IplImage *src)
{
    assert(src->nChannels==1);
    int scan[8] = {0};
    for(int h=1;h<(src->height-1);h++)
    {
        for(int w=1;w<(src->width-1);w++)
        {
            int index = 0;
            if(*(src->imageData+(h)*src->widthStep+w)!=0)
            {
                if(*(src->imageData+(h-1)*src->widthStep+w-1)==0)
                    scan[0] = 1;
                else
                    scan[0] = 0;
                if(*(src->imageData+(h-1)*src->widthStep+w)==0)
                    scan[1] = 1;
                else
                    scan[1] = 0;
                if(*(src->imageData+(h-1)*src->widthStep+w+1)==0)
                    scan[2] = 1;
                else
                    scan[2] = 0;
                if(*(src->imageData+(h)*src->widthStep+w-1)==0)
                    scan[3] = 1;
                else
                    scan[3] = 0;
                if(*(src->imageData+(h)*src->widthStep+w+1)==0)
                    scan[4] = 1;
                else
                    scan[4] = 0;
                if(*(src->imageData+(h+1)*src->widthStep+w-1)==0)
                    scan[5] = 1;
                else
                    scan[5] = 0;
                if(*(src->imageData+(h+1)*src->widthStep+w)==0)
                    scan[6] = 1;
                else
                    scan[6] = 0;
                if(*(src->imageData+(h+1)*src->widthStep+w+1)==0)
                    scan[7] = 1;
                else
                    scan[7] = 0;


                for(int i=0;i<8;i++)
                {
                    index += scan[i]*npow(i);
                }
                //  printf("%d\n" ,index);
                if(erasetable[index] == 1)
                {
                    //printf("%d\n" , erasetable[index]);
                    *(src->imageData+h*src->widthStep+w) = 0;
                }
            }

        }
    }
}

void mask(Mat &src, Mat &dst, int n){
    int c=src.cols, r=src.rows;
    for(int i=0; i<r; i++){
        for(int j=0; j<c; j++){
            if(*(dst.data+i*c+j)!=0 && *(src.data+i*c+j)!=0){
                cout<<"find mask"<<endl;
                *(dst.data+i*c+j)=255;//*(src.data+i*c+j)*n%255;
            }else{
                *(dst.data+i*c+j)=0;
            }
        }
    }
}



//////////////////////////////////////////////////////////////////////////////////////////////////
//http://www.csdn123.com/html/itweb/20130917/124285_124289_124308.htm
int  func_nc8(int *b)
//端点的连通性检测
{
    int n_odd[4] = { 1, 3, 5, 7 };  //四邻域
    int i, j, sum, d[10];

    for (i = 0; i <= 9; i++) {
        j = i;
        if (i == 9) j = 1;
        if (abs(*(b + j)) == 1)
        {
            d[i] = 1;
        }
        else
        {
            d[i] = 0;
        }
    }
    sum = 0;
    for (i = 0; i < 4; i++)
    {
        j = n_odd[i];
        sum = sum + d[j] - d[j] * d[j + 1] * d[j + 2];
    }
    return (sum);
}

void cvHilditchThin(cv::Mat& src, cv::Mat& dst)
{
    if(src.type()!=CV_8UC1)
    {
        printf("只能处理二值或灰度图像\n");
        return;
    }
    //非原地操作时候，copy src到dst
    if(dst.data!=src.data)
    {
        src.copyTo(dst);
    }

    //8邻域的偏移量
    int offset[9][2] = {{0,0},{1,0},{1,-1},{0,-1},{-1,-1},
                        {-1,0},{-1,1},{0,1},{1,1} };
    //四邻域的偏移量
    int n_odd[4] = { 1, 3, 5, 7 };
    int px, py;
    int b[9];                      //3*3格子的灰度信息
    int condition[6];              //1-6个条件是否满足
    int counter;                   //移去像素的数量
    int i, x, y, copy, sum;

    uchar* img;
    int width, height;
    width = dst.cols;
    height = dst.rows;
    img = dst.data;
    int step = dst.step ;
    do
    {

        counter = 0;

        for (y = 0; y < height; y++)
        {

            for (x = 0; x < width; x++)
            {

                //前面标记为删除的像素，我们置其相应邻域值为-1
                for (i = 0; i < 9; i++)
                {
                    b[i] = 0;
                    px = x + offset[i][0];
                    py = y + offset[i][1];
                    if (px >= 0 && px < width &&    py >= 0 && py <height)
                    {
                        // printf("%d\n", img[py*step+px]);
                        if (img[py*step+px] == 255)
                        {
                            b[i] = 1;
                        }
                        else if (img[py*step+px]  == 128)
                        {
                            b[i] = -1;
                        }
                    }
                }
                for (i = 0; i < 6; i++)
                {
                    condition[i] = 0;
                }

                //条件1，是前景点
                if (b[0] == 1) condition[0] = 1;

                //条件2，是边界点
                sum = 0;
                for (i = 0; i < 4; i++)
                {
                    sum = sum + 1 - abs(b[n_odd[i]]);
                }
                if (sum >= 1) condition[1] = 1;

                //条件3， 端点不能删除
                sum = 0;
                for (i = 1; i <= 8; i++)
                {
                    sum = sum + abs(b[i]);
                }
                if (sum >= 2) condition[2] = 1;

                //条件4， 孤立点不能删除
                sum = 0;
                for (i = 1; i <= 8; i++)
                {
                    if (b[i] == 1) sum++;
                }
                if (sum >= 1) condition[3] = 1;

                //条件5， 连通性检测
                if (func_nc8(b) == 1) condition[4] = 1;

                //条件6，宽度为2的骨架只能删除1边
                sum = 0;
                for (i = 1; i <= 8; i++)
                {
                    if (b[i] != -1)
                    {
                        sum++;
                    } else
                    {
                        copy = b[i];
                        b[i] = 0;
                        if (func_nc8(b) == 1) sum++;
                        b[i] = copy;
                    }
                }
                if (sum == 8) condition[5] = 1;

                if (condition[0] && condition[1] && condition[2] &&condition[3] && condition[4] && condition[5])
                {
                    img[y*step+x] = 128; //可以删除，置位GRAY，GRAY是删除标记，但该信息对后面像素的判断有用
                    counter++;
                    //printf("----------------------------------------------\n");
                    //PrintMat(dst);
                }
            }
        }

        if (counter != 0)
        {
            for (y = 0; y < height; y++)
            {
                for (x = 0; x < width; x++)
                {
                    if (img[y*step+x] == 128)
                        img[y*step+x] = 0;

                }
            }
        }

    }while (counter != 0);

}
