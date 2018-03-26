//
// Created by d09 on 18-3-26.
//
#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;

double pixel_sim(Mat targetImage, Mat tempImage)
{
    cvtColor(targetImage,targetImage,CV_BGR2BGRA);
    cvtColor(tempImage,tempImage,CV_BGR2BGRA);

    Mat resultImage;
    absdiff(targetImage,tempImage,resultImage);
    threshold(resultImage,resultImage,5,255.0,CV_THRESH_BINARY);

    double counter = 0;
    for (int i=0; i<resultImage.rows;i++)
    {
        uchar *data = resultImage.ptr<uchar>(i);
        for (int j=0;j<resultImage.cols;j++){
            if (data[j] == 255){
                counter++;
            }
        }

    }
    counter = (1 - counter*10/(targetImage.rows*targetImage.cols));
    return counter;

}

double orb_sim(Mat targetImage, Mat tempImage)
{
    vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    //创建一个ORB类型指针orb
    Ptr<ORB> orb = ORB::create();

    //第一步：检测Oriented FAST角点位置.
    //detect是Feature2D中的方法，orb是子类指针，可以调用
    //看一下detect()方法的原型参数：需要检测的图像，关键点数组，第三个参数为默认值

    orb->detect(targetImage,keypoints_1);
    orb->detect(tempImage,keypoints_2);

    //第二步：根据角点位置计算BRIEF描述子
    //compute是Feature2D中的方法，orb是子类指针，可以调用
    //看一下compute()原型参数：图像，图像的关键点数组，Mat类型的描述子

    orb->compute(targetImage, keypoints_1, descriptors_1);
    orb->compute(tempImage, keypoints_2, descriptors_2);

    //第三步：对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离

    //创建一个匹配点数组，用于承接匹配出的DMatch，其实叫match_points_array更为贴切。matches类型为数组，元素类型为DMatch
    vector<DMatch> matches;

    //创建一个BFMatcher匹配器，BFMatcher类构造函数如下：两个参数都有默认值，但是第一个距离类型下面使用的并不是默认值，而是汉明距离
    //CV_WRAP BFMatcher( int normType=NORM_L2, bool crossCheck=false );
    BFMatcher matcher (NORM_HAMMING);

    //调用matcher的match方法进行匹配,这里用到了描述子，没有用关键点。
    //匹配出来的结果写入上方定义的matches[]数组中
    matcher.match(descriptors_1, descriptors_2, matches);

    //第四步：遍历matches[]数组，找出匹配点的最大距离和最小距离，用于后面的匹配点筛选。
    //这里的距离是上方求出的汉明距离数组，汉明距离表征了两个匹配的相似程度，所以也就找出了最相似和最不相似的两组点之间的距离。
    double min_dist=0, max_dist=0;//定义距离

    for (int i = 0; i < descriptors_1.rows; ++i)//遍历
    {
        double dist = matches[i].distance;
        if(dist<min_dist) min_dist = dist;
        if(dist>max_dist) max_dist = dist;
    }

    std::vector<DMatch> good_matches;
    for (int j = 0; j < descriptors_1.rows; ++j)
    {
        if (matches[j].distance <= max(2*min_dist, 30.0))
            good_matches.push_back(matches[j]);
    }

    double counter = good_matches.size();
    counter = counter / matches.size();

    return counter;

}


