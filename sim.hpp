//
// Created by d09 on 18-3-26.
//

#ifndef MATCH_DIFF_HPP
#define MATCH_DIFF_HPP

#endif //MATCH_DIFF_HPP

#include "opencv2/opencv.hpp"
#include <opencv2/features2d/features2d.hpp>
#include <highgui.h>
using namespace std;
using namespace cv;

double pixel_sim(Mat targetImage, Mat tempImage);
double orb_sim(Mat targetImage, Mat tempImage);

