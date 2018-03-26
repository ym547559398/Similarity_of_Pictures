//
// Created by d09 on 18-3-24.
//

#include "sim.hpp"

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;

int main(int argc, char * argv())
{
    Mat targetImage,tempImage,resultImage;

    targetImage = imread("/home/d09/Videos/frames/save_0.jpg");
    tempImage = imread("/home/d09/Videos/frames/save_4.jpg");
    double count = orb_sim(targetImage,tempImage);
    cout << "The sim of two images is: " << count << endl;

    return 0;

}
