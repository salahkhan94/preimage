#include <iostream>
#include <opencv2/opencv.hpp>
// #include "extra.h" // used in opencv2
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    if (argc != 3) {
        cout << "usage: feature_extraction img1 img2" << endl;
        return 1;
    }
    //-- Read image
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    //cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
    //cv::Ptr<Feature2D> f2d = xfeatures2d::SURF::create();
    //cv::Ptr<Feature2D> f2d = ORB::create();
    // you get the picture, i hope..
    cv::Ptr<Feature2D> f2d = xfeatures2d::SURF::create();
    //-- Step 1: Detect the keypoints:
    std::vector<KeyPoint> keypoints_1, keypoints_2;    
    f2d->detect( img_1, keypoints_1 );
    f2d->detect( img_2, keypoints_2 );

    //-- Step 2: Calculate descriptors (feature vectors)    
    Mat descriptors_1, descriptors_2;    
    f2d->compute( img_1, keypoints_1, descriptors_1 );
    f2d->compute( img_2, keypoints_2, descriptors_2 );

    //-- Step 3: Matching descriptor vectors using BFMatcher :
    BFMatcher matcher;
    std::vector< DMatch > matches;
    Mat output;
    matcher.match( descriptors_1, descriptors_2, matches );
    auto min_max = minmax_element(matches.begin(), matches.end(),
                                [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //When the distance between the descriptors is greater than twice the minimum distance, the match is considered to be 
    //incorrect. But sometimes the minimum distance will be very small, and an empirical value of 30 is set as the lower limit.
    cout<<descriptors_1.rows<<endl;
    cout<<matches.size()<<endl;
    std::vector<DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (matches[i].distance <= max(2 * min_dist, 30.0)) {
        good_matches.push_back(matches[i]);
        }
    }
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, output);
    imshow("all matches", output);
    waitKey(0);
    return 0;

}
