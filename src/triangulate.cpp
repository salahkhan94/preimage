#include <iostream>
#include <stdint.h>

#include <unordered_set>

#include "utils.h"
#include <sophus/se3.hpp>
#include <Eigen/Core>

#include <opencv2/opencv.hpp>
#include <opencv2/sfm/triangulation.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/matx.hpp>

using namespace Eigen;
using namespace std;
using namespace cv;



void triangulate(map<int, std::vector<std::pair<double, double>>> pixel_me, 
                                vector<Sophus::SE3d> poses, const Mat &K, unsigned int nframes, unsigned int npoints) {
    vector<Point2d> pts2d;
    vector<cv::Vec3d> pts3d;    
    vector<Mat>  pts2darr;
    vector<vector<Matx<double, 3, 4> >>  projmarr;
    cv::Vec3d point;
    vector<Matx<double, 3, 4> > projm;

    for(size_t i = 0; i<npoints; ++i) {
        for(size_t j = 0; j<nframes; j++) {
            cv::Point2d pt2d(pixel_me[j][i].first, pixel_me[j][i].second);
            if(pt2d.x && pt2d.y) {
                auto R = poses[j].rotationMatrix();
                auto t = poses[j].translation();
                    Mat T = (Mat_<double>(3, 4) <<
                    R(0, 0), R(0, 1), R(0, 2), t(0, 0),
                    R(1, 0), R(1, 1), R(1, 2), t(1, 0),
                    R(2, 0), R(2, 1), R(2, 2), t(2, 0));
                
                Mat P = K * T;
                
                projm.push_back(P);
                pts2d.push_back(pt2d);
            }
        }
        Mat temp = Mat(pts2d.size(),2,CV_64F,pts2d.data());
        Mat pt2d_mat;
        transpose(temp, pt2d_mat);
        cout<< "pt2d cols : "<<pt2d_mat.cols <<endl;
        
        cv::sfm::triangulateNViews(pt2d_mat,projm, point);
        pts3d.push_back(point);
        //projmarr.push_back(projm);
        //pts2darr.push_back(pt2d_mat);
        projm.clear();
        pts2d.clear();
    }

    //cv::Mat points3d;
    //cv::sfm::triangulatePoints(pts2darr, projmarr, points3d);

    for(size_t i =0; i<pts3d.size();i++) {
        cout<< " Point : " << i+1 <<endl;
        cout<<pts3d[i]<<endl;
        cout<<endl;
    }
}

int main(int argc, const char* argv[]){
    BAProblem ba("/home/tsa/practice/slambook2/preimage/PreImage-taskdata/data");
    cout << "BA Problem created" << endl;
    int num_frames = ba.get_num_frames();
    int num_points = ba.get_num_points();
    auto measurements = ba.getMeasurements();
    auto poses = ba.getPoses();

    num_points = min((int)num_points,atoi(argv[1]));
    num_frames = min((int)num_frames,atoi(argv[2]));

    double f = 128.0;
    Vector2d principal_point(128.0, 72.0);
    Mat K = (Mat_<double>(3, 3) << f, 0, 128, 0, f, 72, 0, 0, 1);

    triangulate(measurements, poses, K, num_frames, num_points);
    
}