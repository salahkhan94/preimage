#include <iostream>
#include <Eigen/Core>
#include "utils.h"
#include <sophus/se3.hpp>
#include <vector>
#include <ceres/ceres.h>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

class ReprojectionError {
 public:
  ReprojectionError(
        const Eigen::Matrix<double, 3, 4>& projection_matrix,
        const Eigen::Vector2d& feature)
        : projection_matrix_(projection_matrix), feature_(feature) {}

  template <typename T>
  bool operator()(const T* input_point, T* reprojection_error) const {
        Eigen::Map<const Eigen::Matrix<T, 4, 1> > point(input_point);

        // Multiply the point with the projection matrix, then perform homogeneous
        // normalization to obtain the 2D pixel location of the reprojection.

        const Eigen::Matrix<T, 2, 1> reprojected_pixel =  (projection_matrix_ * point).hnormalized();
        // Reprojection error is the distance from the reprojection to the observed
        // feature location.
    
        reprojection_error[0] = feature_[0] - reprojected_pixel[0]; 
        reprojection_error[1] = feature_[1] - reprojected_pixel[1];
        return true;
  }
  static ceres::CostFunction * Create(const Eigen::Matrix<double, 3, 4>& projection_matrix_, 
        const Eigen::Vector2d& feature_) {
            return (new ceres::AutoDiffCostFunction<ReprojectionError, 2,4>
                (new ReprojectionError(projection_matrix_,feature_)));
    }

 private:
    const Eigen::Matrix<double, 3, 4>& projection_matrix_;
    const Eigen::Vector2d& feature_;
};


Eigen::Vector3d Triangulate(std::vector<std::pair
    <Eigen::Matrix<double, 3, 4>,Eigen::Vector2d>> datas) {
        Eigen::Vector4d x;
        x << 0,0,0,1;

        ceres::Problem problem;
        ceres::Solver::Options options;
        ceres::Solver::Summary summary;

        for(auto data:datas) {
            ceres::LossFunction* loss= nullptr;
            problem.AddResidualBlock(ReprojectionError::Create(data.first,data.second),loss,&x[0]);
        }

        ceres::Solve(options,&problem, &summary);
        std::cout << summary.FullReport() << "\n";
        return x.hnormalized();
} 


int main(int argc, char** argv) {
    BAProblem ba("/home/tsa/practice/slambook2/preimage/PreImage-taskdata/data");
    cout << "BA Problem created" << endl;
    unsigned int num_frames = ba.get_num_frames();
    unsigned int num_points = ba.get_num_points();
    auto measurements = ba.getMeasurements();
    auto poses = ba.getPoses();
    double f = 128.0;

    Eigen::Matrix<double, 3, 3> K;
    K << f, 0, 128, 0, f, 72, 0, 0, 1;
    num_points = 1;
    num_frames = 9;
    std::vector<std::pair
    <Eigen::Matrix<double, 3, 4>,Eigen::Vector2d>> datas;
    for(size_t i = 0; i<num_points; ++i) {
        for(size_t j = 0; j<num_frames; j++) {
            Eigen::Vector2d pt2d(measurements[j][i].first, measurements[j][i].second);
            if(pt2d[0] && pt2d[1]) {
                auto R = poses[j].rotationMatrix();
                auto t = poses[j].translation();
                Eigen::Matrix<double, 3, 4> T;
                T<< R(0, 0), R(0, 1), R(0, 2), t(0, 0),
                    R(1, 0), R(1, 1), R(1, 2), t(1, 0),
                    R(2, 0), R(2, 1), R(2, 2), t(2, 0);
            
                auto P = K * T;

                Eigen::Vector4d x;
                x << -0.332204,-6.99158,2.23257,1;
                auto proj3d = (P * x);
                
                cout << " True 2d point : "  << endl;
                cout << pt2d << endl;
                cout << " Projected 2d : " << endl;
                cout << proj3d.hnormalized() <<endl;
                cout << " Raw projected " << endl;
                cout << proj3d << endl;
                cout << endl;
                
                std::pair<Eigen::Matrix<double, 3, 4>,Eigen::Vector2d> d;
                d.first = P;
                d.second = pt2d;
                datas.push_back(d);
            }
        }

        auto point = Triangulate(datas);
        std::cout << point <<endl;    
    }
}