#include <iostream>
#include <Eigen/Core>
#include "utils.h"
#include <sophus/se3.hpp>
#include <vector>
#include <ceres/ceres.h>
#include <cmath>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

class ReprojectionError {
 using mat =  Eigen::Matrix<double, 3, 8>;
 public:
    ReprojectionError(
            const Eigen::Matrix<double, 3, 4>& projection_matrix,
            const Eigen::Matrix<double, 2, 8>& feature)
            : projection_matrix_(projection_matrix), feature_(feature),
             coordinates3d_((mat() <<   0.0,  1.0,  1.0,  0.0,  0.0,  1.0,  1.0,  0.0,
                    0.0,  0.0,  1.0,  1.0,  0.0,  0.0,  1.0,  1.0,
                    -1.0, -1.0, -1.0, -1.0,  0.0,  0.0,  0.0,  0.0).finished()) {}

    template <typename T>
    bool operator()(const T* cube_params, T* reprojection_error) const {

           // if(cube_params[3] < T(5)) return false;

            const Eigen::Matrix<T, 3, 8> coods = getCoods(cube_params);
            const Eigen::Matrix<double, 3, 3> R(projection_matrix_.block<3,3>(0,0));
            const Eigen::Matrix<double, 3, 1> t(projection_matrix_.block<3,1>(0,3));
            // Multiply the point with the projection matrix, then perform homogeneous
            // normalization to obtain the 2D pixel location of the reprojection.
            
            const Eigen::Matrix<T, 2, 8> reprojected_pixel =  ( (R * coods).colwise() + t ).colwise().hnormalized();
            // Reprojection error is the distance from the reprojection to the observed
            // feature location.

            for(int i=0; i<8;i++) {
                if(feature_.col(i)[0] && feature_.col(i)[1]) {
                    reprojection_error[2*i] = feature_.col(i)[0] - reprojected_pixel.col(i)[0]; 
                    reprojection_error[(2*i)+1] = feature_.col(i)[1] - reprojected_pixel.col(i)[1];
                }
                else {
                    reprojection_error[2*i] = T(0); 
                    reprojection_error[(2*i)+1] = T(0);
                }
            }
            return true;
    }

    template <typename T>
    Eigen::Matrix<T, 3, 8> getCoods(const T* cube_params) const {
        Eigen::Matrix<T, 3, 8> coods = coordinates3d_ * cube_params[3];
        Eigen::Matrix<T, 3, 1> v;
        v << T(0), T(0), T(1);
        Eigen::AngleAxis<T> anax(cube_params[2],v);
        Eigen::Matrix<T, 3, 3> R = anax.toRotationMatrix();
        Eigen::Matrix<T, 3, 1> t;
        t << cube_params[0], cube_params[1], T(0);
        //coods = (R * coods).colwise() + t;
        return ((R * coods).colwise() + t);
    }

    static ceres::CostFunction * Create(const Eigen::Matrix<double, 3, 4>& projection_matrix_, 
            const Eigen::Matrix<double, 2, 8>& feature_) {
                return (new ceres::AutoDiffCostFunction<ReprojectionError, 16,4>
                    (new ReprojectionError(projection_matrix_,feature_)));
    } 

 private:
    const Eigen::Matrix<double, 3, 4>& projection_matrix_;
    const Eigen::Matrix<double, 2, 8>& feature_;
    const Eigen::Matrix<double, 3, 8> coordinates3d_;    

};

Eigen::Matrix<double, 3, 8> getCoods(const double* cube_params) {
    Eigen::Matrix<double, 3, 8> coordinates3d_;
    coordinates3d_ <<   0.0,  1.0,  1.0,  0.0,  0.0,  1.0,  1.0,  0.0,
                        0.0,  0.0,  1.0,  1.0,  0.0,  0.0,  1.0,  1.0,
                       -1.0, -1.0, -1.0, -1.0,  0.0,  0.0,  0.0,  0.0;

    Eigen::Matrix<double, 3, 8> coods = coordinates3d_ * cube_params[3];
    Eigen::Matrix3d R = Eigen::AngleAxisd(cube_params[2], Eigen::Vector3d(0, 0, 1)).toRotationMatrix();
    Eigen::Vector3d t(cube_params[0], cube_params[1], 0);
    //coods = (R * coods).colwise() + t;
    return ((R * coods).colwise() + t);
}

Eigen::Matrix<double, 3, 8> Triangulate(std::vector<std::pair
    <Eigen::Matrix<double, 3, 4>,Eigen::Matrix<double, 2, 8>>> datas) {
        double cube_params[4] = {1, 1, 0, 10};
        ceres::Problem problem;
        ceres::Solver::Options options;
        ceres::LossFunction* loss= nullptr;//new ceres::CauchyLoss(0.5);
        ceres::Solver::Summary summary;

        for(auto& data:datas) {
            problem.AddResidualBlock(ReprojectionError::Create(data.first,data.second), loss, cube_params);
        }
        ceres::Solve(options,&problem, &summary);
        std::cout << summary.FullReport() << "\n";
        for(auto &n : cube_params) {
            std::cout << n << " ";
        }
        std::cout << std::endl;
        return getCoods(cube_params);
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

    cout << K << endl;

    num_points = 8;
    num_frames = 10;
    
    std::vector<std::pair
        <Eigen::Matrix<double, 3, 4>,Eigen::Matrix<double, 2, 8>>> datas;
            
    for(size_t j = 0; j<num_frames; j++) {
        auto R = poses[j].rotationMatrix();
        auto t = poses[j].translation();
        Eigen::Matrix<double, 3, 4> T;
        T<< R(0, 0), R(0, 1), R(0, 2), t(0, 0),
            R(1, 0), R(1, 1), R(1, 2), t(1, 0),
            R(2, 0), R(2, 1), R(2, 2), t(2, 0);

        auto P = K * T;
        std::pair<Eigen::Matrix<double, 3, 4>,Eigen::Matrix<double, 2, 8>> d;
        Eigen::Matrix<double, 2, 8> features;
        for(size_t i = 0; i<num_points; ++i) {
            Eigen::Vector2d pt2d(measurements[j][i].first, measurements[j][i].second);
            features.col(i) = pt2d;
        }
        d.first = P;
        d.second = features;
        datas.push_back(d);
    }
    cout << "data created" << endl;
    auto points = Triangulate(datas);
    std::cout << points <<endl;
}