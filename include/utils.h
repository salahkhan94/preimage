#include <bits/stdc++.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <sophus/se3.hpp>

using namespace std;
/// Read BAL dataset from file
class BAProblem {
public:
    /// load bal data from text file
    explicit BAProblem(const std::string path);

    ~BAProblem() {
    }
    /// save results to ply pointcloud
    map<int, vector<pair<int, int>>> getMeasurements();
    vector<Sophus::SE3d> getPoses();
    int get_num_frames();
    int get_num_points();
    

private:
    int num_frames_;
    int num_points_;
    int num_observations_;
    int num_parameters_;
    bool use_quaternions_;

    map<int, vector<pair<int, int>>> measurements_;
    vector<Sophus::SE3d> poses_;
};
