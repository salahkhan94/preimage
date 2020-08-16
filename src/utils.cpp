#include "utils.h"

BAProblem::BAProblem(const std::string path):
    num_frames_(10), num_points_(8) {
    std::cout << "BA Problem constructor" << endl;
    num_observations_ = num_frames_* num_points_;
    std::ifstream infile;
    std::string filename;
    filename = path + "/" + "keypoints";
    infile.open(filename);
    int im = 0, val;
    pair<double, double> pixelcoods;
    for(int i = 1; i<=num_observations_; i++) {
        infile >> val >> pixelcoods.first >> pixelcoods.second;
        measurements_[im].push_back(pixelcoods);
        //std::cout << "image id : " << im << " pixel coods : " << pixelcoods.first << " " << pixelcoods.second << endl;
        if(i%num_points_ == 0) im++;
    }
    infile.close();
    double w,x,y,z;

    Eigen::Quaterniond qt(0.5, 0.5, 0.5, 0.5);
    Eigen::Vector3d tt(0,0,0);
    Sophus::SE3d T(qt, tt);

    for(int i = 0; i<num_frames_; i++) {
        filename = path + "/Pose/" + std::to_string(i) + "_2.txt";
        infile.open(filename);
        infile >> w;
        infile >> x;
        infile >> y;
        infile >> z;
        //cout << w << " " << x << " "<< y << " " << z << endl;
        Eigen::Quaterniond quat(w,x,y,z);
        infile >> x;
        infile >> y;
        infile >> z;
        //cout  << x << " " << y << " " << z << endl;
        Eigen::Vector3d t(x, y, z);
        quat =  quat * qt;
        t = -1 * (quat.toRotationMatrix().inverse() * t);
        Sophus::SE3d pose(quat,t);
//        cout<<"quaternion " << i<<endl; 
//        cout<< quat.x() << " " << quat.y() <<" " << quat.z()<< " " << quat.w()<<endl;
//        cout<<"translation " <<i <<endl;
//        cout<< t.x() << " " << t.y() <<" " << t.z()<<endl;
//        cout<<endl;
        poses_.push_back(pose);
        infile.close();
    }

    std::cout << "Header: " << num_frames_
              << " " << num_points_
              << " " << num_observations_ << endl;
}

map<int, vector<pair<double, double>>> BAProblem::getMeasurements() {
    return measurements_;
}
vector<Sophus::SE3d> BAProblem::getPoses () {
    return poses_;
}
int BAProblem::get_num_frames() {
    return num_frames_;
}
int BAProblem::get_num_points() {
    return num_points_;
}
