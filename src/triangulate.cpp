#include <iostream>
#include <stdint.h>

#include <unordered_set>

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
//#include "g2o/math_groups/se3quat.h"
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/stuff/sampler.h"
#include "utils.h"
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <opencv2/opencv.hpp>

using namespace Eigen;
using namespace std;
using namespace cv;


Point2f pixel2cam(const Point2d &p, const Mat &K) {
  return Point2f
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}


void triangulate(map<int, std::vector<std::pair<double, double>>> pixel_me, 
                                vector<Sophus::SE3d> poses, const Mat &K) {
    vector<Point2f> pts_1, pts_2, pts_3, pts_4;
    int i1 = 0,i2 = 1,i3 = 1,i4 = 2;

    auto R1 = poses[i1].rotationMatrix();
    auto R2 = poses[i2].rotationMatrix();
    auto R3 = poses[i3].rotationMatrix();
    auto R4 = poses[i4].rotationMatrix();

    auto t1 = poses[i1].translation();
    auto t2 = poses[i2].translation();
    auto t3 = poses[i3].translation();
    auto t4 = poses[i4].translation();

    Mat T1 = (Mat_<float>(3, 4) <<
        R1(0, 0), R1(0, 1), R1(0, 2), t1(0, 0),
        R1(1, 0), R1(1, 1), R1(1, 2), t1(1, 0),
        R1(2, 0), R1(2, 1), R1(2, 2), t1(2, 0));

    cout << "T1 : " << endl;
    cout << T1 <<endl;
    
    Mat T2 = (Mat_<float>(3, 4) <<
        R2(0, 0), R2(0, 1), R2(0, 2), t2(0, 0),
        R2(1, 0), R2(1, 1), R2(1, 2), t2(1, 0),
        R2(2, 0), R2(2, 1), R2(2, 2), t2(2, 0));
    
    cout << "T2 : " << endl;
    cout << T2 <<endl;    
    
    Mat T3 = (Mat_<float>(3, 4) <<
        R3(0, 0), R3(0, 1), R3(0, 2), t3(0, 0),
        R3(1, 0), R3(1, 1), R3(1, 2), t3(1, 0),
        R3(2, 0), R3(2, 1), R3(2, 2), t3(2, 0));
    cout << "T3 : " << endl;
    cout << T3 <<endl;

    Mat T4 = (Mat_<float>(3, 4) <<
        R4(0, 0), R4(0, 1), R4(0, 2), t4(0, 0),
        R4(1, 0), R4(1, 1), R4(1, 2), t4(1, 0),
        R4(2, 0), R4(2, 1), R4(2, 2), t4(2, 0));
    
    cout << "T4 : " << endl;
    cout << T4 <<endl;
    
    for(int i = 0; i<4;i++) {
        cv::Point2d p1(pixel_me[i1][i].first, pixel_me[i1][i].second);
        pts_1.push_back(pixel2cam(p1, K));
        cv::Point2d p2(pixel_me[i2][i].first, pixel_me[i2][i].second);
        pts_2.push_back(pixel2cam(p2, K));
    }
    Mat pts_4d;
    cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

    for(int i = 0; i<4; i++) {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3, 0);
        cout<<"Point : "<<i<<endl;
        cout<<x.at<float>(0,0) << " "<< x.at<float>(1,0) << " "<< x.at<float>(2,0) << " "<<endl;
    }

    for(int i = 0; i<4;i++) {
        cv::Point2d p1(pixel_me[i3][i].first, pixel_me[i3][i].second);
        pts_3.push_back(pixel2cam(p1, K));
        cv::Point2d p2(pixel_me[i4][i].first, pixel_me[i4][i].second);
        pts_4.push_back(pixel2cam(p2, K));
    }

    Mat a;
    cv::triangulatePoints(T3, T4, pts_3, pts_4, a);
    
    for(int i = 0; i<4; i++) {
        Mat x = a.col(i);
        x /= x.at<float>(3, 0);
        cout<<"Point : "<<i<<endl;
        cout<<x.at<float>(0,0) << " "<< x.at<float>(1,0) << " "<< x.at<float>(2,0) << " "<<endl;
    }

}

int main(int argc, const char* argv[]){
    BAProblem ba("/home/tsa/practice/slambook2/preimage/PreImage-taskdata/data");
    cout << "BA Problem created" << endl;
    int num_frames = ba.get_num_frames();
    int num_points = ba.get_num_points();
    auto measurements = ba.getMeasurements();
    auto poses = ba.getPoses();

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
    g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver))
    );
    optimizer.setAlgorithm(solver);
    //Build graph optimization, first set g2o
    
    /*typedef g2o::BlockSolver< g2o::BlockSolverTraits< 6 , 3 >> Block;   // pose dimension is 6, The landmark dimension is 3 
    std::unique_ptr<Block:: LinearSolverType > linearSolver ( new g2o::LinearSolverCSparse<Block::PoseMatrixType>()); // linear equation solver 
    std::unique_ptr<Block> solver_ptr ( new  Block ( std ::move (linearSolver)));      // Matrix block solver 
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( std::move(solver_ptr) );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );
    */
   
    double f = 128.0;
    Vector2d principal_point(128.0, 72.0);
    Mat K = (Mat_<double>(3, 3) << f, 0, 128, 0, f, 72, 0, 0, 1);

    triangulate(measurements, poses, K);
/*    vector<g2o::SE3Quat,
        aligned_allocator<g2o::SE3Quat> > true_poses;
    g2o::CameraParameters * cam_params
        = new g2o::CameraParameters (f, principal_point, 0.);
    cam_params->setId(0);
    
    optimizer.addParameter(cam_params);



    int vertex_id = 0;
    vector<Vector3d> init_points;
    Vector3d point;
    double u,v;
    Sophus::SE3d T;
    for(size_t i =0; i<num_points; ++i) {
        if(i!=5) {
            u = measurements[0][i].first;
            v = measurements[0][i].second;
            T = poses[0];
        }
        else {
            u = measurements[1][i].first;
            v = measurements[1][i].second;
            T = poses[1];
        }
        double z = 1;
        double x = (u - principal_point.x()) * z / f;
        double y = (v - principal_point.y()) * z / f; 
    
        Vector3d point(x,y,z);
        point = T.inverse() * point;
        init_points.push_back(point);
    }
    
    vector<g2o::VertexSE3Expmap *> vertex_pose_intrinsics;
    vector<g2o::VertexSBAPointXYZ *> vertex_points;
    
    for (size_t i=0; i<num_frames; ++i) {
        auto poseg = poses[i];
        g2o::SE3Quat pose(poseg.rotationMatrix(),poseg.translation());
        g2o::VertexSE3Expmap * v_se3
            = new g2o::VertexSE3Expmap();
        v_se3->setId(vertex_id);
        v_se3->setEstimate(pose);
        v_se3->setFixed(true);
        cout << v_se3->estimate() << endl;
        optimizer.addVertex(v_se3);
        vertex_pose_intrinsics.push_back(v_se3);
        vertex_id++;
    }
    
    int point_id=vertex_id;
    int point_num = 0;
    double sum_diff2 = 0;

    unordered_map<int,int> pointid_2_trueid;
    unordered_set<int> inliers;

    for (size_t i=0; i<num_points; ++i){
        g2o::VertexSBAPointXYZ * v_p
            = new g2o::VertexSBAPointXYZ();
        v_p->setId(point_id);
        v_p->setMarginalized(true);
        v_p->setEstimate(init_points.at(i));
        cout << v_p->estimate() << endl;
        optimizer.addVertex(v_p);
        vertex_points.push_back(v_p);
        point_id++;
    }
    int fr_id = 0;
    int pt_id = 0;
    std::cout << "Points and Pose Vertices added" << endl;
    for(size_t i = 1; i<=num_points*num_frames; ++i) {
        if(measurements[fr_id][pt_id].first) {
            g2o::EdgeProjectXYZ2UV * edge
              = new g2o::EdgeProjectXYZ2UV();
            edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(vertex_pose_intrinsics[fr_id]));
            edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(vertex_points[pt_id]));
            edge->setMeasurement(Vector2d(measurements[fr_id][pt_id].first, 
                                    measurements[fr_id][pt_id].second));
            edge->setInformation(Matrix2d::Identity());
            edge->setRobustKernel(new g2o::RobustKernelHuber());
            edge->setParameterId(0,0);
            cout<< edge->measurement() << endl;
            optimizer.addEdge(edge);
        }
        pt_id++;
        if(i%num_points == 0) {
            pt_id = 0;
            fr_id++;
        }
    }

    std::cout<<"Edges added"<<endl;


    optimizer.initializeOptimization();
    std::cout << "Optimizer initialized" <<endl;
    optimizer.optimize(40);
    std::cout << "Optimized" << endl;

    for(int i = 0; i<vertex_points.size(); i++) {
       auto pt = vertex_points[i]->estimate();
       cout << "x : " << pt.x() << " y : " 
            << pt.y() << " z : " << pt.z() << endl;
    }
    */
}