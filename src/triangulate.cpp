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
using namespace Eigen;
using namespace std;


int main(int argc, const char* argv[]){
    BAProblem ba("/home/tsa/practice/slambook2/preimage/PreImage-taskdata/data");
    int num_frames = ba.get_num_frames();
    int num_points = ba.get_num_points();
    auto measurements = ba.getMeasurements();
    auto poses = ba.getPoses();
    bool ROBUST_KERNEL = true;
    bool STRUCTURE_ONLY = false;
    bool DENSE = true;

    std::cout << "ROBUST_KERNEL: " << ROBUST_KERNEL << endl;
    std::cout << "STRUCTURE_ONLY: " << STRUCTURE_ONLY<< endl;
    std::cout << "DENSE: "<<  DENSE << endl;

      //Build graph optimization, first set g2o
    typedef g2o::BlockSolver< g2o::BlockSolverTraits< 6 , 3 >> Block;   // pose dimension is 6, The landmark dimension is 3 
    std::unique_ptr<Block:: LinearSolverType > linearSolver ( new g2o::LinearSolverCSparse<Block::PoseMatrixType>()); // linear equation solver 
    std::unique_ptr<Block> solver_ptr ( new  Block ( std ::move (linearSolver)));      // Matrix block solver 
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( std::move(solver_ptr) );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );

    double f= 128.0;
    Vector2d principal_point(128.0, 72.0);

    vector<g2o::SE3Quat,
        aligned_allocator<g2o::SE3Quat> > true_poses;
    g2o::CameraParameters * cam_params
        = new g2o::CameraParameters (f, principal_point, 0.);
    cam_params->setId(0);

    if (!optimizer.addParameter(cam_params)) {
        assert(false);
    }

    int vertex_id = 0;
    vector<Vector3d> init_points;
    Vector3d point;
    double u,v;
    Sophus::SE3d T;
    for(size_t i =0; i<num_points; ++i) {
        if(i!=6) {
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
        optimizer.addVertex(v_se3);
        vertex_pose_intrinsics.push_back(v_se3);
        vertex_id++;
    }
    /*
    int point_id=vertex_id;
    int point_num = 0;
    double sum_diff2 = 0;

    std::cout << endl;
    unordered_map<int,int> pointid_2_trueid;
    unordered_set<int> inliers;

    for (size_t i=0; i<num_points; ++i){
        g2o::VertexSBAPointXYZ * v_p
            = new g2o::VertexSBAPointXYZ();
        v_p->setId(point_id);
        v_p->setMarginalized(true);
        v_p->setEstimate(init_points.at(i));
        optimizer.addVertex(v_p);
        vertex_points.push_back(v_p);
    }
    int fr_id = 0;
    int pt_id = 0;
    for(size_t i = 1; i<=num_points*num_frames; ++i) {
        if(measurements[fr_id][pt_id].first) {
            g2o::EdgeProjectXYZ2UV * edge
              = new g2o::EdgeProjectXYZ2UV();
            edge->setVertex(0, vertex_pose_intrinsics[fr_id]);
            edge->setVertex(1, vertex_points[pt_id]);
            edge->setMeasurement(Vector2d(measurements[fr_id][pt_id].first, 
                                    measurements[fr_id][pt_id].second));
            edge->setInformation(Matrix2d::Identity());
            edge->setRobustKernel(new g2o::RobustKernelHuber());
            optimizer.addEdge(edge);
        }
        pt_id++;
        if(i%num_points == 0) {
            pt_id = 0;
            fr_id++;
        }
    }


    optimizer.initializeOptimization();
    optimizer.optimize(40);
    for(int i = 0; i<vertex_points.size(); i++) {
       auto pt = vertex_points[i]->estimate();
       cout << "x : " << pt.x() << " y : " 
            << pt.y() << " z : " << pt.z() << endl;
    }
    */
}