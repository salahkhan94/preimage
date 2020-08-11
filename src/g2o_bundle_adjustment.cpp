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
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/stuff/sampler.h"
#include "utils.h"
#include <sophus/se3.hpp>

using namespace Eigen;
using namespace std;

class Sample {
 public:
  static int uniform(int from, int to) { return static_cast<int>(g2o::Sampler::uniformRand(from, to)); }
};

int main(int argc, const char* argv[]){
  

    BAProblem ba("/home/tsa/practice/slambook2/preimage/PreImage-taskdata/data");
    cout << "BA Problem created" << endl;
    int num_frames = ba.get_num_frames();
    int num_points = ba.get_num_points();
    auto measurements = ba.getMeasurements();
    auto poses = ba.getPoses();

    //num_points = 4;

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    bool DENSE = 0;
    bool STRUCTURE_ONLY = 0;
    bool ROBUST_KERNEL = 0;
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    if (DENSE) {
        linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    } else {
        linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();
    }

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver))
    );
    optimizer.setAlgorithm(solver);

    double focal_length = 128.0;
    Vector2d principal_point(128.0, 72.0);

    vector<g2o::SE3Quat,
        aligned_allocator<g2o::SE3Quat> > true_poses;
    g2o::CameraParameters * cam_params
        = new g2o::CameraParameters (focal_length, principal_point, 0.);
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
        double x = (u - principal_point.x()) * z / focal_length;
        double y = (v - principal_point.y()) * z / focal_length; 
    
        Vector3d point(x,y,z);
        point = T.inverse() * point;
        init_points.push_back(point);
    }

    
    vector<g2o::VertexSE3Expmap *> vertex_poses;
    vector<g2o::VertexSBAPointXYZ *> vertex_points;
    
    for (size_t i=0; i<num_frames; ++i) {
        
        auto poseg = poses[i];
        g2o::SE3Quat pose(poseg.rotationMatrix(),poseg.translation());
        
        cout << "Rotation Matrix :" << endl;
        cout << poseg.rotationMatrix() << endl;
        cout << "Translation vector : " << endl;
        cout << poseg.translation() << endl;
        cout << "Pose : " << endl;
        cout << pose <<endl;

        g2o::VertexSE3Expmap * v_se3
            = new g2o::VertexSE3Expmap();

        v_se3->setId(vertex_id);
        v_se3->setEstimate(pose);
        v_se3->setFixed(false);

        cout << v_se3->estimate() << endl;
        
        optimizer.addVertex(v_se3);
        true_poses.push_back(pose);
        vertex_poses.push_back(v_se3);
        vertex_id++;
    }

    
    // int point_id = vertex_id;
    // int point_num = 0;
    // double sum_diff2 = 0;

    // unordered_map<int,int> pointid_2_trueid;
    // unordered_set<int> inliers;

    // for (size_t i=0; i<num_points; ++i){
    //     g2o::VertexSBAPointXYZ * v_p
    //         = new g2o::VertexSBAPointXYZ();
    //     v_p->setId(point_id);
    //     v_p->setMarginalized(true);
    //     v_p->setEstimate(init_points.at(i));

    //     cout << v_p->estimate() << endl;

    //     optimizer.addVertex(v_p);
    //     vertex_points.push_back(v_p);
    //     point_id++;
    // }

    int point_id=vertex_id;

    cout << endl;
    

    for (size_t i=0; i<num_points; ++i){
        g2o::VertexSBAPointXYZ * v_p
            = new g2o::VertexSBAPointXYZ();
        v_p->setId(point_id);
        v_p->setMarginalized(true);
        v_p->setEstimate(init_points.at(i));

        cout << "Point " << i <<endl;
        cout << v_p->estimate() << endl;
        cout << endl;

        optimizer.addVertex(v_p);
        vertex_points.push_back(v_p);
        point_id++;

        for (size_t j=0; j<true_poses.size(); ++j){
            Vector2d z(measurements[j][i].first, 
                                    measurements[j][i].second);

            if (z[0]>0 && z[1]>0 && z[0]<256 && z[1]<144){
            
                g2o::EdgeProjectXYZ2UV * e
                    = new g2o::EdgeProjectXYZ2UV();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_p));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>
                            (optimizer.vertices().find(j)->second));
                e->setMeasurement(z);
                cout << "Measurement : " << endl;
                cout<< e->measurement() << endl;
                e->information() = Matrix2d::Identity();
                if (ROBUST_KERNEL) {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                }
                e->setParameterId(0, 0);
                optimizer.addEdge(e);
            }
        }
    }
    cout << endl;
    optimizer.initializeOptimization();
    optimizer.setVerbose(true);
    if (STRUCTURE_ONLY){
        g2o::StructureOnlySolver<3> structure_only_ba;
        cout << "Performing structure-only BA:"   << endl;
        g2o::OptimizableGraph::VertexContainer points;
        for (g2o::OptimizableGraph::VertexIDMap::const_iterator it = optimizer.vertices().begin(); it != optimizer.vertices().end(); ++it) {
        g2o::OptimizableGraph::Vertex* v = static_cast<g2o::OptimizableGraph::Vertex*>(it->second);
        if (v->dimension() == 3)
            points.push_back(v);
        }
        structure_only_ba.calc(points, 10);
    }
    //optimizer.save("test.g2o");
    cout << endl;
    cout << "Performing full BA:" << endl;
    optimizer.optimize(50);
    cout << endl;

    for(size_t i=0; i<vertex_points.size();i++) {
        cout << "Point " << i <<endl;
        cout << vertex_points[i]->estimate() <<endl;
    }
}
