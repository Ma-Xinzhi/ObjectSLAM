#include <gtest/gtest.h>
#include "InitializeQuadric.h"

// TODO 这里的椭球初始化有很大的问题，在根据矩阵求解的过程中，涉及到求解特征值和特征向量
// Eigen中将特征值按照升序进行了排列，每个特征值对应一列特征向量
// 然而该过程中涉及到顺序的问题，实际的数值不是一定按照升序排列的，所以求取的仅仅是数值，并不知道具体数值属于哪个轴总共会出现6种情况
// 而顺序的问题不会影响到最后通过结果重新生成的结果，因为特征值、特征向量的排列问题不会影响特征值分解结果

TEST(InitializeQuadric, BuildQuadricFromQDual){
    Eigen::Matrix4d defaultQuadric = (Eigen::Matrix4d() <<  1.0, 0.0, 0.0, 0.0,
                                                            0.0, 1.0, 0.0, 0.0,
                                                            0.0, 0.0, 1.0, 0.0,
                                                            0.0, 0.0, 0.0, -1.0).finished();
    Eigen::AngleAxisd r_vec(M_PI/4, Eigen::Vector3d(0,0,1));
    Eigen::Matrix3d R = r_vec.toRotationMatrix();
    Eigen::Vector3d t, scale;
    t << -1.897, 1.254, 1.139;
    g2o::SE3Quat pose(R, t);
//    scale << 0.9, 0.8, 0.7;
    scale << 1.6, 1.4, 2.0;
    Eigen::Matrix4d identity = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d scaleQuadric = identity;
    scaleQuadric(0,0) = pow(scale[0], 2);
    scaleQuadric(1,1) = pow(scale[1], 2);
    scaleQuadric(2,2) = pow(scale[2], 2);
    scaleQuadric(3,3) = -1;
    Eigen::Matrix4d Twc = pose.to_homogeneous_matrix();
    Eigen::Matrix4d constrainedQuadric = Twc * scaleQuadric * Twc.transpose();

    LOG(INFO) << "ConstrainedQuadric: " << std::endl
              << constrainedQuadric << std::endl
              << "Determinant: " << constrainedQuadric.determinant() << std::endl;

    InitializeQuadric init(640, 480);
//    std::shared_ptr<g2o::Quadric> Q1_ptr = init.BuildQuadricFromQDual(defaultQuadric);
    std::shared_ptr<g2o::Quadric> Q2_ptr = init.BuildQuadricFromQDual(constrainedQuadric);

//    Eigen::Matrix4d actual_1 = Q1_ptr->toSymMatrix();
    Eigen::Matrix4d actual_2 = Q2_ptr->toSymMatrix();

//    LOG(INFO) << "ActualDefaultQuadric: " << std::endl
//              << actual_1 << std::endl;
    LOG(INFO) << "ActualConstrainedQuadric: " << std::endl
              << actual_2 << std::endl;

//    double diff1 = (actual_1 - defaultQuadric).array().abs().maxCoeff();
    double diff2 = (actual_2 - constrainedQuadric).array().abs().maxCoeff();

//    ASSERT_LT(diff1, 0.1);
    ASSERT_LT(diff2, 0.1);
}

int main(int argc, char** argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
//    Eigen::AngleAxisd r_vec(M_PI/4, Eigen::Vector3d(0,0,1));
//    Eigen::Matrix3d R = r_vec.toRotationMatrix();
//    Eigen::Vector3d t, scale;
//    t << 1, 1, 1;
//    g2o::SE3Quat pose(R, t);
//    scale << 0.9, 0.8, 0.7;
//    scale << 0.7, 0.8, 0.9;
//    scale << 0.8, 0.7, 0.9;
//    Eigen::Matrix4d identity = Eigen::Matrix4d::Identity();
//    Eigen::Matrix4d scaleQuadric = identity;
//    scaleQuadric(0,0) = pow(scale[0], 2);
//    scaleQuadric(1,1) = pow(scale[1], 2);
//    scaleQuadric(2,2) = pow(scale[2], 2);
//    scaleQuadric(3,3) = -1;
//    Eigen::Matrix4d Twc = pose.to_homogeneous_matrix();
//    Eigen::Matrix4d constrainedQuadric = Twc * scaleQuadric * Twc.transpose();
//
//    LOG(INFO) << "ConstrainedQuadric: " << std::endl
//              << constrainedQuadric << std::endl
//              << "Determinant: " << constrainedQuadric.determinant() << std::endl;
//
//    InitializeQuadric init(640, 480);
//    std::shared_ptr<g2o::Quadric> Q2_ptr = init.BuildQuadricFromQDual(constrainedQuadric);
//
//    Eigen::Matrix4d actual = Q2_ptr->toSymMatrix();
//
//    LOG(INFO) << "ActualConstrainedQuadric: " << std::endl
//              << actual << std::endl;
//    return 0;
}