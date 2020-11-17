#ifndef OBJECTSLAM_INITIALIZEQUADRIC_H
#define OBJECTSLAM_INITIALIZEQUADRIC_H

#include "Quadric.h"
#include "Observation.h"

#include <unordered_map>

#include <Eigen/Core>
#include <glog/logging.h>

class InitializeQuadric{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    InitializeQuadric(int image_height, int image_width);

    std::shared_ptr<g2o::Quadric> BuildQuadric(const Observations& obs, const Eigen::Matrix3d& calib);
    std::shared_ptr<g2o::Quadric> BuildQuadric(const std::vector<g2o::SE3Quat>& poses, const std::vector<Eigen::VectorXd>& detections,
                                               const Eigen::Matrix3d& calib);
    std::shared_ptr<g2o::Quadric> BuildQuadricFromQDual(const Eigen::Matrix4d& QDual);

    bool GetResult() const{ return mbResult; }

private:
    Eigen::MatrixXd GetPlanesHomo(const std::vector<g2o::SE3Quat>& poses, const std::vector<Eigen::VectorXd>& detections,
                                  const Eigen::Matrix3d& calib);
    Eigen::MatrixXd GetVectorFromPlanesHomo(const Eigen::MatrixXd& planes);
    Eigen::Matrix4d GetQDualFromVectors(const Eigen::MatrixXd& planevecs);
    Eigen::MatrixXd GenerateProjectionMatrix(const g2o::SE3Quat& campose, const Eigen::Matrix3d& calib);
    Eigen::MatrixXd FromDetectionToLines(const Eigen::VectorXd& detection);

private:
    int mImageWidth;
    int mImageHeight;
    bool mbResult;
};

#endif //OBJECTSLAM_INITIALIZEQUADRIC_H
