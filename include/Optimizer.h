#ifndef OBJECTSLAM_OPTIMIZER_H
#define OBJECTSLAM_OPTIMIZER_H

#include "MapPoint.h"
#include "Map.h"
#include "KeyFrame.h"
#include "Frame.h"

#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"

class Optimizer{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    static int PoseOptimization(std::shared_ptr<Frame> pFrame);

    static g2o::SE3Quat toSE3Quat(const Eigen::Matrix4d& pose);
};

#endif //OBJECTSLAM_OPTIMIZER_H
