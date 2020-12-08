#ifndef OBJECTSLAM_SYSTEM_H
#define OBJECTSLAM_SYSTEM_H

#include "Tracking.h"
#include "LocalMapping.h"
#include "Frame.h"
#include "FrameDrawer.h"
#include "Map.h"
#include "MapDrawer.h"
#include "Viewer.h"

#include <Eigen/Core>

#include <string>
#include <thread>
#include <memory>

class System{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    System(const std::string& strSettingsFile);

    ~System();

    void TrackWithSingleObject(const Eigen::VectorXd& pose, const Eigen::VectorXd& detection,
                               const cv::Mat& im = cv::Mat());

    void TrackWithObjects(const Eigen::VectorXd& pose, const std::vector<Eigen::VectorXd>& detections,
                          const cv::Mat& im = cv::Mat());

    void TrackRGBD(const cv::Mat& img, const cv::Mat& depthmap, const double& timestamp);

    void ShutDown();

private:
    std::unique_ptr<Viewer> mpViewer;
    std::unique_ptr<Tracking> mpTracker;

    std::shared_ptr<FrameDrawer> mpFrameDrawer;
    std::shared_ptr<MapDrawer> mpMapDrawer;

    std::shared_ptr<LocalMapping> mpLocalMapper;
    std::shared_ptr<Map> mpMap;

    std::thread mtLocalMapping;

};

#endif //OBJECTSLAM_SYSTEM_H
