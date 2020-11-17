#ifndef OBJECTSLAM_SYSTEM_H
#define OBJECTSLAM_SYSTEM_H

#include "Track.h"
#include "Frame.h"
#include "FrameDrawer.h"
#include "Map.h"
#include "MapDrawer.h"
#include "Viewer.h"

#include <Eigen/Core>

#include <string>

class System{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    System(const std::string& strSettingsFile);

    void TrackWithSingleObject(const Eigen::VectorXd& pose, const Eigen::VectorXd& detection,
                               const cv::Mat& im = cv::Mat());

    void TrackWithObjects(const Eigen::VectorXd& pose, const std::vector<Eigen::VectorXd>& detections,
                          const cv::Mat& im = cv::Mat());

    void ShutDown();

private:
    std::shared_ptr<Viewer> mpViewer;
    std::shared_ptr<FrameDrawer> mpFrameDrawer;
    std::shared_ptr<MapDrawer> mpMapDrawer;

    std::shared_ptr<Track> mpTrack;
    std::shared_ptr<Map> mpMap;

};

#endif //OBJECTSLAM_SYSTEM_H
