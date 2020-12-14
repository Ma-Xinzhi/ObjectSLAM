#ifndef OBJECTSLAM_TRACK_H
#define OBJECTSLAM_TRACK_H

#include "Map.h"
#include "Frame.h"
#include "Observation.h"
#include "InitializeQuadric.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Detector.h"

class Track{
public:
    Track(const std::string& strSettingPath, std::shared_ptr<Map> pmap, std::shared_ptr<MapDrawer> pmapdrawer,
          std::shared_ptr<FrameDrawer> pframedrawer);

    void GrabPoseAndSingleObject(const g2o::SE3Quat& pose, std::shared_ptr<Observation> bbox, const cv::Mat& img_RGB);
    void GrabPoseAndObjects(const g2o::SE3Quat& pose, const Observations& bbox, const cv::Mat& img_RGB);

    std::shared_ptr<Frame> GetCurrentFrame() const { return mpCurrentFrame; }
    cv::Mat GetCurrentImage() const { return mCurImg; }
    Eigen::Matrix3d GetCalib() const { return mCalib; }

private:
    void UpdateObjectObservation();
    void CheckInitialization();
    void ProcessVisualization();

    bool NeedNewKeyFrame();

private:
    std::shared_ptr<InitializeQuadric> mpInitailizeQuadric;
    std::shared_ptr<FrameDrawer> mpFrameDrawer;
    std::shared_ptr<MapDrawer> mpMapDrawer;
    std::shared_ptr<Map> mpMap;
    std::shared_ptr<Frame> mpCurrentFrame;

    std::unique_ptr<Detector> mpDetector;

    cv::Mat mCurImg;

    int mImgWidth, mImgHeight;

    Eigen::Matrix3d mCalib;
};

#endif //OBJECTSLAM_TRACK_H
