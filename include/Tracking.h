#ifndef OBJECTSLAM_TRACK_H
#define OBJECTSLAM_TRACK_H

#include "Map.h"
#include "Frame.h"
#include "Observation.h"
#include "InitializeQuadric.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "ORBextractor.h"

class Tracking{
public:
    Tracking(const std::string& strSettingPath, std::shared_ptr<Map> pmap, std::shared_ptr<MapDrawer> pmapdrawer,
          std::shared_ptr<FrameDrawer> pframedrawer);

    void GrabPoseAndSingleObject(const g2o::SE3Quat& pose, std::shared_ptr<Observation> bbox, const cv::Mat& img_RGB);
    void GrabPoseAndObjects(const g2o::SE3Quat& pose, const Observations& bbox, const cv::Mat& img_RGB);
    void GrabRGBDImageAndSingleObject(const cv::Mat& img_RGB, const cv::Mat& depth, std::shared_ptr<Observation> bbox);

    std::shared_ptr<Frame> GetCurrentFrame() const { return mpCurrentFrame; }
    cv::Mat GetCurrentImage() const { return mCurImg; }
    Eigen::Matrix3d GetCalib() const { return mCalib; }

private:
    void Track();

    void UpdateObjectObservation();
    void CheckInitialization();
    void ProcessVisualization();

    bool NeedNewKeyFrame();

private:
    std::shared_ptr<InitializeQuadric> mpInitailizeQuadric;
    std::shared_ptr<FrameDrawer> mpFrameDrawer;
    std::shared_ptr<MapDrawer> mpMapDrawer;
    std::shared_ptr<Map> mpMap;
    std::shared_ptr<ORBextractor> mpORBextractor;

    std::shared_ptr<Frame> mpCurrentFrame;
    cv::Mat mCurImg;
    cv::Mat mGrayImg;

    int mImgWidth, mImgHeight;

    int mMinFrames, mMaxFrames;

    float mbf;
    float mThDepth;

    float mDepthMapFactor;

    bool mbRGB;

    Eigen::Matrix3d mCalib;
    cv::Mat mK;
    cv::Mat mDistCoef;
};

#endif //OBJECTSLAM_TRACK_H
