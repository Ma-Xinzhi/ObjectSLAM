#ifndef OBJECTSLAM_TRACK_H
#define OBJECTSLAM_TRACK_H

#include "Map.h"
#include "Frame.h"
#include "Observation.h"
#include "InitializeQuadric.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "ORBextractor.h"

#include <list>

class Tracking{
public:
    Tracking(const std::string& strSettingPath, std::shared_ptr<Map> pMap, std::shared_ptr<MapDrawer> pMapDrawer,
          std::shared_ptr<FrameDrawer> pFrameDrawer);

    void GrabPoseAndSingleObject(const g2o::SE3Quat& pose, std::shared_ptr<Observation> bbox, const cv::Mat& img_RGB);
    void GrabPoseAndObjects(const g2o::SE3Quat& pose, const Observations& bbox, const cv::Mat& img_RGB);
    void GrabRGBDImageAndSingleObject(const cv::Mat& img_RGB, const cv::Mat& depth, std::shared_ptr<Observation> bbox);

    Eigen::Matrix3d GetCalib() const { return mCalib; }

private:
    void UpdateObjectObservation();
    void CheckInitialization();
//    void ProcessVisualization();

    /*-------------------------------------------------------------*/
    // Main tracking function. It is independent of the input sensor.
    void Track();

    // Map initialization for stereo and RGB-D
    void StereoInitialization();

    void CheckReplacedInLastFrame();
    bool TrackReferenceKeyFrame();
    void UpdateLastFrame();
    bool TrackWithMotionModel();

    bool Relocalization();

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();

    bool TrackLocalMap();
    void SearchLocalPoints();

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

public:
    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Current Frame
//    Frame mCurrentFrame;
    std::shared_ptr<Frame> mpCurrentFrame;
    cv::Mat mCurImg; // RGB
    cv::Mat mGrayImg;

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    std::list<cv::Mat> mlRelativeFramePoses;
    std::list<KeyFrame*> mlpReferences;
    std::list<double> mlFrameTimes;
    std::list<bool> mlbLost;

    void Reset();

private:
    std::shared_ptr<InitializeQuadric> mpInitailizeQuadric;
    std::shared_ptr<FrameDrawer> mpFrameDrawer;
    std::shared_ptr<MapDrawer> mpMapDrawer;
    std::shared_ptr<Map> mpMap;
    std::shared_ptr<ORBextractor> mpORBextractor;

    // Motion Model
    Eigen::Matrix4d mVelocity;
    bool mbVelocity;

    int mImgWidth, mImgHeight;

    int mMinFrames, mMaxFrames;

    float mbf;
    float mThDepth;

    float mDepthMapFactor;

    bool mbRGB;

    Eigen::Matrix3d mCalib;
    cv::Mat mK;
    cv::Mat mDistCoef;

    //Current matches in frame
    int mnMatchesInliers;

    //Last Frame, KeyFrame and Relocalisation Info
    std::shared_ptr<KeyFrame> mpLastKeyFrame;
    std::shared_ptr<Frame> mpLastFrame;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;

    //Local Map
    std::shared_ptr<KeyFrame> mpReferenceKF;
    std::vector<std::shared_ptr<KeyFrame>> mvpLocalKeyFrames;
    std::vector<std::shared_ptr<MapPoint>> mvpLocalMapPoints;

   std::list<std::shared_ptr<MapPoint>> mlpTemporalPoints;

};

#endif //OBJECTSLAM_TRACK_H
