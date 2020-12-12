#ifndef OBJECTSLAM_TRACK_H
#define OBJECTSLAM_TRACK_H

#include "Map.h"
#include "Frame.h"
#include "Detector.h"
#include "InitializeQuadric.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "ORBextractor.h"
#include "LocalMapping.h"

#include <list>

class Tracking{
public:
    Tracking(const std::string& strSettingPath, const std::shared_ptr<Map>& pMap, const std::shared_ptr<MapDrawer>& pMapDrawer,
          const std::shared_ptr<FrameDrawer>& pFrameDrawer);

    void GrabPoseAndSingleObject(const g2o::SE3Quat& pose, const Object& bbox, const cv::Mat& img_RGB);
    void GrabPoseAndObjects(const g2o::SE3Quat& pose, const Objects& bbox, const cv::Mat& img_RGB);
    void GrabRGBDImage(const cv::Mat& img_RGB, const cv::Mat& depth, double timestamp);

    void SetLocalMapper(const std::shared_ptr<LocalMapping>& pLocalMapper) {mpLocalMapping = pLocalMapper;}

    cv::Mat GetK() const { return mK; }

private:
    void ObjectObservationCulling(Objects& obs);
    void QuadricInitialization();
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
    std::list<Eigen::Matrix4d> mlRelativeFramePoses;
    std::list<std::shared_ptr<KeyFrame>> mlpReferences;
    std::list<double> mlFrameTimes;
    std::list<bool> mlbLost;

    void Reset();

private:
    std::shared_ptr<InitializeQuadric> mpInitailizeQuadric;
    std::shared_ptr<FrameDrawer> mpFrameDrawer;
    std::shared_ptr<MapDrawer> mpMapDrawer;
    std::shared_ptr<Map> mpMap;
    std::shared_ptr<ORBextractor> mpORBextractor;

    std::shared_ptr<LocalMapping> mpLocalMapping;

    std::unique_ptr<Detector> mDetector;
    float mfThresh;

    int mnSize, mnBorder;

    // Motion Model
    Eigen::Matrix4d mVelocity; // T_last_current
    bool mbVelocity;

    int mImgWidth, mImgHeight;

    int mMinFrames, mMaxFrames;

    float mbf;
    float mThDepth;

    float mDepthMapFactor;

    bool mbRGB;

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
