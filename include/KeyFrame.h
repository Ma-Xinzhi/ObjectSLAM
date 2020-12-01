#ifndef OBJECTSLAM_KEYFRAME_H
#define OBJECTSLAM_KEYFRAME_H

#include "Frame.h"
#include "ORBextractor.h"

#include <mutex>
#include <set>
#include <map>
#include <Eigen/Core>

class MapPoint;
class Map;

class KeyFrame{
public:
    KeyFrame(std::shared_ptr<Frame> F, std::shared_ptr<Map> pMap);

    // Pose functions
    void SetPose(const Eigen::Matrix4d &Twc);
    Eigen::Matrix4d GetPose();
    Eigen::Matrix4d GetPoseInverse();
    Eigen::Matrix3d GetRotation();
    Eigen::Vector3d GetCameraCenter();
    Eigen::Vector3d GetTranslation();

    // Covisibility graph functions
    void AddConnection(std::shared_ptr<KeyFrame> pKF, const int &weight);
    void EraseConnection(std::shared_ptr<KeyFrame> pKF);
    void UpdateConnections();
    void UpdateBestCovisibles();
    std::set<std::shared_ptr<KeyFrame>> GetConnectedKeyFrames();
    std::vector<std::shared_ptr<KeyFrame> > GetVectorCovisibleKeyFrames();
    std::vector<std::shared_ptr<KeyFrame>> GetBestCovisibilityKeyFrames(int nums);
    std::vector<std::shared_ptr<KeyFrame>> GetCovisiblesByWeight(int w);
    int GetWeight(std::shared_ptr<KeyFrame> pKF);

    // Spanning tree functions
    void AddChild(std::shared_ptr<KeyFrame> pKF);
    void EraseChild(std::shared_ptr<KeyFrame> pKF);
    void ChangeParent(std::shared_ptr<KeyFrame> pKF);
    std::set<std::shared_ptr<KeyFrame>> GetChilds();
    std::shared_ptr<KeyFrame> GetParent();
    bool hasChild(std::shared_ptr<KeyFrame> pKF);

    // MapPoint observation functions
    void AddMapPoint(std::shared_ptr<MapPoint> pMP, const size_t &idx);
    void EraseMapPointMatch(const size_t &idx);
    void EraseMapPointMatch(std::shared_ptr<MapPoint> pMP);
    void ReplaceMapPointMatch(const size_t &idx, std::shared_ptr<MapPoint> pMP);
    std::set<std::shared_ptr<MapPoint>> GetMapPoints();
    std::vector<std::shared_ptr<MapPoint>> GetMapPointMatches();
    int TrackedMapPoints(int minObs);
    std::shared_ptr<MapPoint> GetMapPoint(size_t idx);

    // KeyPoint functions
    std::vector<size_t> GetFeaturesInArea(float x, float y, float r, const int minLevel=-1, const int maxLevel=-1) const;
    Eigen::Vector3d UnprojectStereo(int i);

    // Image
    bool IsInImage(float x, float y) const;

    // Enable/Disable bad flag changes
    void SetNotErase();
    void SetErase();

    // Set/check bad flag
    void SetBadFlag();
    bool isBad();

    static bool WeightComp( int a, int b){
        return a>b;
    }

    static bool IdComp(std::shared_ptr<KeyFrame> pKF1, std::shared_ptr<KeyFrame> pKF2){
        return pKF1->mnId<pKF2->mnId;
    }
public:

    static unsigned long int nNextId;
    unsigned long int mnId;
    const unsigned long int mnFrameId;

    const double mTimeStamp;

    // Grid (to speed up feature matching)
    const int mnGridCols;
    const int mnGridRows;
    const float mfGridElementWidthInv;
    const float mfGridElementHeightInv;

    // Variables used by the tracking
    long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnFuseTargetForKF;

    // Variables used by the local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnBAFixedForKF;

    // Calibration parameters
    const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;

    // Number of KeyPoints
    const int N;

    // KeyPoints, stereo coordinate and descriptors (all associated by an index)
    const std::vector<cv::KeyPoint> mvKeys;
    const std::vector<cv::KeyPoint> mvKeysUn;
    const std::vector<float> mvuRight; // negative value for monocular points
    const std::vector<float> mvDepth; // negative value for monocular points
    const cv::Mat mDescriptors;

    // Pose relative to parent (this is computed when bad flag is activated)
    Eigen::Matrix4d mTcp;

    // Scale
    const int mnScaleLevels;
    const float mfScaleFactor;
    const float mfLogScaleFactor;
    const std::vector<float> mvScaleFactors;
    const std::vector<float> mvLevelSigma2;
    const std::vector<float> mvInvLevelSigma2;

    // Image bounds and calibration
    const float mnMinX;
    const float mnMinY;
    const float mnMaxX;
    const float mnMaxY;
    const cv::Mat mK;

private:
    Eigen::Matrix4d mTwc;
    Eigen::Matrix3d mRwc;
    Eigen::Vector3d mtwc;

    std::vector<std::shared_ptr<MapPoint>> mvpMapPoints;

    // Grid over the image to speed up feature matching
    std::vector<std::vector<std::vector<size_t>>> mGrid;

    std::map<std::shared_ptr<KeyFrame>,int> mConnectedKeyFrameWeights;
    std::vector<std::shared_ptr<KeyFrame>> mvpOrderedConnectedKeyFrames;
    std::vector<int> mvOrderedWeights;

    // Spanning Tree and Loop Edges
    bool mbFirstConnection;
    std::shared_ptr<KeyFrame> mpParent;
    std::set<std::shared_ptr<KeyFrame>> mspChildrens;

    //这个可能和闭环相关，先不考虑
    std::set<std::shared_ptr<KeyFrame>> mspLoopEdges;

    // Bad flags
    bool mbNotErase;
    bool mbToBeErased;
    bool mbBad;

    // 防止出现交叉引用
    std::weak_ptr<Map> mpMap;

    std::mutex mMutexPose;
    std::mutex mMutexConnections;
    std::mutex mMutexFeatures;
};

#endif //OBJECTSLAM_KEYFRAME_H
