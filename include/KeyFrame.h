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
    KeyFrame(Frame& F, std::shared_ptr<Map> pMap);

    // Pose functions
    void SetPose(const Eigen::Matrix4d &Tcw);
    Eigen::Matrix4d GetPose();
    Eigen::Matrix4d GetPoseInverse();
    Eigen::Vector3d GetCameraCenter();
    Eigen::Matrix3d GetRotation();
    Eigen::Vector3d GetTranslation();

private:
    Eigen::Matrix4d mTwc;
    Eigen::Matrix3d mRwc;
    Eigen::Vector3d mtwc;

    std::vector<std::shared_ptr<MapPoint>> mvpMapPoints;

    // Grid over the image to speed up feature matching
    std::vector<std::vector<std::vector<size_t>>> mGrid;

    std::map<std::shared_ptr<KeyFrame>,int> mConnectedKeyFrameWeights;
    std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;
    std::vector<int> mvOrderedWeights;

    // Spanning Tree and Loop Edges
    bool mbFirstConnection;
    KeyFrame* mpParent;
    std::set<KeyFrame*> mspChildrens;
    std::set<KeyFrame*> mspLoopEdges;

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
