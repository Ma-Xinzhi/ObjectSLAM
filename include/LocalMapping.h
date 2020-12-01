#ifndef OBJECTSLAM_LOCALMAPPING_H
#define OBJECTSLAM_LOCALMAPPING_H

#include "Map.h"

#include <mutex>

class Tracking;
class LocalMapping{
public:
    LocalMapping(std::shared_ptr<Map> pMap);
    
    void SetTracker(std::shared_ptr<Tracking> pTracker);

    // Main function
    void Run();

    void InsertKeyFrame(std::shared_ptr<KeyFrame> pKF);

    // Thread Synch
    void RequestStop();
    void RequestReset();
    bool Stop();
    void Release();
    bool isStopped();
    bool StopRequested();
    bool AcceptKeyFrames();
    void SetAcceptKeyFrames(bool flag);
    bool SetNotStop(bool flag);

    void InterruptBA();

    void RequestFinish();
    bool isFinished();

    int KeyframesInQueue(){
        std::unique_lock<std::mutex> lock(mMutexNewKFs);
        return mlNewKeyFrames.size();
    }

protected:

    bool CheckNewKeyFrames();
    void ProcessNewKeyFrame();
    void CreateNewMapPoints();

    void MapPointCulling();
    void SearchInNeighbors();

    void KeyFrameCulling();

    Eigen::Matrix3d ComputeF12(std::shared_ptr<KeyFrame> &pKF1, std::shared_ptr<KeyFrame> &pKF2);

    Eigen::Matrix3d SkewSymmetricMatrix(const Eigen::Vector3d &v);

    bool mbMonocular;

    void ResetIfRequested();
    bool mbResetRequested;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    std::shared_ptr<Map> mpMap;

    std::shared_ptr<Tracking> mpTracker;

    std::list<std::shared_ptr<KeyFrame>> mlNewKeyFrames;

    std::shared_ptr<KeyFrame> mpCurrentKeyFrame;

    std::list<std::shared_ptr<MapPoint>> mlpRecentAddedMapPoints;

    std::mutex mMutexNewKFs;

    bool mbAbortBA;

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;

    bool mbAcceptKeyFrames;
    std::mutex mMutexAccept;

};

#endif //OBJECTSLAM_LOCALMAPPING_H
