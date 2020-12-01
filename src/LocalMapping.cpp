#include "LocalMapping.h"
#include "Tracking.h"
#include "Optimizer.h"
#include "ORBmatcher.h"

#include <opencv2/core/eigen.hpp>

LocalMapping::LocalMapping(std::shared_ptr<Map> pMap): mpMap(pMap), mbResetRequested(false),
    mbFinishRequested(false), mbFinished(true), mbAbortBA(false), mbStopped(false), mbStopRequested(false),
    mbNotStop(false), mbAcceptKeyFrames(true){}

void LocalMapping::SetTracker(std::shared_ptr<Tracking> pTracker) {
    mpTracker = pTracker;
}

void LocalMapping::Run() {
    mbFinished = false;
    while(true){
        SetAcceptKeyFrames(false);

        if(CheckNewKeyFrames()){
            ProcessNewKeyFrame();

            MapPointCulling();

            CreateNewMapPoints();


        }
    }

    SetFinish();
}

void LocalMapping::SetAcceptKeyFrames(bool flag) {
    std::unique_lock<std::mutex> lk(mMutexAccept);
    mbAcceptKeyFrames = flag;
}

bool LocalMapping::CheckNewKeyFrames() {
    std::unique_lock<std::mutex> lk(mMutexAccept);
    return !mlNewKeyFrames.empty();
}

void LocalMapping::ProcessNewKeyFrame() {
    {
        std::unique_lock<std::mutex> lk(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // 存在与关键帧匹配的地图点，并未与该关键帧进行关联
    std::vector<std::shared_ptr<MapPoint>> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for(int i=0; i<vpMapPointMatches.size(); ++i){
        std::shared_ptr<MapPoint> pMP = vpMapPointMatches[i];
        if(pMP){
            if(!pMP->isBad()){
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame)){
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    pMP->UpdateNormalAndDepth();
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                    // 在构建关键帧的时候，双目/RGB-D相机能够自己构造一些地图点
                    mlpRecentAddedMapPoints.push_back(pMP);
            }
        }
    }

    // 更新共视图间的连接关系
    mpCurrentKeyFrame->UpdateConnections();

    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

// 检查新添加的地图点
void LocalMapping::MapPointCulling() {
    auto lit = mlpRecentAddedMapPoints.begin();
    unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs = 3;

    while(lit != mlpRecentAddedMapPoints.end()){
        std::shared_ptr<MapPoint> pMP = *lit;
        if(pMP->isBad())
            lit = mlpRecentAddedMapPoints.erase(lit);
        // 如果匹配率太低的话，舍弃该点
        else if(pMP->GetFoundRatio() < 0.25){
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        // 如果当前帧相较于地图点创建时的关键帧有点距离，但是观测次数很少，舍弃该点
        else if(nCurrentKFid-pMP->mnFirstKFid >= 2 && pMP->Observations() <= 3){
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        // 如果当前帧相较于地图点创建时的关键帧太远，舍弃该点
        else if(nCurrentKFid-pMP->mnFirstKFid >= 3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }
}

void LocalMapping::CreateNewMapPoints() {
    int nn = 10;

    std::vector<std::shared_ptr<KeyFrame>> vpNeiphKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    ORBmatcher matcher(0.6, false);

    Eigen::Matrix4d Twc1 = mpCurrentKeyFrame->GetPose();
    Eigen::Matrix3d Rwc1 = mpCurrentKeyFrame->GetRotation();
    Eigen::Vector3d Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    float fx1 = mpCurrentKeyFrame->fx;
    float fy1 = mpCurrentKeyFrame->fy;
    float cx1 = mpCurrentKeyFrame->cx;
    float cy1 = mpCurrentKeyFrame->cy;
    float invfx1 = mpCurrentKeyFrame->invfx;
    float invfy1 = mpCurrentKeyFrame->invfy;

    float ratioFactor = 1.5*mpCurrentKeyFrame->mfScaleFactor;

    int nnew = 0;

    for(int i=0; i<vpNeiphKFs.size(); ++i){
        // 这里的意思应该是共视关系最大的建立地图点后，如果又有新的关键帧就不需要再建立新的地图点
        if(i>0 && CheckNewKeyFrames())
            return;

        std::shared_ptr<KeyFrame> pKF2 = vpNeiphKFs[i];

        Eigen::Vector3d Ow2 = pKF2->GetCameraCenter();
        Eigen::Vector3d vBaseline = Ow2-Ow1;
        float baseline = vBaseline.norm();

        if(baseline < pKF2->mb)
            continue;

        Eigen::Matrix3d F12 = ComputeF12(mpCurrentKeyFrame, pKF2);

        std::vector<std::pair<size_t, size_t>> vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, F12, vMatchedIndices);
    }
}

Eigen::Matrix3d LocalMapping::ComputeF12(std::shared_ptr<KeyFrame> &pKF1, std::shared_ptr<KeyFrame> &pKF2) {
    Eigen::Matrix3d Rwkf1 = pKF1->GetRotation();
    Eigen::Matrix3d Rwkf2 = pKF2->GetRotation();
    Eigen::Vector3d twkf1 = pKF1->GetTranslation();
    Eigen::Vector3d twkf2 = pKF2->GetTranslation();

    Eigen::Matrix3d R12 = Rwkf1.transpose() * Rwkf2;
    Eigen::Vector3d t12 = twkf2 - twkf1;

    Eigen::Matrix3d t12x = SkewSymmetricMatrix(t12);

    Eigen::Matrix3d K1;
    cv::cv2eigen(pKF1->mK, K1);
    Eigen::Matrix3d K2;
    cv::cv2eigen(pKF2->mK, K2);

    return K1.transpose().inverse()*t12x*R12*K2.inverse();
}

bool LocalMapping::AcceptKeyFrames() {
    std::unique_lock<std::mutex> lk(mMutexAccept);
    return mbAcceptKeyFrames;
}

bool LocalMapping::SetNotStop(bool flag) {
    std::unique_lock<std::mutex> lk(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA() {
    mbAbortBA = true;
}

void LocalMapping::RequestStop() {
    std::unique_lock<std::mutex> lk(mMutexStop);
    mbStopRequested = true;
    std::unique_lock<std::mutex> lk2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop() {
    std::unique_lock<std::mutex> lk(mMutexStop);
    if(mbStopRequested && !mbNotStop){
        mbStopped = true;
        LOG(INFO) << "Local Mapping STOP..." << std::endl;
        return true;
    }
    return false;
}

bool LocalMapping::isStopped() {
    std::unique_lock<std::mutex> lk(mMutexStop);
    return mbStopped;
}

bool LocalMapping::StopRequested() {
    std::unique_lock<std::mutex> lk(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::RequestReset() {
    {
        std::unique_lock<std::mutex> lk(mMutexReset);
        mbResetRequested = true;
    }

    while(true){
        {
            std::unique_lock<std::mutex> lk(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
}

void LocalMapping::ResetIfRequested() {
    std::unique_lock<std::mutex> lk(mMutexReset);
    if(mbResetRequested){
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested = false;
    }
}

void LocalMapping::RequestFinish() {
    std::unique_lock<std::mutex> lk(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish() {
    std::unique_lock<std::mutex> lk(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish() {
    std::unique_lock<std::mutex> lk(mMutexFinish);
    mbFinished = true;
    std::unique_lock<std::mutex> lk2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished() {
    std::unique_lock<std::mutex> lk(mMutexFinish);
    return mbFinished;
}

Eigen::Matrix3d LocalMapping::SkewSymmetricMatrix(const Eigen::Vector3d &v) {
    Eigen::Matrix3d Skew_Mat;
    Skew_Mat << 0, -v[2], v[1],
                v[2], 0, -v[0],
                -v[1], v[0], 0;
    return Skew_Mat;
}

void LocalMapping::Release() {
    std::unique_lock<std::mutex> lk(mMutexStop);
    std::unique_lock<std::mutex> lk2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    mlNewKeyFrames.clear();

    LOG(INFO) << "Local Mapping RELEASE..." << std::endl;
}