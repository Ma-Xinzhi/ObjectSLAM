#include "LocalMapping.h"
#include "Tracking.h"
#include "Optimizer.h"
#include "ORBmatcher.h"

#include <opencv2/core/eigen.hpp>

LocalMapping::LocalMapping(std::shared_ptr<Map> pMap): mpMap(pMap), mbResetRequested(false),
    mbFinishRequested(false), mbFinished(true), mbAbortBA(false), mbStopped(false), mbStopRequested(false),
    mbNotStop(false), mbAcceptKeyFrames(true), mbMonocular(false){}

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

//            if(!CheckNewKeyFrames())
            SearchInNeighbors();

            mbAbortBA = false;

            if(!CheckNewKeyFrames() && !StopRequested()){
                if(mpMap->KeyFramesInMap()>2)
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA, mpMap);

                KeyFrameCulling();
            }
        }
        else if(Stop()){
            while(isStopped() && !CheckFinish())
                usleep(3000);
            if(CheckFinish())
                break;
        }

        ResetIfRequested();

        SetAcceptKeyFrames(true);

        if(CheckFinish())
            break;

        usleep(3000);
    }

    SetFinish();
}

void LocalMapping::InsertKeyFrame(std::shared_ptr<KeyFrame> pKF) {
    std::unique_lock<std::mutex> lk(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA = true;
}

void LocalMapping::SetAcceptKeyFrames(bool flag) {
    std::unique_lock<std::mutex> lk(mMutexNewKFs);
    mbAcceptKeyFrames = flag;
}

bool LocalMapping::CheckNewKeyFrames() {
    std::unique_lock<std::mutex> lk(mMutexNewKFs);
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
                // 匹配到的地图已有地图点
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame)){
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    pMP->UpdateNormalAndDepth();
                    pMP->ComputeDistinctiveDescriptors();
                }
//                else // this can only happen for new stereo points inserted by the Tracking
                    // 在构建关键帧的时候，双目/RGB-D相机能够自己构造一些地图点
//                    mlpRecentAddedMapPoints.push_back(pMP);
            }
        }
    }

    // 更新共视图间的连接关系
    mpCurrentKeyFrame->UpdateConnections();

    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

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
            // 如果当前帧相较于地图点创建时的关键帧太远，将该点删去，不作为后续剔除对象
        else if(nCurrentKFid-pMP->mnFirstKFid >= 3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }
}

void LocalMapping::CreateNewMapPoints() {

    std::vector<std::pair<float, int>> vDepthIdx;
    vDepthIdx.reserve(mpCurrentKeyFrame->N);
    for (int i = 0; i < mpCurrentKeyFrame->N; ++i) {
        float z = mpCurrentKeyFrame->mvDepth[i];
        if(z > 0)
            vDepthIdx.emplace_back(std::make_pair(z, i));
    }

    int nPoints = 0;
    if(!vDepthIdx.empty()){
        sort(vDepthIdx.begin(), vDepthIdx.end());

        for(auto& item : vDepthIdx){
            int idx = item.second;

            bool bCreateNew = false;

            std::shared_ptr<MapPoint> pMP = mpCurrentKeyFrame->GetMapPoint(idx);
            if(!pMP)
                bCreateNew = true;
            else if(pMP->Observations()<1){
                bCreateNew = true;
                mpCurrentKeyFrame->EraseMapPointMatch(idx);
            }

            if(bCreateNew){
                Eigen::Vector3d x3D = mpCurrentKeyFrame->UnprojectStereo(idx);
                std::shared_ptr<MapPoint> pNewMP = std::make_shared<MapPoint>(x3D, mpCurrentKeyFrame, mpMap);
                pNewMP->AddObservation(mpCurrentKeyFrame, idx);
                mpCurrentKeyFrame->AddMapPoint(pNewMP, idx);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mlpRecentAddedMapPoints.push_back(pNewMP);
                nPoints++;
            }
            // TODO 这里增加多少个地图点合适
            if(nPoints > 100 && item.first > mpCurrentKeyFrame->mThDepth)
                break;
        }
    }
    LOG(INFO) << "Create new " << nPoints << " map points" << std::endl;
}

// 检查新添加的地图点
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

void LocalMapping::SearchInNeighbors() {
    int nn = 10;
    std::vector<std::shared_ptr<KeyFrame>> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    std::vector<std::shared_ptr<KeyFrame>> vpTargetKFs;

    for(auto& pKFi : vpNeighKFs){
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        std::vector<std::shared_ptr<KeyFrame>> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for(auto& pKFi2 : vpSecondNeighKFs){
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF == mpCurrentKeyFrame->mnId || pKFi2->mnId == mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
            pKFi2->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;
        }
    }

    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    std::vector<std::shared_ptr<MapPoint>> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(auto& pKFi : vpTargetKFs)
        matcher.Fuse(pKFi, vpMapPointMatches);

    // Search matches by projection from target KFs in current KF
    std::vector<std::shared_ptr<MapPoint>> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());
    for(auto& pKFi : vpTargetKFs){
        std::vector<std::shared_ptr<MapPoint>> vpMapPointsKFi = pKFi->GetMapPointMatches();
        for(auto& pMP : vpMapPointsKFi){
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }
    matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates);

    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    int npoints = 0;
    for(auto& pMP : vpMapPointMatches){
        if(pMP){
            if(!pMP->isBad()){
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
                npoints++;
            }
        }
    }
    LOG(INFO) << "Current key frame matched map points: " << npoints << std::endl;

    mpCurrentKeyFrame->UpdateConnections();

}

void LocalMapping::KeyFrameCulling() {
    std::vector<std::shared_ptr<KeyFrame>> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    int thObs = 3;

    for(auto& pKF : vpLocalKeyFrames){
        if(pKF->mnId == 0)
            continue;
        std::vector<std::shared_ptr<MapPoint>> vpMapPoints = pKF->GetMapPointMatches();
        int nRedundantObservations = 0;
        int nMPs = 0;
        for(int i=0; i<vpMapPoints.size(); ++i){
            std::shared_ptr<MapPoint> pMP = vpMapPoints[i];
            if(pMP){
                if(!pMP->isBad()){
                    if(!mbMonocular){
                        if(pKF->mvDepth[i]<0 || pKF->mvDepth[i]>pKF->mThDepth)
                            continue;
                    }

                    nMPs++;
                    // 该帧的地图点的观察大于阈值，且尺度接近或更好的观察也大于阈值
                    if(pMP->Observations()>thObs){
                        int scaleLevel = pKF->mvKeysUn[i].octave;
                        std::map<std::shared_ptr<KeyFrame>, size_t> obs = pMP->GetObservations();
                        int nObs = 0;
                        for(auto& ob : obs){
                            std::shared_ptr<KeyFrame> pKFi = ob.first;
                            if(pKFi == pKF)
                                continue;
                            int scaleLeveli = pKFi->mvKeysUn[ob.second].octave;

                            if(scaleLeveli <= scaleLevel+1){
                                nObs++;
                                if(nObs>=thObs)
                                    break;
                            }
                        }
                        if(nObs >= thObs)
                            nRedundantObservations++;
                    }
                }
            }
        }
        // 剔除冗余关键帧
        if(nRedundantObservations > 0.9*nMPs)
            pKF->SetBadFlag();
    }
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