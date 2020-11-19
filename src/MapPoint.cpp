#include "MapPoint.h"
#include "Map.h"

long unsigned int MapPoint::nNextId = 0;
std::mutex MapPoint::mGlobalMutex;

MapPoint::MapPoint(const Eigen::Vector3d &Pos, std::shared_ptr<KeyFrame> pRefKF, std::shared_ptr<Map> pMap){}

MapPoint::MapPoint(const Eigen::Vector3d &Pos, std::shared_ptr<Map> pMap, std::shared_ptr<Frame> pFrame,
                   const int &idxF): mWorldPos(Pos), mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0),
                   mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mpRefKF(nullptr), mnVisible(1),
                   mnFound(1), mbBad(false), mpReplaced(nullptr), mpMap(pMap){
    Eigen::Vector3d Ow = pFrame->GetCameraCenter();
    mNormalVector = mWorldPos - Ow;
    mNormalVector = mNormalVector / mNormalVector.norm();

    Eigen::Vector3d PC = Pos - Ow;
    float dist = PC.norm();
    int level = pFrame->mvKeysUn[idxF].octave;
    float levelScaleFactor = pFrame->mvScaleFactors[level];
    int nlevels = pFrame->mnScaleLevels;

    mfMaxDistance = dist*levelScaleFactor;
    mfMinDistance = mfMaxDistance/pFrame->mvScaleFactors[nlevels-1];

    pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);

    std::unique_lock<std::mutex> lk(mpMap.lock()->mMutexPointCreation);
    mnId = nNextId++;
}

void MapPoint::SetWorldPos(const Eigen::Vector3d &Pos) {
    std::unique_lock<std::mutex> lk2(mGlobalMutex);
    std::unique_lock<std::mutex> lk(mMutexPos);
    mWorldPos = Pos;
}

Eigen::Vector3d MapPoint::GetWorldPos(){
    std::unique_lock<std::mutex> lk(mMutexPos);
    return mWorldPos;
}

Eigen::Vector3d MapPoint::GetNormal() {
    std::unique_lock<std::mutex> lk(mMutexPos);
    return mNormalVector;
}

std::shared_ptr<KeyFrame> MapPoint::GetReferenceKeyFrame() {
    std::unique_lock<std::mutex> lk(mMutexFeatures);
    return mpRefKF;
}

void MapPoint::AddObservation(std::shared_ptr<KeyFrame> pKF, size_t idx) {
    std::unique_lock<std::mutex> lk(mMutexFeatures);
    if(mObservations.count(pKF))
        return;
    mObservations[pKF] = idx;
    if(pKF->mvuRight[idx] >= 0)
        nObs += 2;
    else
        nObs++;
}

std::map<std::shared_ptr<KeyFrame>,size_t> MapPoint::GetObservations() {
    std::unique_lock<std::mutex> lk(mMutexFeatures);
    return mObservations;
}

void MapPoint::EraseObservation(std::shared_ptr<KeyFrame> pKF) {
    bool bBad = false;
    {
        std::unique_lock<std::mutex> lk(mMutexFeatures);
        if(mObservations.count(pKF)){
            int idx = mObservations[pKF];
            if(pKF->mvuRight[idx]>=0)
                nObs-=2;
            else
                nObs--;
            mObservations.erase(pKF);

            if(mpRefKF.get() == pKF.get())
                mpRefKF = std::shared_ptr<KeyFrame>(mObservations.begin()->first);

            if(nObs<=2)
                bBad = true;
        }
    }
    if(bBad)
        SetBadFlag();
}

int MapPoint::Observations() {
    std::unique_lock<std::mutex> lk(mMutexFeatures);
    return nObs;
}

void MapPoint::SetBadFlag() {
    std::map<std::shared_ptr<KeyFrame>, size_t> obs;
    {
        std::unique_lock<std::mutex> lk1(mMutexFeatures);
        std::unique_lock<std::mutex> lk2(mMutexPos);
        mbBad = true;
        obs = mObservations;
        mObservations.clear();
    }
    for(const auto& item : obs){
        std::shared_ptr<KeyFrame> pKF = item.first;
        pKF->EraseMapPointMatch(item.second);
    }
    mpMap.lock()->EraseMapPoint(std::shared_ptr<MapPoint>(this));
}

std::shared_ptr<MapPoint> MapPoint::GetReplaced() {
    std::unique_lock<std::mutex> lk1(mMutexFeatures);
    std::unique_lock<std::mutex> lk2(mMutexPos);
    return mpReplaced;
}

void MapPoint::Replace(std::shared_ptr<MapPoint> pMP) {
    if(pMP->mnId == mnId)
        return;
    int nvisible, nfound;
    std::map<std::shared_ptr<KeyFrame>, size_t> obs;
    {
        std::unique_lock<std::mutex> lk1(mMutexFeatures);
        std::unique_lock<std::mutex> lk2(mMutexPos);
        obs = mObservations;
        mObservations.clear();
        mbBad = true;
        nvisible = mnVisible;
        nfound = mnFound;
        mpReplaced = pMP;
    }
    for(const auto& ob : obs){
        std::shared_ptr<KeyFrame> pKF = ob.first;

        if(pMP->IsInKeyFrame(pKF)){
            pKF->ReplaceMapPointMatch(ob.second, pMP);
            pMP->AddObservation(pKF, ob.second);
        }else
            pKF->EraseMapPointMatch(ob.second);
    }
    pMP->IncreaseFound(nfound);
    pMP->IncreaseVisible(nvisible);
    pMP->ComputeDistinctiveDescriptors();

    mpMap.lock()->EraseMapPoint(std::shared_ptr<MapPoint>(this));
}

bool MapPoint::isBad() {
    std::unique_lock<std::mutex> lk1(mMutexFeatures);
    std::unique_lock<std::mutex> lk2(mMutexPos);
    return mbBad;
}

void MapPoint::IncreaseFound(int n) {
    std::unique_lock<std::mutex> lk(mMutexFeatures);
    mnFound += n;
}

void MapPoint::IncreaseVisible(int n) {
    std::unique_lock<std::mutex> lk(mMutexFeatures);
    mnVisible += n;
}

float MapPoint::GetFoundRatio() {
    std::unique_lock<std::mutex> lk(mMutexFeatures);
    return static_cast<float>(mnFound) / mnVisible;
}

void MapPoint::ComputeDistinctiveDescriptors() {
    std::vector<cv::Mat> vDescriptors;
    std::map<std::shared_ptr<KeyFrame>, size_t> observations;
    {
        std::unique_lock<std::mutex> lk(mMutexFeatures);
        if(mbBad)
            return;
        observations = mObservations;
    }
    if(observations.empty())
        return;
    vDescriptors.reserve(observations.size());

    for(auto& ob : observations){
        std::shared_ptr<KeyFrame> pKF = ob.first;
        if(!pKF->isBad())
            vDescriptors.push_back(pKF->mDescriptors.row(ob.second));
    }

    if(vDescriptors.empty())
        return;

    int N = vDescriptors.size();
    float Distances[N][N];
    for(int i=0; i<N; i++){
        Distances[i][i] = 0;
        for(int j=i+1; j<N; j++){
            int dist = ORBmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
            Distances[i][j] = dist;
            Distances[j][i] = dist;
        }
    }

    int BestMedian = INT_MAX;
    int BestIdx = 0;

    for(int i=0; i<N; i++){
        std::vector<int> vDists(Distances[i], Distances[i]+N);
        std::sort(vDists.begin(), vDists.end());

        int median = vDists[0.5*(N-1)];
        if(median < BestMedian){
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        std::unique_lock<std::mutex> lk(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();
    }
}

cv::Mat MapPoint::GetDescriptor() {
    std::unique_lock<std::mutex> lk(mMutexFeatures);
    return mDescriptor.clone();
}

int MapPoint::GetIndexInKeyFrame(std::shared_ptr<KeyFrame> pKF) {
    std::unique_lock<std::mutex> lk(mMutexFeatures);
    if(mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}

bool MapPoint::IsInKeyFrame(std::shared_ptr<KeyFrame> pKF) {
    std::unique_lock<std::mutex> lk(mMutexFeatures);
    return mObservations.count(pKF);
}

void MapPoint::UpdateNormalAndDepth() {
    std::map<std::shared_ptr<KeyFrame>, size_t> observations;
    std::shared_ptr<KeyFrame> pRefKF;
    Eigen::Vector3d Pos;
    {
        std::unique_lock<std::mutex> lk1(mMutexFeatures);
        std::unique_lock<std::mutex> lk2(mMutexPos);
        if(mbBad)
            return;
        observations = mObservations;
        pRefKF = mpRefKF;
        Pos = mWorldPos;
    }

    if(observations.empty())
        return;

    Eigen::Vector3d normal = Eigen::Vector3d::Zero();

    int n=0;
    for(auto& ob : observations){
        std::shared_ptr<KeyFrame> pKF = ob.first;
        Eigen::Vector3d Owi = pKF->GetCameraCenter();
        Eigen::Vector3d normali = mWorldPos - Owi;
        normal = normal + normali / normali.norm();
        n++;
    }

    Eigen::Vector3d PC = Pos - pRefKF->GetCameraCenter();
    float dist = PC.norm();
    int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
    float levelScaleFactor = pRefKF->mvScaleFactors[level];
    int nlevels = pRefKF->mnScaleLevels;

    {
        std::unique_lock<std::mutex> lk3(mMutexPos);
        mfMaxDistance = dist*levelScaleFactor;
        mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nlevels-1];
        mNormalVector = normal/n;
    }
}

float MapPoint::GetMinDistanceInvariance() {
    std::unique_lock<std::mutex> lk(mMutexPos);
    return 0.8*mfMinDistance;
}

float MapPoint::GetMaxDistanceInvariance() {
    std::unique_lock<std::mutex> lk(mMutexPos);
    return 1.2*mfMaxDistance;
}

// 距离越小，说明层数越高，所对应的尺度也就越大
int MapPoint::PredictScale(const float &currentDist, std::shared_ptr<KeyFrame> pKF) {
    float ratio;
    {
        std::unique_lock<std::mutex> lk(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }
    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale >= pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;

    return nScale;

}

int MapPoint::PredictScale(const float &currentDist, std::shared_ptr<Frame> pF) {
    float ratio;
    {
        std::unique_lock<std::mutex> lk(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }
    int nScale = ceil(log(ratio)/pF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale >= pF->mnScaleLevels)
        nScale = pF->mnScaleLevels-1;

    return nScale;
}

