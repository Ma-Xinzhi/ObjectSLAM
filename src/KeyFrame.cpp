#include "KeyFrame.h"
#include "MapPoint.h"
#include "Map.h"
#include "ORBmatcher.h"

unsigned long int KeyFrame::nNextId = 0;
KeyFrame::KeyFrame(const std::shared_ptr<Frame>& F, const std::shared_ptr<Map>& pMap):
    mnFrameId(F->mnId), mTimeStamp(F->mTimeStamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
    mfGridElementWidthInv(Frame::mfGridElementWidthInv), mfGridElementHeightInv(Frame::mfGridElementHeightInv),
    mnTrackReferenceForFrame(-1), mnFuseTargetForKF(-1), mnBALocalForKF(-1), mnBAFixedForKF(-1),
    fx(Frame::fx), fy(Frame::fy), cx(Frame::cx), cy(Frame::cy), invfx(Frame::invfx), invfy(Frame::invfy),
    mbf(F->mbf), mb(F->mb), mThDepth(F->mThDepth), N(F->N), mvKeys(F->mvKeys), mvKeysUn(F->mvKeysUn),
    mvuRight(F->mvuRight), mvDepth(F->mvDepth), mDescriptors(F->mDescriptors.clone()),mnScaleLevels(F->mnScaleLevels),
    mfScaleFactor(F->mfScaleFactor), mfLogScaleFactor(F->mfLogScaleFactor), mvScaleFactors(F->mvScaleFactors),
    mvLevelSigma2(F->mvLevelSigma2), mvInvLevelSigma2(F->mvInvLevelSigma2), mnMinX(Frame::mnMinX), mnMinY(Frame::mnMinY),
    mnMaxX(Frame::mnMaxX), mnMaxY(Frame::mnMaxY), mK(F->mK),mbFirstConnection(true), mpParent(nullptr), mbNotErase(false),
    mbToBeErased(false), mbBad(false), mpMap(pMap)
{
    mnId = nNextId++;

    mGrid.resize(mnGridCols);
    for(int i=0; i<mnGridCols; i++){
        mGrid[i].resize(mnGridRows);
        for(int j=0; j<mnGridRows; j++)
            mGrid[i][j] = F->mGrid[i][j];
    }

    mvpMapPoints.resize(F->mvpMapPoints.size());
    for(int i=0; i<F->mvpMapPoints.size(); i++)
        mvpMapPoints[i] = F->mvpMapPoints[i];

    SetPose(F->GetPose());
}

void KeyFrame::SetPose(const Eigen::Matrix4d &Twc) {
    mTwc = Twc;
    mRwc = Twc.block(0,0,3,3);
    mtwc = Twc.col(3).head(3);
}

Eigen::Matrix4d KeyFrame::GetPose() {
    std::unique_lock<std::mutex> lk(mMutexPose);
    return mTwc;
}

Eigen::Matrix3d KeyFrame::GetRotation() {
    std::unique_lock<std::mutex> lk(mMutexPose);
    return mRwc;
}

Eigen::Vector3d KeyFrame::GetCameraCenter() {
    std::unique_lock<std::mutex> lk(mMutexPose);
    return mtwc;
}

Eigen::Vector3d KeyFrame::GetTranslation() {
    std::unique_lock<std::mutex> lk(mMutexPose);
    return mtwc;
}

void KeyFrame::AddConnection(const std::shared_ptr<KeyFrame>& pKF, const int &weight)
{
    {
        std::unique_lock<std::mutex> lk(mMutexConnections);
        if(!mConnectedKeyFrameWeights.count(pKF))
            mConnectedKeyFrameWeights[pKF] = weight;
        else if(mConnectedKeyFrameWeights[pKF] != weight)
            mConnectedKeyFrameWeights[pKF] = weight;
        else
            return;
    }
    UpdateBestCovisibles();
}

void KeyFrame::UpdateBestCovisibles() {
    std::unique_lock<std::mutex> lk(mMutexConnections);
    std::vector<std::pair<int, std::shared_ptr<KeyFrame>>> vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    for(auto& item : mConnectedKeyFrameWeights)
        vPairs.emplace_back(std::make_pair(item.second, item.first));
    std::sort(vPairs.begin(), vPairs.end());
    std::vector<std::shared_ptr<KeyFrame>> vKFs;
    std::vector<int> vWs;
    for(auto& item : vPairs){
        vKFs.push_back(item.second);
        vWs.push_back(item.first);
    }
    std::reverse(vKFs.begin(), vKFs.end());
    std::reverse(vWs.begin(), vWs.end());

    mvpOrderedConnectedKeyFrames = vKFs;
    mvOrderedWeights = vWs;
}

std::set<std::shared_ptr<KeyFrame>> KeyFrame::GetConnectedKeyFrames() {
    std::unique_lock<std::mutex> lk(mMutexConnections);
    std::set<std::shared_ptr<KeyFrame>> s;
    for(auto& item : mConnectedKeyFrameWeights)
        s.insert(item.first);
    return s;
}

std::vector<std::shared_ptr<KeyFrame> > KeyFrame::GetVectorCovisibleKeyFrames() {
    std::unique_lock<std::mutex> lk(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

std::vector<std::shared_ptr<KeyFrame>> KeyFrame::GetBestCovisibilityKeyFrames(int nums) {
    std::unique_lock<std::mutex> lk(mMutexConnections);
    if(mvpOrderedConnectedKeyFrames.size() < nums)
        return mvpOrderedConnectedKeyFrames;
    else
        return std::vector<std::shared_ptr<KeyFrame>>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+nums);
}

std::vector<std::shared_ptr<KeyFrame>> KeyFrame::GetCovisiblesByWeight(int w) {
    std::unique_lock<std::mutex> lk(mMutexConnections);
    if(mvpOrderedConnectedKeyFrames.empty())
        return std::vector<std::shared_ptr<KeyFrame>>();
    auto it = std::upper_bound(mvOrderedWeights.begin(), mvOrderedWeights.end(),w, KeyFrame::WeightComp);
    if(it == mvOrderedWeights.end())
        return std::vector<std::shared_ptr<KeyFrame>>();
    else{
        int n = it - mvOrderedWeights.begin();
        return std::vector<std::shared_ptr<KeyFrame>>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n);
    }
}

int KeyFrame::GetWeight(std::shared_ptr<KeyFrame> pKF) {
    std::unique_lock<std::mutex> lk(mMutexConnections);
    if(mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}

void KeyFrame::AddMapPoint(const std::shared_ptr<MapPoint>& pMP, const size_t &idx) {
    std::unique_lock<std::mutex> lk(mMutexFeatures);
    mvpMapPoints[idx] = pMP;
}

void KeyFrame::EraseMapPointMatch(const size_t &idx) {
    std::unique_lock<std::mutex> lk(mMutexFeatures);
    mvpMapPoints[idx].reset();
}

void KeyFrame::EraseMapPointMatch(const std::shared_ptr<MapPoint>& pMP) {
    std::unique_lock<std::mutex> lk(mMutexFeatures);
    int idx = pMP->GetIndexInKeyFrame(shared_from_this());
    if(idx >= 0)
        mvpMapPoints[idx].reset();
}

void KeyFrame::ReplaceMapPointMatch(const size_t &idx, const std::shared_ptr<MapPoint>& pMP) {
    std::unique_lock<std::mutex> lk(mMutexFeatures);
    mvpMapPoints[idx] = pMP;
}

std::set<std::shared_ptr<MapPoint>> KeyFrame::GetMapPoints() {
    std::unique_lock<std::mutex> lk(mMutexFeatures);
    std::set<std::shared_ptr<MapPoint>> s;
    for(auto& mp : mvpMapPoints){
        if(mp.expired())
            continue;
        if(!mp.lock()->isBad())
            s.insert(mp.lock());
    }
    return s;
}

int KeyFrame::TrackedMapPoints(int minObs) {
    std::unique_lock<std::mutex> lk(mMutexFeatures);
    int nPoints = 0;
    bool bCheckObs = minObs > 0;
    for(auto& pMp : mvpMapPoints){
        if(!pMp.expired()){
            if(!pMp.lock()->isBad()){
                if(bCheckObs){
                    if(pMp.lock()->Observations()>=minObs)
                        nPoints++;
                } else
                    nPoints++;
            }
        }
    }
    return nPoints;
}

std::vector<std::shared_ptr<MapPoint>> KeyFrame::GetMapPointMatches() {
    std::unique_lock<std::mutex> lk(mMutexFeatures);
    std::vector<std::shared_ptr<MapPoint>> vpMapPoints;
    vpMapPoints.reserve(mvpMapPoints.size());
    for(auto& pMP : mvpMapPoints)
        vpMapPoints.push_back(pMP.lock());
    return vpMapPoints;
}

std::shared_ptr<MapPoint> KeyFrame::GetMapPoint(size_t idx) {
    std::unique_lock<std::mutex> lk(mMutexFeatures);
    return mvpMapPoints[idx].lock();
}

void KeyFrame::UpdateConnections() {
    std::map<std::shared_ptr<KeyFrame>, int> KFcounter;
    std::vector<std::weak_ptr<MapPoint>> vpMp;
    {
        std::unique_lock<std::mutex> lk(mMutexFeatures);
        vpMp = mvpMapPoints;
    }
    // 当前关键帧观察到的地图点
    for(auto& pMp : vpMp){
        if(pMp.expired())
            continue;

        if(pMp.lock()->isBad())
            continue;

        std::map<std::shared_ptr<KeyFrame>, size_t> observations = pMp.lock()->GetObservations();
        // 地图点在其他关键帧中的观测
        for(auto& ob : observations){
            if(ob.first->mnId == mnId)
                continue;
            KFcounter[ob.first]++;
        }
    }

    if(KFcounter.empty())
        return;

    int nmax = 0;
    std::shared_ptr<KeyFrame> pKFmax = nullptr;
    int th = 15;

    // 如果当前帧的地图点有超过一定阈值数量，在其他关键帧中观察到，当前帧与这些关键帧就建立联系，其他关键帧增加当前帧之间的关联
    std::vector<std::pair<int, std::shared_ptr<KeyFrame>>> vPairs;
    vPairs.reserve(KFcounter.size());
    for(auto& item : KFcounter){
        if(item.second > nmax){
            nmax = item.second;
            pKFmax = item.first;
        }
        if(item.second >= th){
            vPairs.emplace_back(std::make_pair(item.second, item.first));
            item.first->AddConnection(shared_from_this(), item.second);
        }
    }
    if(vPairs.empty()){
        vPairs.emplace_back(std::make_pair(nmax, pKFmax));
        pKFmax->AddConnection(shared_from_this(), nmax);
    }

    // 默认按升序排列
    std::sort(vPairs.begin(), vPairs.end());
    std::vector<std::shared_ptr<KeyFrame>> vKFs;
    std::vector<int> vWs;

    for(auto& pair : vPairs){
        vKFs.push_back(pair.second);
        vWs.push_back(pair.first);
    }
    // 按降序排列
    std::reverse(vKFs.begin(), vKFs.end());
    std::reverse(vWs.begin(), vWs.end());

    {
        std::unique_lock<std::mutex> lk(mMutexConnections);

        mConnectedKeyFrameWeights = KFcounter;
        mvpOrderedConnectedKeyFrames = vKFs;
        mvOrderedWeights = vWs;

        if(mbFirstConnection && mnId != 0){
            mpParent = mvpOrderedConnectedKeyFrames.front();
            mpParent->AddChild(shared_from_this());
            mbFirstConnection = false;
        }
    }
}

void KeyFrame::AddChild(const std::shared_ptr<KeyFrame>& pKF) {
    std::unique_lock<std::mutex> lk(mMutexConnections);
    mspChildrens.insert(pKF);
}

void KeyFrame::EraseChild(const std::shared_ptr<KeyFrame>& pKF) {
    std::unique_lock<std::mutex> lk(mMutexConnections);
    mspChildrens.erase(pKF);
}

void KeyFrame::ChangeParent(const std::shared_ptr<KeyFrame>& pKF) {
    std::unique_lock<std::mutex> lk(mMutexConnections);
    mpParent = pKF;
    pKF->AddChild(shared_from_this());
}

std::set<std::shared_ptr<KeyFrame>> KeyFrame::GetChilds() {
    std::unique_lock<std::mutex> lk(mMutexConnections);
    return mspChildrens;
}

std::shared_ptr<KeyFrame> KeyFrame::GetParent() {
    std::unique_lock<std::mutex> lk(mMutexConnections);
    return mpParent;
}

bool KeyFrame::hasChild(const std::shared_ptr<KeyFrame>& pKF) {
    std::unique_lock<std::mutex> lk(mMutexConnections);
    return mspChildrens.count(pKF);
}

// TODO 这里的删除还要好好考虑考虑，如果

void KeyFrame::SetNotErase() {
    std::unique_lock<std::mutex> lk(mMutexConnections);
    mbNotErase = true;
}

void KeyFrame::SetErase() {
    {
        std::unique_lock<std::mutex> lk(mMutexConnections);
        // TODO 这里的判断条件好好考虑考虑，怎样才能让一个不可删除的关键帧变为可删除的
        if(mspLoopEdges.empty()){
            mbNotErase = false;
        }
    }
    if(!mbNotErase){
        SetBadFlag();
    }
}

// 第一帧不删除，其余帧删除的时候需要处理子节点，为其找到相应的父节点
void KeyFrame::SetBadFlag() {
    {
        std::unique_lock<std::mutex> lk(mMutexConnections);
        if(mnId == 0)
            return;
    }
    for(auto& item : mConnectedKeyFrameWeights)
        item.first->EraseConnection(shared_from_this());

    {
        std::unique_lock<std::mutex> lk(mMutexConnections);
        std::unique_lock<std::mutex> lk1(mMutexFeatures);

        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        std::set<std::shared_ptr<KeyFrame>> sParentCandidates;
        sParentCandidates.insert(mpParent);

        // 对自己的每个孩子重新分配父节点
        while(!mspChildrens.empty()){
            bool bContinue = false;

            int max = -1;
            std::shared_ptr<KeyFrame> pC;
            std::shared_ptr<KeyFrame> pP;

            for(auto& child : mspChildrens){
                if(child->isBad())
                    continue;
                // 检查是否有候选的父节点与子节点相关联
                std::vector<std::shared_ptr<KeyFrame>> vpConnected = child->GetVectorCovisibleKeyFrames();
                for(auto& pConnected : vpConnected){
                    for(auto& candidate : sParentCandidates){
                        if(pConnected->mnId == candidate->mnId){
                            int w = child->GetWeight(pConnected);
                            if(w > max){
                                pC = child;
                                pP = pConnected;
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }
            if(bContinue){
                pC->ChangeParent(pP);
                sParentCandidates.insert(pP);
                mspChildrens.erase(pC);
            } else
                break;
        }

        // 如果仍有未找到父节点的子节点，就将其父节点设置为当前要删除关键帧的父节点
        if(!mspChildrens.empty()){
            for(auto& child : mspChildrens)
                child->ChangeParent(mpParent);
        }

        mpParent->EraseChild(shared_from_this());
        mTcp = mTwc.inverse() * mpParent->GetPose();
        mbBad = true;
    }

    mpMap.lock()->EraseKeyFrame(shared_from_this());

}

bool KeyFrame::isBad() {
    std::unique_lock<std::mutex> lk(mMutexConnections);
    return mbBad;
}

// 应该先删除关键帧观测到的地图点，再删除关键帧之间的关联
void KeyFrame::EraseConnection(const std::shared_ptr<KeyFrame>& pKF) {
    bool bUpdate = false;
    {
        std::unique_lock<std::mutex> lk(mMutexConnections);
        if(mConnectedKeyFrameWeights.count(pKF)){
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate = true;
        }
    }
    // 删除后，地图点被观测的数据也会发生变动
    if(bUpdate)
        UpdateConnections();
}

std::vector<size_t> KeyFrame::GetFeaturesInArea(float x, float y, float r, const int minLevel, const int maxLevel) const {
    std::vector<size_t> vIndices;
    vIndices.resize(N);

    int nMinCellX = std::max(0, (int)floor((x-r-mnMinX)*mfGridElementWidthInv));
    if(nMinCellX >= mnGridCols)
        return vIndices;

    int nMaxCellX = std::min(mnGridCols-1, (int)ceil((x+r-mnMinX)*mfGridElementWidthInv));
    if(nMaxCellX < 0)
        return vIndices;

    int nMinCellY = std::max(0, (int)floor((y-r-mnMinY)*mfGridElementHeightInv));
    if(nMinCellY >= mnGridRows)
        return vIndices;

    int nMaxCellY = std::min(mnGridRows-1, (int)ceil((y+r-mnMinX)*mfGridElementHeightInv));
    if(nMaxCellY < 0)
        return vIndices;

    bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int i=nMinCellX; i<=nMaxCellX; ++i){
        for(int j=nMinCellY; j<=nMaxCellY; ++j){
            std::vector<size_t> vCell = mGrid[i][j];
            for(auto& index : vCell){
                cv::KeyPoint kp = mvKeysUn[index];
                if(bCheckLevels){
                    if(kp.octave < minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kp.octave > maxLevel)
                            continue;
                }

                float distx = kp.pt.x - x;
                float disty = kp.pt.y - y;
                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(index);
            }
        }
    }

    return vIndices;
}

bool KeyFrame::IsInImage(float x, float y) const {
    return x>=mnMinX && x<=mnMaxX && y>=mnMinY && y<=mnMaxY;
}

Eigen::Vector3d KeyFrame::UnprojectStereo(int i) {
    float z = mvDepth[i];
    if(z > 0){
        float u = mvKeys[i].pt.x;
        float v = mvKeys[i].pt.y;
        float x = (u-cx)*invfx*z;
        float y = (v-cy)*invfy*z;
        Eigen::Vector3d Pc;
        Pc << x, y, z;

        std::unique_lock<std::mutex> lk(mMutexPose);
        return mRwc*Pc + mtwc;
    } else
        return Eigen::Vector3d::Zero();
}