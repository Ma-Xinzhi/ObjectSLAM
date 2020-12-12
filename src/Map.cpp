#include "Map.h"

#include <glog/logging.h>

Map::Map(): mnMaxKFId(0) {}

void Map::AddKeyFrame(const std::shared_ptr<KeyFrame>& pKF) {
    std::unique_lock<std::mutex> lk(mMutexMap);
    mspKeyFrames.insert(pKF);
    if(pKF->mnId > mnMaxKFId)
        mnMaxKFId = pKF->mnId;
}

void Map::AddMapPoint(const std::shared_ptr<MapPoint>& pMp) {
    std::unique_lock<std::mutex> lk(mMutexMap);
    mspMapPoints.insert(pMp);
}

void Map::AddObjectObservation(const Object &Ob){
    //　TODO 假设每类物体只有一个，按简单的方式来处理，需要考虑一类物体多个的数据关联问题
    int label = Ob.mObjectId;
    if(mObjectObservations.find(label) != mObjectObservations.end()){
        mObjectObservations[label].push_back(Ob);
    } else{
        Objects obs;
        obs.push_back(Ob);
        mObjectObservations.insert(std::make_pair(label, obs));
    }
}

void Map::AddQuadric(const std::shared_ptr<g2o::Quadric>& pQuadric) {
    std::unique_lock<std::mutex> lk(mMutexMap);
    mvpQuadric.push_back(pQuadric);
}

void Map::EraseKeyFrame(const std::shared_ptr<KeyFrame>& pKF) {
    std::unique_lock<std::mutex> lk(mMutexMap);
    mspKeyFrames.erase(pKF);
}

void Map::EraseMapPoint(const std::shared_ptr<MapPoint>& pMp) {
    std::unique_lock<std::mutex> lk(mMutexMap);
    mspMapPoints.erase(pMp);
}

bool Map::EraseObjectObservation(int label) {
    if(mObjectObservations.find(label) == mObjectObservations.end()){
        LOG(WARNING) << "No Observation of Label " << label;
        return false;
    }else{
        mObjectObservations.erase(label);
        return true;
    }
}

void Map::SetReferenceMapPoints(const std::vector<std::shared_ptr<MapPoint>> &vpMPs) {
    std::unique_lock<std::mutex> lk(mMutexMap);
    mvpReferenceMapPoints = vpMPs;
}

std::vector<std::shared_ptr<KeyFrame>> Map::GetAllKeyFrame() {
    std::unique_lock<std::mutex> lk(mMutexMap);
    return std::vector<std::shared_ptr<KeyFrame>>(mspKeyFrames.begin(), mspKeyFrames.end());
}

std::vector<std::shared_ptr<MapPoint>> Map::GetAllMapPoints() {
    std::unique_lock<std::mutex> lk(mMutexMap);
    return std::vector<std::shared_ptr<MapPoint>>(mspMapPoints.begin(), mspMapPoints.end());
}

std::vector<std::shared_ptr<MapPoint>> Map::GetReferenceMapPoints() {
    std::unique_lock<std::mutex> lk(mMutexMap);
    return mvpReferenceMapPoints;
}

std::vector<std::shared_ptr<g2o::Quadric>> Map::GetAllQuadric(){
    std::unique_lock<std::mutex> lk(mMutexMap);
    return mvpQuadric;
}

long unsigned int Map::KeyFramesInMap() {
    std::unique_lock<std::mutex> lk(mMutexMap);
    return mspKeyFrames.size();
}

long unsigned int Map::MapPointsInMap() {
    std::unique_lock<std::mutex> lk(mMutexMap);
    return mspMapPoints.size();
}

long unsigned int Map::GetMaxKFId() {
    std::unique_lock<std::mutex> lk(mMutexMap);
    return mnMaxKFId;
}

void Map::clear() {
    mspKeyFrames.clear();
    mspMapPoints.clear();
    mnMaxKFId = 0;
    mvpReferenceMapPoints.clear();
}