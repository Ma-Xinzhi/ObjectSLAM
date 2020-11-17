#include "Map.h"

#include <glog/logging.h>


void Map::AddQuadric(std::shared_ptr<g2o::Quadric> pQuadric) {
    std::unique_lock<std::mutex> lk(mMutexMap);
    mvpQuadric.push_back(pQuadric);
}

std::vector<std::shared_ptr<g2o::Quadric>> Map::GetAllQuadric(){
    std::unique_lock<std::mutex> lk(mMutexMap);
    return mvpQuadric;
}

void Map::AddKeyFrame(std::shared_ptr<Frame> pkeyframe) {
    std::unique_lock<std::mutex> lk(mMutexMap);
    mvpKeyFrame.push_back(pkeyframe);
}

std::vector<std::shared_ptr<Frame>> Map::GetAllKeyFrame() {
    std::unique_lock<std::mutex> lk(mMutexMap);
    return mvpKeyFrame;
}

void Map::AddObservation(std::shared_ptr<Observation> pob) {
    //　TODO 先按一个物体最简单的方式来处理，考虑数据关联的问题
    int label = pob->mLabel;
    if(mmObjectObservation.find(label) != mmObjectObservation.end()){
        mmObjectObservation[label].push_back(pob);
    } else{
        Observations obs;
        obs.push_back(pob);
        mmObjectObservation.insert(std::make_pair(label, obs));
    }
}

bool Map::DeleteObservation(int label) {
    if(mmObjectObservation.find(label) == mmObjectObservation.end()){
        LOG(WARNING) << "No Observation of Label " << label;
        return false;
    }else{
        mmObjectObservation.erase(label);
        return true;
    }

}