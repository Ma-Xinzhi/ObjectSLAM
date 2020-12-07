#ifndef OBJECTSLAM_MAP_H
#define OBJECTSLAM_MAP_H

#include <vector>
#include <memory>
#include <mutex>
#include <map>

#include "Frame.h"
#include "KeyFrame.h"
#include "Quadric.h"
#include "Observation.h"
#include "MapPoint.h"

class Map{
public:
    Map();

    void AddQuadric(const std::shared_ptr<g2o::Quadric>& pQuadric);
    std::vector<std::shared_ptr<g2o::Quadric>> GetAllQuadric();

    void AddKeyFrame(const std::shared_ptr<KeyFrame>& pKF);
    void EraseKeyFrame(const std::shared_ptr<KeyFrame>& pKF);
    std::vector<std::shared_ptr<KeyFrame>> GetAllKeyFrame();

    void AddMapPoint(const std::shared_ptr<MapPoint>& pMp);
    void EraseMapPoint(const std::shared_ptr<MapPoint>& pMp);
    std::vector<std::shared_ptr<MapPoint>> GetAllMapPoints();

    void SetReferenceMapPoints(const std::vector<std::shared_ptr<MapPoint>>& vpMPs);
    std::vector<std::shared_ptr<MapPoint>> GetReferenceMapPoints();

    void AddObservation(const std::shared_ptr<Observation>& pOb);
    bool EraseObservation(int label);
    std::map<int, Observations> GetAllObservation() const { return mmObjectObservation; };

    void clear();

    long unsigned int MapPointsInMap();
    long unsigned int KeyFramesInMap();

    long unsigned int GetMaxKFId();

    std::vector<std::shared_ptr<KeyFrame>> mvpKeyFrameOrigins;

    std::mutex mMutexMapUpdate;

    // This avoid that two points are created simultaneously in separate threads (id conflict)
    std::mutex mMutexPointCreation;

private:
    // 定位建图线程和可视化线程共享资源，需要锁
    std::mutex mMutexMap;

    std::set<std::shared_ptr<KeyFrame>> mspKeyFrames;
    std::set<std::shared_ptr<MapPoint>> mspMapPoints;

    std::vector<std::shared_ptr<MapPoint>> mvpReferenceMapPoints;

    std::vector<std::shared_ptr<g2o::Quadric>> mvpQuadric;
    std::vector<std::shared_ptr<KeyFrame>> mvpKeyFrame;

    std::map<int, Observations> mmObjectObservation; //地图中当前所有可能构成物体的观测，将语义类别作为关键字

    long unsigned int mnMaxKFId;
};

#endif //OBJECTSLAM_MAP_H
