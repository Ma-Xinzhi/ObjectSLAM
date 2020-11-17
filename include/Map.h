#ifndef OBJECTSLAM_MAP_H
#define OBJECTSLAM_MAP_H

#include <vector>
#include <memory>
#include <mutex>
#include <map>

#include "Frame.h"
#include "Quadric.h"
#include "Observation.h"

class Map{
public:
    Map() {}
    void AddQuadric(std::shared_ptr<g2o::Quadric> pQuadric);
    std::vector<std::shared_ptr<g2o::Quadric>> GetAllQuadric();

    void AddKeyFrame(std::shared_ptr<Frame> pkeyframe);
    std::vector<std::shared_ptr<Frame>> GetAllKeyFrame();

    void AddObservation(std::shared_ptr<Observation> pobservation);
    bool DeleteObservation(int label);
    std::map<int, Observations> GetAllObservation() const { return mmObjectObservation; };

private:
    // 定位建图线程和可视化线程共享资源，需要锁
    std::mutex mMutexMap;

    std::vector<std::shared_ptr<g2o::Quadric>> mvpQuadric;
    std::vector<std::shared_ptr<Frame>> mvpKeyFrame;

    std::map<int, Observations> mmObjectObservation; //地图中当前所有可能构成物体的观测，将语义类别作为关键字
};

#endif //OBJECTSLAM_MAP_H
