#ifndef OBJECTSLAM_OBSERVATION_H
#define OBJECTSLAM_OBSERVATION_H

#include <Eigen/Core>

#include <vector>
#include <memory>

class KeyFrame;
// 检测结果类====
typedef struct Object
{
    Eigen::Vector4d mBbox;  // 边框
    float mProb;             // 置信度
//    std::string mObjectName;// 物体类别名 没有很好的实现方式，而且这个名称也不是很重要
    int mObjectId;           // 类别id
    std::shared_ptr<KeyFrame> mpKeyFrame; // 物体的检测从哪一帧来
//    std::shared_ptr<g2o::Quadric> mpQuadric; // 该物体检测对应那个3D中的物体
} Object;

typedef std::vector<Object> Objects;

#endif //OBJECTSLAM_OBSERVATION_H
