#ifndef OBJECTSLAM_OBSERVATION_H
#define OBJECTSLAM_OBSERVATION_H

#include "Quadric.h"
#include "Frame.h"

struct Observation{
    int mLabel;
    Eigen::Vector4d mBbox;
    double mProb;
    std::weak_ptr<Frame> mpFrame;
    std::weak_ptr<g2o::Quadric> mpQuadric;
};
typedef std::vector<std::shared_ptr<Observation>> Observations;

//struct Observation3D{
//    unsigned int mLabel;
//    std::shared_ptr<g2o::Quadric> mpObj;
//    double mProb;
//    std::shared_ptr<Frame> mpFrame;
//    Observation3D(unsigned int label, std::shared_ptr<g2o::Quadric> pobj, double prob):Label(label), pObj(pobj), Prob(prob){}
//};
//typedef std::vector<std::shared_ptr<Observation3D>> Observation3Ds;


#endif //OBJECTSLAM_OBSERVATION_H
