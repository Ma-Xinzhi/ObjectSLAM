#ifndef OBJECTSLAM_MAPDRAWER_H
#define OBJECTSLAM_MAPDRAWER_H

#include <mutex>
#include <string>
#include <map>
#include <vector>
#include <memory>

#include <pangolin/pangolin.h>

#include "Map.h"

class MapDrawer{
public:
    MapDrawer(const std::string& strSettingPath,std::shared_ptr<Map> pmap);

    void DrawCurrentCamera(const pangolin::OpenGlMatrix& Twc);
    void DrawTrajectory();
    void DrawEllipsoids();
    void DrawAxisNormal();

    void SetCurrentCameraPose(const Eigen::Matrix4d& pose);
    pangolin::OpenGlMatrix GetCurrentOpenGLMatrix();

private:
    float mKeyFrameSize;
    float mKeyFrameLineWidth;
    float mGraphLineWidth;
    float mPointSize;
    float mCameraSize;
    float mCameraLineWidth;

    // 这里的位姿在跟踪线程中设置，在显示线程中应用，需要锁
    std::mutex mMutexCamera;

    Eigen::Matrix4d mCameraPose;
    std::shared_ptr<Map> mpMap;

};



#endif //OBJECTSLAM_MAPDRAWER_H
