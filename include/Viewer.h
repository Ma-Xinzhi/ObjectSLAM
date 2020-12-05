#ifndef OBJECTSLAM_VIEWER_H
#define OBJECTSLAM_VIEWER_H

#include <opencv2/opencv.hpp>

#include "MapDrawer.h"
#include "FrameDrawer.h"

#include <thread>
#include <mutex>

class Viewer{
public:
    Viewer(const std::string& strSettingPath, std::shared_ptr<MapDrawer> pmapdrawer,
            std::shared_ptr<FrameDrawer> pframedrawer);
    ~Viewer();

    void Run();

    void RequestFinish();
    void RequestStop();

    bool IsFinished();
    bool IsStopped();

    void Release();

private:
    std::shared_ptr<MapDrawer> mpMapDrawer;
    std::shared_ptr<FrameDrawer> mpFrameDrawer;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    bool Stop();
    bool mbStopped;
    bool mbStopRequested;
    std::mutex mMutexStop;

    std::thread mthread;

    double mT;
    int mImgHeight;
    int mImgWidth;

    float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;
};

#endif //OBJECTSLAM_VIEWER_H
