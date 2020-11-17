#ifndef OBJECTSLAM_FRAMEDRAWER_H
#define OBJECTSLAM_FRAMEDRAWER_H

#include <opencv2/opencv.hpp>
#include <mutex>

#include "Map.h"
#include "Frame.h"

class Track;
class FrameDrawer{
public:
    FrameDrawer(const std::shared_ptr<Map>& pmap);

    void Update(Track* ptrack);

    cv::Mat DrawFrame();
    // 显示检测框，投影椭圆和框
    cv::Mat DrawFrameAll();

private:
    void DrawObservationOnImage(cv::Mat& img);
    void DrawProjectionOnImage(cv::Mat& img);

    // image会在Track线程中更新，也会在Viewer线程中显示，属于共享的数据需要加锁
    std::mutex mMutex;

    std::shared_ptr<Map> mpMap;
    std::shared_ptr<Frame> mpCurrentFrame;

    cv::Mat mImg;
    Observations mvpObservation;
    Eigen::Matrix3d mCalib;
};

#endif //OBJECTSLAM_FRAMEDRAWER_H
