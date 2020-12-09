#ifndef OBJECTSLAM_FRAMEDRAWER_H
#define OBJECTSLAM_FRAMEDRAWER_H

#include <opencv2/opencv.hpp>
#include <mutex>

#include "Map.h"
#include "Frame.h"

class Tracking;
class FrameDrawer{
public:
    FrameDrawer(const std::shared_ptr<Map>& pmap);

    void Update(Tracking* pTracker);

    cv::Mat DrawFrame();
    // 显示检测框，投影椭圆和框
    cv::Mat DrawFrameAll();

private:
    void DrawTextInfoOnImage(cv::Mat& img, int state, cv::Mat& imText);
    void DrawObservationOnImage(cv::Mat& img);
    void DrawProjectionOnImage(cv::Mat& img);

    cv::Scalar ObjectIdToColor(int obj_id);

    // image会在Track线程中更新，也会在Viewer线程中显示，属于共享的数据需要加锁
    std::mutex mMutex;

    std::shared_ptr<Map> mpMap;
    std::shared_ptr<Frame> mpCurrentFrame;

    cv::Mat mImg;
    int N;
    std::vector<cv::KeyPoint> mvCurrentKeys;
    std::vector<bool> mvbMap, mvbVO;
    int mnTracked, mnTrackedVO;
    std::vector<cv::KeyPoint> mvIniKeys;
    std::vector<int> mvIniMatches;
    int mState;

    Objects mvObservations;
    cv::Mat mK;
};

#endif //OBJECTSLAM_FRAMEDRAWER_H
