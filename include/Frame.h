#ifndef OBJECTSLAM_FRAME_H
#define OBJECTSLAM_FRAME_H

#include <opencv2/core.hpp>
#include <memory>
#include <vector>
#include "Thirdparty/g2o/g2o/types/se3quat.h"

struct Observation;

class Frame {
public:

//    Frame(const cv::Mat& image);
    Frame(const g2o::SE3Quat& pose, std::shared_ptr<Observation> bbox, const cv::Mat& image);
    Frame(const g2o::SE3Quat& pose, const std::vector<std::shared_ptr<Observation>>& bbox, const cv::Mat& image);

    void SetKeyFrame();

    void SetDetectionResults(const std::vector<std::shared_ptr<Observation>>& detection_results) { mvpObservation = detection_results; }
    std::vector<std::shared_ptr<Observation>> GetDetectionResults() const{ return mvpObservation; }

    void SetPose(const g2o::SE3Quat& pose_wc) { mTwc = pose_wc; }
    g2o::SE3Quat GetPose() const { return mTwc; }
    Eigen::Vector3d GetCameraCenter() const { return mTwc.translation(); }
    cv::Mat GetImg() const { return mFrameImg; }

private:
    int mFrameId;  // image topic sequence id, fixed
    int mKeyFrameId;
    cv::Mat mFrameImg;

    bool mbIsKeyframe;

    std::vector<std::shared_ptr<Observation>> mvpObservation; // object detection result
    g2o::SE3Quat mTwc;  // optimized pose  cam to world

};

#endif //OBJECTSLAM_FRAME_H
