#include "Frame.h"
#include "Observation.h"

Frame::Frame(const g2o::SE3Quat& pose, const std::vector<std::shared_ptr<Observation>>& bbox, const cv::Mat& Image):
    mTwc(pose), mvpObservation(bbox), mFrameImg(Image){
    static int id = 0;
    mFrameId = id++;
    mKeyFrameId = -1;
    mbIsKeyframe = false;
}

Frame::Frame(const g2o::SE3Quat& pose, std::shared_ptr<Observation> bbox, const cv::Mat& Image):
    mTwc(pose), mFrameImg(Image){
    static int id = 0;
    mFrameId = id++;
    mKeyFrameId = -1;
    mbIsKeyframe = false;
    mvpObservation.push_back(bbox);
}

void Frame::SetKeyFrame() {
    static int kf_id = 0;
    mKeyFrameId = kf_id++;
    mbIsKeyframe = true;
}

