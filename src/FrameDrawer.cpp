#include "FrameDrawer.h"
#include "Tracking.h"

#include <opencv2/core/eigen.hpp>

FrameDrawer::FrameDrawer(const std::shared_ptr<Map> &pmap): mpMap(pmap) {}

cv::Mat FrameDrawer::DrawFrame() {
    cv::Mat img;
    {
        std::unique_lock<std::mutex> lk(mMutex);
        mImg.copyTo(img);
    }
    DrawObservationOnImage(img);
    return img;
}

cv::Mat FrameDrawer::DrawFrameAll() {
    cv::Mat img;
    {
        std::unique_lock<std::mutex> lk(mMutex);
        mImg.copyTo(img);
    }
    DrawObservationOnImage(img);
    DrawProjectionOnImage(img);
    return img;
}

void FrameDrawer::DrawObservationOnImage(cv::Mat& img) {
    for(const auto& ob : mvpObservation){
        Eigen::Vector4d bbox = ob->mBbox;
        int label = ob->mLabel;
        cv::Rect rect(cv::Point(bbox[0], bbox[1]), cv::Point(bbox[2], bbox[3]));
        cv::rectangle(img, rect, cv::Scalar(0,0,255),2);
        cv::putText(img, std::to_string(label), cv::Point(bbox[0], bbox[1]), CV_FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0,255,0), 2);
    }
}

void FrameDrawer::DrawProjectionOnImage(cv::Mat& img) {
    std::vector<std::shared_ptr<g2o::Quadric>> quadrics = mpMap->GetAllQuadric();
    for(const auto& quadric : quadrics){
        g2o::SE3Quat pose = mpCurrentFrame->GetPose();
        if(quadric->CheckObservability(pose)){
            Eigen::Matrix3d Calib;
            cv::cv2eigen(mK, Calib);
            // 两种方式求解投影的bounding box
//            Eigen::Vector4d rect = quadric->ProjectOntoImageRectByEquation(pose, Calib);
            Vector5d ellipse = quadric->ProjectOntoImageEllipse(pose, Calib);
            double angle = ellipse[4] * 180 / M_PI;
            // 这里的角度是度数单位，并且是顺时针旋转，计算的时候是按照逆时针旋转计算，需要取个反
            cv::ellipse(img, cv::Point(ellipse[0], ellipse[1]), cv::Size(ellipse[2], ellipse[3]),
                        -angle, 0, 360, cv::Scalar(0,255,0));
            Eigen::Vector4d rect = quadric->GetBboxFromEllipse(ellipse);
            cv::rectangle(img, cv::Point(rect[0], rect[1]), cv::Point(rect[2], rect[3]), cv::Scalar(0,255,0), 2);
        }
    }
}

void FrameDrawer::Update(std::shared_ptr<Tracking> pTracker) {
    std::unique_lock<std::mutex> lk(mMutex);
    pTracker->mCurImg.copyTo(mImg);

    mpCurrentFrame = pTracker->mpCurrentFrame;
    mvpObservation = mpCurrentFrame->GetDetectionResults();
    mvCurrentKeys = mpCurrentFrame->mvKeys;
    N = mvCurrentKeys.size();


    mK = pTracker->GetK();
}