#include "FrameDrawer.h"
#include "Tracking.h"

#include <opencv2/core/eigen.hpp>

FrameDrawer::FrameDrawer(const std::shared_ptr<Map> &pmap): mpMap(pmap) {}

cv::Mat FrameDrawer::DrawFrame() {
    cv::Mat img;
    std::vector<cv::KeyPoint> vCurrentKeys;
    std::vector<bool> vbVO, vbMap;
    int state;
    {
        std::unique_lock<std::mutex> lk(mMutex);
        mImg.copyTo(img);

        state = mState;

        if(mState == Tracking::SYSTEM_NOT_READY)
            mState = Tracking::NO_IMAGES_YET;

        if(mState == Tracking::OK){
            vCurrentKeys = mvCurrentKeys;
            vbVO = mvbVO;
            vbMap = mvbMap;
        }
        else if(mState == Tracking::LOST)
            vCurrentKeys = mvCurrentKeys;
    }

    if(state == Tracking::OK){
        mnTracked = 0;
        mnTrackedVO = 0;
        float r = 5;
        int n = vCurrentKeys.size();
        for (int i = 0; i < n; ++i) {
            cv::Point2f pt1, pt2;
            pt1.x = vCurrentKeys[i].pt.x-r;
            pt1.y = vCurrentKeys[i].pt.y-r;
            pt2.x = vCurrentKeys[i].pt.x+r;
            pt2.y = vCurrentKeys[i].pt.y+r;
            // 与地图中的地图点相关联的特征点
            if(vbMap[i]){
                cv::rectangle(img, pt1, pt2, cv::Scalar(0,255,0));
                cv::circle(img, vCurrentKeys[i].pt, 2, cv::Scalar(0,255,0), -1);
                mnTracked++;
            }
            // 与上一帧中创建的地图点相关联的特征点，感觉这里在实际中不会出现，地图点一旦被创建都是增加观测信息的
            else{
                cv::rectangle(img, pt1, pt2, cv::Scalar(0,0,255));
                cv::circle(img, vCurrentKeys[i].pt, 2, cv::Scalar(0,0,255), -1);
                mnTrackedVO++;
            }
        }
    }

//    DrawObservationOnImage(img);
    return img;
}

cv::Mat FrameDrawer::DrawFrameAll() {
    cv::Mat img;
    std::vector<cv::KeyPoint> vCurrentKeys;
    std::vector<bool> vbVO, vbMap;
    int state;
    {
        std::unique_lock<std::mutex> lk(mMutex);
        mImg.copyTo(img);

        state = mState;

        if(mState == Tracking::SYSTEM_NOT_READY)
            mState = Tracking::NO_IMAGES_YET;

        if(mState == Tracking::OK){
            vCurrentKeys = mvCurrentKeys;
            vbVO = mvbVO;
            vbMap = mvbMap;
        }
        else if(mState == Tracking::LOST)
            vCurrentKeys = mvCurrentKeys;
    }

    if(state == Tracking::OK){
        mnTracked = 0;
        mnTrackedVO = 0;
        float r = 5;
        int n = vCurrentKeys.size();
        for (int i = 0; i < n; ++i) {
            cv::Point2f pt1, pt2;
            pt1.x = vCurrentKeys[i].pt.x-r;
            pt1.y = vCurrentKeys[i].pt.y-r;
            pt2.x = vCurrentKeys[i].pt.x+r;
            pt2.y = vCurrentKeys[i].pt.y+r;
            // 与地图中的地图点相关联的特征点
            if(vbMap[i]){
                cv::rectangle(img, pt1, pt2, cv::Scalar(0,255,0));
                cv::circle(img, vCurrentKeys[i].pt, 2, cv::Scalar(0,255,0), -1);
                mnTracked++;
            }
                // 与上一帧中创建的地图点相关联的特征点，感觉这里在实际中不会出现，地图点一旦被创建都是增加观测信息的
            else{
                cv::rectangle(img, pt1, pt2, cv::Scalar(0,0,255));
                cv::circle(img, vCurrentKeys[i].pt, 2, cv::Scalar(0,0,255), -1);
                mnTrackedVO++;
            }
        }
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
        cv::putText(img, std::to_string(label), cv::Point(bbox[0], bbox[1]), CV_FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0,255,0), 2);
    }
}

void FrameDrawer::DrawProjectionOnImage(cv::Mat& img) {
    std::vector<std::shared_ptr<g2o::Quadric>> quadrics = mpMap->GetAllQuadric();
    for(const auto& quadric : quadrics){
        Eigen::Matrix4d Twc = mpCurrentFrame->GetPose();
        Eigen::Matrix3d R = Twc.block(0,0,3,3);
        Eigen::Vector3d t = Twc.col(3).head(3);
        g2o::SE3Quat pose(R,t);
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

void FrameDrawer::Update(Tracking* pTracker) {
    std::unique_lock<std::mutex> lk(mMutex);
    pTracker->mCurImg.copyTo(mImg);

    mK = pTracker->GetK();

    mpCurrentFrame = pTracker->mpCurrentFrame;
    mvpObservation = mpCurrentFrame->GetDetectionResults();
    mvCurrentKeys = mpCurrentFrame->mvKeys;
    N = mvCurrentKeys.size();
    mvbMap = std::vector<bool>(N, false);
    mvbVO = std::vector<bool>(N, false);

    if(pTracker->mLastProcessedState == Tracking::OK){
        for(int i=0; i<N; ++i){
            std::shared_ptr<MapPoint> pMP = mpCurrentFrame->mvpMapPoints[i];
            if(pMP){
                if(!mpCurrentFrame->mvbOutlier[i]){
                    if(pMP->Observations()>0)
                        mvbMap[i] = true;
                    else
                        mvbVO[i] = true;
                }
            }
        }
    }

    mState = pTracker->mLastProcessedState;
}