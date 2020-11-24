#ifndef OBJECTSLAM_FRAME_H
#define OBJECTSLAM_FRAME_H

#include "ORBextractor.h"
#include "Thirdparty/g2o/g2o/types/se3quat.h"

#include <opencv2/core.hpp>
#include <memory>
#include <vector>

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

struct Observation;
class MapPoint;

class Frame {
public:

    Frame(const cv::Mat& imGray, const cv::Mat& depth, double timeStamp, std::shared_ptr<ORBextractor> extractor,
          const cv::Mat& K, const cv::Mat& distCoef, float bf, float thDepth);
    Frame(const g2o::SE3Quat& pose, std::shared_ptr<Observation> bbox, const cv::Mat& image);
    Frame(const g2o::SE3Quat& pose, const std::vector<std::shared_ptr<Observation>>& bbox, const cv::Mat& image);

    void ExtractORB(const cv::Mat &img);

    void SetKeyFrame();

    std::vector<size_t> GetFeaturesInArea(float x, float y, float r, const int minLevel=-1, const int maxLevel=-1) const;

    void SetDetectionResults(const std::vector<std::shared_ptr<Observation>>& detection_results) { mvpObservation = detection_results; }
    std::vector<std::shared_ptr<Observation>> GetDetectionResults() const{ return mvpObservation; }

//    void SetPose(const g2o::SE3Quat& pose_wc) { mTwc_se3 = pose_wc; }
    void SetPose(const Eigen::Matrix4d& pose_wc);
//    g2o::SE3Quat GetPose() const { return mTwc_se3; }
    Eigen::Matrix4d GetPose() const { return mTwc; }
    Eigen::Vector3d GetCameraCenter() const { return mtwc; }
    Eigen::Matrix3d GetRotation() const { return mRwc; }
    Eigen::Matrix3d GetTranslation() const { return mRwc; }

    cv::Mat GetImg() const { return mFrameImg; }

public:
    static long unsigned int nNextId;
    long unsigned int mnId;  // image topic sequence id, fixed

    cv::Mat mFrameImg;

    double mTimeStamp;

    float mbf;
    float mb;
    float mThDepth;
    int N;

    std::vector<cv::KeyPoint> mvKeys, mvKeysRight;
    std::vector<cv::KeyPoint> mvKeysUn;

    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    std::vector<std::shared_ptr<MapPoint>> mvpMapPoints;
    std::vector<bool> mvbOutlier;

    cv::Mat mDescriptors;

    std::vector<float> mvuRight;
    std::vector<float> mvDepth;

    std::shared_ptr<ORBextractor> mpORBextractor;

    cv::Mat mK;
    cv::Mat mDistCoef;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;

    // Undistorted Image Bounds (computed once).
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;
    // 计算一次
    static bool mbInitialComputations;

    // Scale pyramid info.
    int mnScaleLevels;
    float mfScaleFactor;
    float mfLogScaleFactor;
    std::vector<float> mvScaleFactors;
    std::vector<float> mvInvScaleFactors;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

private:
    void ComputeStereoFromRGBD(const cv::Mat& imDepth);
    void AssignFeaturesToGrid();
    void UndistortKeyPoints();
    void ComputeImageBounds(const cv::Mat& img);

    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    std::vector<std::shared_ptr<Observation>> mvpObservation; // object detection result
    g2o::SE3Quat mTwc_se3;  // optimized pose  cam to world

    Eigen::Matrix4d mTwc;
    Eigen::Matrix3d mRwc;
    Eigen::Vector3d mtwc;
};



#endif //OBJECTSLAM_FRAME_H
