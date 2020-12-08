#ifndef OBJECTSLAM_ORBMATCHER_H
#define OBJECTSLAM_ORBMATCHER_H

#include "MapPoint.h"
#include "KeyFrame.h"
#include "Frame.h"

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

class ORBmatcher{
public:
    ORBmatcher(float nratio = 0.6, bool CheckOri=true);

    // Computes the Hamming distance between two ORB descriptors
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    // Search matches between Frame keypoints and projected MapPoints. Returns number of matches
    // Used to track the local map (Tracking)
    int SearchByProjection(const std::shared_ptr<Frame>& pF, const std::vector<std::shared_ptr<MapPoint>> &vpMapPoints, float th=3);

    // Project MapPoints tracked in last frame into the current frame and search matches.
    // Used to track from previous frame (Tracking)
    int SearchByProjection(const std::shared_ptr<Frame>& pCurrentFrame, const std::shared_ptr<Frame>& pLastFrame, float th, bool bMono);

    // Project MapPoints seen in KeyFrame into the Frame and search matches.
    // Used in relocalisation (Tracking)
    int SearchByProjection(const std::shared_ptr<Frame>& pCurrentFrame, const std::shared_ptr<KeyFrame>& pKF, const std::set<std::shared_ptr<MapPoint>> &sAlreadyFound, float th, int ORBdist);
    int SearchByProjection(const std::shared_ptr<Frame>& pCurrentFrame, const std::shared_ptr<KeyFrame>& pKF, float th=3);

    // Project MapPoints using a Similarity Transformation and search matches.
    // Used in loop detection (Loop Closing)
//    int SearchByProjection(std::shared_ptr<KeyFrame> pKF, cv::Mat Scw, const std::vector<std::shared_ptr<MapPoint>> &vpPoints, std::vector<std::shared_ptr<MapPoint>> &vpMatched, int th);

    // Matching to triangulate new MapPoints. Check Epipolar Constraint.
    int SearchForTriangulation(const std::shared_ptr<KeyFrame>& pKF1, const std::shared_ptr<KeyFrame>& pKF2, const Eigen::Matrix3d& F12,
                               std::vector<std::pair<size_t, size_t>> &vMatchedPairs);

    // Search matches between MapPoints seen in KF1 and KF2 transforming by a Sim3 [s12*R12|t12]
    // In the stereo and RGB-D case, s12=1
    int SearchBySim3(const std::shared_ptr<KeyFrame>& pKF1, const std::shared_ptr<KeyFrame>& pKF2, std::vector<std::shared_ptr<MapPoint>> &vpMatches12, const float &s12, const Eigen::Matrix3d &R12, const Eigen::Vector3d &t12, const float th);

    // Project MapPoints into KeyFrame and search for duplicated MapPoints.
    int Fuse(const std::shared_ptr<KeyFrame>& pKF, const std::vector<std::shared_ptr<MapPoint>> &vpMapPoints, float th=3.0);

    // Project MapPoints into KeyFrame using a given Sim3 and search for duplicated MapPoints.
    int Fuse(const std::shared_ptr<KeyFrame>& pKF, Eigen::Matrix4d Scw, const std::vector<std::shared_ptr<MapPoint>> &vpPoints, float th, std::vector<std::shared_ptr<MapPoint>> &vpReplacePoint);

public:

    static const int TH_LOW;
    static const int TH_HIGH;
    static const int HISTO_LENGTH;


private:

    bool CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const Eigen::Matrix3d &F12, const std::shared_ptr<KeyFrame> pKF);

    float RadiusByViewingCos(float viewCos);

    void ComputeThreeMaxima(std::vector<int>* hist, int L, int &ind1, int &ind2, int &ind3);

    float mfNNratio;
    bool mbCheckOrientation;
};

#endif //OBJECTSLAM_ORBMATCHER_H
