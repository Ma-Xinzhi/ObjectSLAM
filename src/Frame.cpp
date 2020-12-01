#include "Frame.h"
#include "Observation.h"
#include "MapPoint.h"
#include "KeyFrame.h"

#include <opencv2/imgproc.hpp>

long unsigned int Frame::nNextId = 0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame(const cv::Mat &imGray, const cv::Mat &depth, double timeStamp, std::shared_ptr<ORBextractor> extractor,
             const cv::Mat &K, const cv::Mat &distCoef, float bf, float thDepth): mpORBextractor(extractor),
             mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth){
    mnId = nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractor->GetLevels();
    mfScaleFactor = mpORBextractor->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractor->GetScaleFactors();
    mvInvScaleFactors = mpORBextractor->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractor->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractor->GetInverseScaleSigmaSquares();

    ExtractORB(imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(depth);

    mvpMapPoints = std::vector<std::shared_ptr<MapPoint>>(N, nullptr);
    mvbOutlier = std::vector<bool>(N, false);

    if(mbInitialComputations){
        ComputeImageBounds(imGray);

        mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations = false;
    }

    mb = mbf / fx;

    mpReferenceKF = nullptr;

    //TODO 这里如果是Keyframe的话，将目标检测的信息考虑进来(答，在KeyFrame里面做)
    AssignFeaturesToGrid();
}

Frame::Frame(const g2o::SE3Quat& pose, const std::vector<std::shared_ptr<Observation>>& bbox, const cv::Mat& Image):
    mTwc(pose), mvpObservation(bbox), mFrameImg(Image){
    mnId = nNextId++;
}

Frame::Frame(const g2o::SE3Quat& pose, std::shared_ptr<Observation> bbox, const cv::Mat& Image):
    mTwc(pose), mFrameImg(Image){
    static int id = 0;
    mnId = id++;
    mvpObservation.push_back(bbox);
}

void Frame::ExtractORB(const cv::Mat &img) {
    (*mpORBextractor)(img, cv::Mat(), mvKeys, mDescriptors);
}

void Frame::SetPose(const Eigen::Matrix4d &pose_wc) {
    mTwc = pose_wc;
    mtwc = mTwc.col(3).head(3);
    mRwc = mTwc.block(0,0,3,3);
}

// r是搜索的范围
std::vector<size_t> Frame::GetFeaturesInArea(float x, float y, float r, const int minLevel, const int maxLevel) const {
    std::vector<size_t> vIndices;
    vIndices.reserve(N);

    int nMinCellX = std::max(0, (int)floor((x-r-mnMinX)*mfGridElementWidthInv));
    if(nMinCellX >= FRAME_GRID_COLS)
        return vIndices;

    int nMaxCellX = std::min(FRAME_GRID_COLS-1, (int)ceil((x+r-mnMinX)*mfGridElementWidthInv));
    if(nMaxCellX < 0)
        return vIndices;

    int nMinCellY = std::max(0, (int)floor((y-r-mnMinY)*mfGridElementHeightInv));
    if(nMinCellY >= FRAME_GRID_ROWS)
        return vIndices;

    int nMaxCellY = std::min(0, (int)ceil((y+r-mnMinY)*mfGridElementHeightInv));
    if(nMaxCellY < 0)
        return vIndices;

    bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix=nMinCellX; ix<=nMaxCellX; ix++){
        for(int iy=nMinCellY; iy<=nMaxCellY; iy++){
            std::vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;
            for(auto& index : vCell){
                cv::KeyPoint kpUn = mvKeysUn[index];
                if(bCheckLevels){
                    if(kpUn.octave < minLevel)
                        continue;
                    if(maxLevel >= 0)
                        if(kpUn.octave > maxLevel)
                            continue;
                }
                float distx = kpUn.pt.x-x;
                float disty = kpUn.pt.y-y;
                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(index);
            }
        }
    }

    return vIndices;
}

bool Frame::isInFrustum(std::shared_ptr<MapPoint> pMP, float viewingCosLimit) {
    pMP->mbTrackInView = false;

    Eigen::Vector3d P = pMP->GetWorldPos();

    Eigen::Vector3d Pc = mRwc.transpose()*P-mRwc.transpose()*mtwc;

    float PcX = Pc[0];
    float PcY = Pc[1];
    float PcZ = Pc[2];

    if(PcZ < 0)
        return false;

    float invZ = 1/PcZ;

    float u = fx*PcX*invZ + cx;
    float v = fy*PcY*invZ + cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    float maxDistance = pMP->GetMaxDistanceInvariance();
    float minDistance = pMP->GetMinDistanceInvariance();

    float dist = Pc.norm();

    if(dist<minDistance || dist>maxDistance)
        return false;

    Eigen::Vector3d Pn = pMP->GetNormal();

    float viewCos = Pc.dot(Pn)/dist;

    if(viewCos < viewingCosLimit)
        return false;

    int nPredictedLevel = pMP->PredictScale(dist, std::shared_ptr<Frame>(this));

    // 在Tracking中使用的数据
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u-mbf*invZ;
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel = nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}

Eigen::Vector3d Frame::UnprojectStereo(int i) {
    float z = mvDepth[i];
    if(z>0){
        Eigen::Vector3d Point;
        cv::KeyPoint kp = mvKeysUn[i];
        Point[0] = (kp.pt.x-cx)*z*invfx;
        Point[1] = (kp.pt.y-cy)*z*invfy;
        Point[2] = z;
        return mRwc*Point+mtwc;
    }
    else
        return {};
}

void Frame::UndistortKeyPoints() {
    if(mDistCoef.at<float>(0) == 0){
        mvKeysUn = mvKeys;
        return;
    }

    cv::Mat mat(N, 2, CV_32F);
    for(int i=0; i<N; i++){
        mat.at<float>(i, 0) = mvKeys[i].pt.x;
        mat.at<float>(i, 1) = mvKeys[i].pt.y;
    }

    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
    mat = mat.reshape(1);

    mvKeysUn.resize(N);
    for(int i=0; i<N; i++){
        cv::KeyPoint kp;
        kp.pt.x = mat.at<float>(i, 0);
        kp.pt.y = mat.at<float>(i, 1);
        mvKeysUn[i] = kp;
    }
}

void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth) {
    mvuRight = std::vector<float>(N, -1);
    mvDepth = std::vector<float>(N, -1);

    // 求解右图中的点横坐标用的是去畸变的
    // 深度值还是使用的原图像，原因可能是像素之间是对应关系
    for (int i = 0; i < N; ++i) {
        cv::KeyPoint kp = mvKeys[i];
        cv::KeyPoint kpU = mvKeysUn[i];
        float d = imDepth.at<float>(kp.pt.y, kp.pt.x);
        if(d>0){
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

void Frame::ComputeImageBounds(const cv::Mat &img) {
    if(mDistCoef.at<float>(0) != 0){
        cv::Mat mat(4, 2, CV_32F);
        mat.at<float>(0,0) = 0.0f; mat.at<float>(0,1) = 0.0f;
        mat.at<float>(1,0) = img.cols; mat.at<float>(1,1) = 0.0f;
        mat.at<float>(2,0) = img.rows; mat.at<float>(2,1) = 0.0f;
        mat.at<float>(3,0) = img.rows; mat.at<float>(3,1) = img.cols;

        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
        mat = mat.reshape(1);

        mnMinX = std::min(mat.at<float>(0,0), mat.at<float>(2,0));
        mnMaxX = std::min(mat.at<float>(1,0), mat.at<float>(3,0));
        mnMinY = std::min(mat.at<float>(0,1), mat.at<float>(1,1));
        mnMaxY = std::min(mat.at<float>(2,1), mat.at<float>(3,1));
    }
    else{
        mnMinX = 0.0f;
        mnMaxX = img.cols;
        mnMinY = 0.0f;
        mnMaxY = img.rows;
    }
}

void Frame::AssignFeaturesToGrid() {
    int nReserve = N / (FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(int i = 0; i < FRAME_GRID_COLS; i++)
        for (int j = 0; j < FRAME_GRID_ROWS; ++j)
            mGrid[i][j].reserve(nReserve);
    for (int i = 0; i < N; ++i) {
        cv::KeyPoint kp = mvKeys[i];
        int nGridPosX, nGridPosY;
        if(PosInGrid(kp, nGridPosX, nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY) {
    posX = std::round((kp.pt.x-mnMinX) * mfGridElementWidthInv);
    posY = std::round((kp.pt.y-mnMinY) * mfGridElementHeightInv);
    if(posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
        return false;
    return true;
}
