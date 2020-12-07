#include "Tracking.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include "utils/dataprocess_utils.h"

#include <set>
#include <opencv2/core/eigen.hpp>

Tracking::Tracking(const std::string &strSettingPath, std::shared_ptr<Map> pMap, std::shared_ptr<MapDrawer> pMapDrawer,
             std::shared_ptr<FrameDrawer> pFrameDrawer): mState(NO_IMAGES_YET), mpMap(pMap), mpMapDrawer(pMapDrawer),
             mpFrameDrawer(pFrameDrawer), mnLastRelocFrameId(0), mpCurrentFrame(nullptr), mpLastFrame(nullptr),
             mpLastKeyFrame(nullptr), mpReferenceKF(nullptr), mbVelocity(false){
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    double fx = fSettings["Camera.fx"];
    double fy = fSettings["Camera.fy"];
    double cx = fSettings["Camera.cx"];
    double cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3 != 0){
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mImgHeight = fSettings["Camera.height"];
    mImgWidth = fSettings["Camera.width"];

    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps == 0)
        fps = 30;

    mMinFrames = 0;
    mMaxFrames = (int)fps;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    LOG(INFO) << std::endl  << "ORB Extractor Parameters: " << std::endl
              << "- Number of Features: " << nFeatures << std::endl
              << "- Scale Levels: " << nLevels << std::endl
              << "- Scale Factor: " << fScaleFactor << std::endl
              << "- Initial Fast Threshold: " << fIniThFAST << std::endl
              << "- Minimum Fast Threshold: " << fMinThFAST << std::endl;

    mpInitailizeQuadric = std::make_shared<InitializeQuadric>(mImgHeight, mImgWidth);
    mpORBextractor = std::make_shared<ORBextractor>(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    mThDepth = mbf*(float)fSettings["ThDepth"]/fx;

    mDepthMapFactor = fSettings["DepthMapFactor"];
    if(fabs(mDepthMapFactor) < 1e-5)
        mDepthMapFactor = 1;
    else
        mDepthMapFactor = 1.0f/mDepthMapFactor;

    fSettings.release();
}

// 先根据每一帧数据进行跟踪，给定关键帧条件，设置关键帧
// 对关键帧进行目标检测，检查是否能够初始化物体实例
void Tracking::GrabPoseAndSingleObject(const g2o::SE3Quat &pose, std::shared_ptr<Observation> bbox, const cv::Mat &img_RGB) {
    mCurImg = img_RGB;
    // 这里不能用mpCurrentFrame私有变量来作为Frame对象的shared_ptr
    // 因为只有一个类Tracking，该私有变量指针所指代的对象会发生更换，造成weak_ptr表示的shared_ptr为空
    // 这里主要的问题是该创建的Frame对象的shared_ptr并未存储在map类中，造成删除
    mpCurrentFrame = std::make_shared<Frame>(pose, bbox, img_RGB);

    mpFrameDrawer->Update(this);
    // TODO 这里先简单将位姿结果作为已知，直接进行赋值，后续需要跟踪图像求解位姿
    mpMapDrawer->SetCurrentCameraPose(pose.to_homogeneous_matrix());

    // TODO 增加关键帧的判断，先简单处理为全部是关键帧
//    if(NeedNewKeyFrame()){
//        mpMap->AddKeyFrame(mpCurrentFrame);
//        UpdateObjectObservation();
//        CheckInitialization();
//    }
}

void Tracking::GrabPoseAndObjects(const g2o::SE3Quat &pose, const Observations &bbox, const cv::Mat &img_RGB) {

}

void Tracking::GrabRGBDImage(const cv::Mat &img_RGB, const cv::Mat &depth, double timestamp) {
    mCurImg = img_RGB;
    cv::Mat imDepth = depth;
    if(mCurImg.channels()==3){
        if(mbRGB)
            cv::cvtColor(mCurImg, mGrayImg, CV_RGB2GRAY);
        else
            cv::cvtColor(mCurImg, mGrayImg, CV_BGR2GRAY);
    }
    else if(mCurImg.channels()==4){
        if(mbRGB)
            cv::cvtColor(mCurImg, mGrayImg, CV_RGBA2GRAY);
        else
            cv::cvtColor(mCurImg, mGrayImg, CV_BGRA2GRAY);
    }

    if(fabs(mDepthMapFactor-1.0f)>1e-5 || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth, CV_32F, mDepthMapFactor);

    mpCurrentFrame = std::make_shared<Frame>(mGrayImg, imDepth, timestamp, mpORBextractor, mK, mDistCoef, mbf, mThDepth);

    Track();
}

void Tracking::Track() {
    if(mState == NO_IMAGES_YET)
        mState = NOT_INITIALIZED;

    mLastProcessedState = mState;

    // Map 不能改变
    std::unique_lock<std::mutex> lk(mpMap->mMutexMapUpdate);

    if(mState == NOT_INITIALIZED){
        StereoInitialization();
        // TODO 这里需要进行物体检测,再update viewer
        mpFrameDrawer->Update(this);
        if(mpLastFrame){
            mbVelocity = true;
            mVelocity = Eigen::Matrix4d::Identity();
        }
        if(mState != OK)
            return;
    }
    else{
        bool bOK = false;
        if(mState == OK){
            /// 这里是因为Local Mapping中可能会融合一些地图点
            CheckReplacedInLastFrame();
            // TODO 这里跟踪参考关键帧考虑采用光流法跟踪，作为一个粗略估计
//            if(!mbVelocity || mpCurrentFrame->mnId < mnLastRelocFrameId+2)
//                bOK = TrackReferenceKeyFrame();
//            else{
//                bOK = TrackWithMotionModel();
//                if(!bOK)
//                    bOK = TrackReferenceKeyFrame();
//            }
            // 先按照前后帧进行跟踪
            if(mbVelocity)
                bOK = TrackWithMotionModel();
        }
        else{
            // TODO 重定位方法的考虑
//            bOK = Relocalization();
            LOG(INFO) << "Tracking is lost...." << std::endl;
            return;
        }

        mpCurrentFrame->mpReferenceKF = mpReferenceKF;

        // 前后帧跟踪后，跟踪局部地图
        if(bOK)
            bOK = TrackLocalMap();

        // Update drawer
        mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        if(bOK){
            mState = OK;
            if(mpLastFrame){
                Eigen::Matrix4d Twl = mpLastFrame->GetPose();
                Eigen::Matrix4d Twc = mpCurrentFrame->GetPose();
                mVelocity = Twl.inverse()*Twc;
                mbVelocity = true;
            }
            else{
                mVelocity = Eigen::Matrix4d::Identity();
                mbVelocity = false;
            }

            mpMapDrawer->SetCurrentCameraPose(mpCurrentFrame->GetPose());

            for (int i = 0; i < mpCurrentFrame->N; ++i) {
                std::shared_ptr<MapPoint> pMP = mpCurrentFrame->mvpMapPoints[i];
                if(pMP){
                    if(pMP->Observations() < 1){
                        mpCurrentFrame->mvbOutlier[i] = false;
                        mpCurrentFrame->mvpMapPoints[i] = nullptr;
                    }
                }
            }

            if(NeedNewKeyFrame())
                CreateNewKeyFrame();

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            // 大致意思就是当前帧中的检测出的outlier是否作为outlier，由构建的关键帧local BA决定，所以可以直接构建，传给LocalMapping
            // 为了避免下一帧根据当前帧的outlier点进行跟踪，将其舍弃
            for (int i = 0; i < mpCurrentFrame->N; ++i) {
                if(mpCurrentFrame->mvpMapPoints[i] && mpCurrentFrame->mvbOutlier[i])
                    mpCurrentFrame->mvpMapPoints[i] = nullptr;
            }
        }
        else
            mState = LOST;

        mpLastFrame = mpCurrentFrame;
    }

    // 这里是存储信息，用于后面恢复完整的相机轨迹
    Eigen::Matrix4d Twc = mpCurrentFrame->GetPose();
    mlRelativeFramePoses.emplace_back(mpCurrentFrame->mpReferenceKF->GetPose().inverse()*Twc);
    mlpReferences.push_back(mpReferenceKF);
    mlFrameTimes.push_back(mpCurrentFrame->mTimeStamp);
    mlbLost.push_back(mState == LOST);
}

// 剔除当前帧中不好的检测结果
void Tracking::UpdateObjectObservation() {
    Observations obs = mpCurrentFrame->GetDetectionResults();
    for(auto iter=obs.begin(); iter!=obs.end(); ){
        Eigen::Vector4d measurement = (*iter)->mBbox;
        //TODO 两个参数设置，检测框的大小和距离图像边界距离
        int config_border = 10;
        int config_size = 10;
        if(!calibrateMeasurement(measurement, mImgHeight, mImgWidth, config_border, config_size)){
            obs.erase(iter);
        }else{
//            (*iter)->mpKeyFrame = mpCurrentFrame;
            mpMap->AddObservation(*iter);
            iter++;
        }
    }
    mpCurrentFrame->SetDetectionResults(obs);
}

void Tracking::CheckInitialization() {
    // TODO 参数设置，需要几帧数据进行初始化
    int config_minimum_initialization_frame = 3;
    // TODO 先按照每一个语义类别只有一种物体来计算，认为相同语义类别就是一个物体
    // 这部分数据关联的问题后续再想办法,先实现简单版本
    std::vector<std::shared_ptr<g2o::Quadric>> objects = mpMap->GetAllQuadric();
    std::set<int> exist_objects;
    for(const auto& object : objects){
        exist_objects.insert(object->GetLabel());
    }
    std::map<int, Observations> observations = mpMap->GetAllObservation();
    for(auto iter = observations.begin(); iter != observations.end(); iter++){
        if(exist_objects.find(iter->first) == exist_objects.end()){
            Observations obs = iter->second;
            int frame_num = obs.size();
            if(frame_num < config_minimum_initialization_frame)
                continue;
            LOG(INFO) << "Frame number: " << frame_num << std::endl;

            Eigen::Matrix3d calib;
            cv::cv2eigen(mK, calib);
            std::shared_ptr<g2o::Quadric> Q_ptr = mpInitailizeQuadric->BuildQuadric(obs, calib);

            if(mpInitailizeQuadric->GetResult()){
                Q_ptr->SetObservation(obs);
                for(auto& ob : obs)
                    ob->mpQuadric = Q_ptr;
                mpMap->AddQuadric(Q_ptr);
                LOG(INFO) << std::endl
                          << "-------- INITIALIZE NEW OBJECT BY SVD ---------" << std::endl
                          << "Label ID: " << Q_ptr->GetLabel() << std::endl
                          << "Instance ID: " << Q_ptr->GetInstanceID() << std::endl
                          << "Pose: " << Q_ptr->toVector().transpose() << std::endl
                          << "Scale: " << Q_ptr->GetScale().transpose() << std::endl
                          << std::endl;
            }
        }
    }
}

void Tracking::StereoInitialization() {
    if(mpCurrentFrame->N > 500){
        mpCurrentFrame->SetPose(Eigen::Matrix4d::Identity());

        std::shared_ptr<KeyFrame> pKFinit = std::make_shared<KeyFrame>(mpCurrentFrame, mpMap);

        mpMap->AddKeyFrame(pKFinit);

        for (int i = 0; i < mpCurrentFrame->N; ++i) {
            float z = mpCurrentFrame->mvDepth[i];
            if(z>0){
                Eigen::Vector3d x3D = mpCurrentFrame->UnprojectStereo(i);
                std::shared_ptr<MapPoint> pNewMP = std::make_shared<MapPoint>(x3D, pKFinit, mpMap);
                pNewMP->AddObservation(pKFinit, i);
                pKFinit->AddMapPoint(pNewMP, i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mpCurrentFrame->mvpMapPoints[i] = pNewMP;
            }
        }

        LOG(INFO) << "New map created with " << mpMap->MapPointsInMap() << " points" << std::endl;

        mpLocalMapping->InsertKeyFrame(pKFinit);

        mpCurrentFrame->mpReferenceKF = pKFinit;

        mpLastFrame = mpCurrentFrame;
        mpLastKeyFrame = pKFinit;
        mnLastKeyFrameId = pKFinit->mnId;

        mvpLocalKeyFrames.push_back(pKFinit);
        mvpLocalMapPoints = mpMap->GetAllMapPoints();
        mpReferenceKF = pKFinit;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
        mpMap->mvpKeyFrameOrigins.push_back(pKFinit);

        mpMapDrawer->SetCurrentCameraPose(mpCurrentFrame->GetPose());

        mState = OK;
    }
}

void Tracking::CheckReplacedInLastFrame() {
    for (int i = 0; i < mpLastFrame->N; ++i) {
        std::shared_ptr<MapPoint> pMP = mpLastFrame->mvpMapPoints[i];
        if(pMP)
        {
            std::shared_ptr<MapPoint> pRep = pMP->GetReplaced();
            if(pRep)
                mpLastFrame->mvpMapPoints[i] = pRep;
        }
    }
}
/// 这里有问题，这么选择的话需要知道当前帧的位姿信息，但是当前帧的位姿是不知道的
/// 根据参考关键帧的追踪，其实主要是重定位的时候用到或者前后帧跟踪结果不好
// TODO 不用DBoW进行匹配，可否考虑采用光流法，进行快速的粗略估计
bool Tracking::TrackReferenceKeyFrame() {
//    std::vector<uchar> statue;
//    cv::Mat error;
//    cv::calcOpticalFlowPyrLK(mpReferenceKF)
//    int nmatches = matcher.SearchByProjection(mpCurrentFrame, mpReferenceKF);
//
//    if(nmatches < 15)
//        return false;
//    //先按照上一帧位姿结果给定当前帧初值
//    mpCurrentFrame->SetPose(mpLastFrame->GetPose());
//
//    Optimizer::PoseOptimization(mpCurrentFrame);
//
//    int nmatchesMap = 0;
//    for(int i=0; i<mpCurrentFrame->N; i++){
//
//    }
}

// 根据上一帧的参考关键帧优化的位姿，对上一帧的位姿信息进行更新
// 再根据速度信息，估计当前帧的位姿，进行匹配
bool Tracking::TrackWithMotionModel() {
    ORBmatcher matcher(0.9, true);
    /// 这里的意思应该是关键帧的位姿会进行优化，所以存储相对位姿，再根据优化更新上一帧的位姿，先省略
    UpdateLastFrame();

    LOG(INFO) << "Velocity:\n" << mVelocity << std::endl;

    mpCurrentFrame->SetPose(mpLastFrame->GetPose()*mVelocity);
    // 这里似乎没有必要
    std::fill(mpCurrentFrame->mvpMapPoints.begin(), mpCurrentFrame->mvpMapPoints.end(), nullptr);

    int th = 15;

    int nmatches = matcher.SearchByProjection(mpCurrentFrame, mpLastFrame, th, false);

    if(nmatches < 20){
        std::fill(mpCurrentFrame->mvpMapPoints.begin(), mpCurrentFrame->mvpMapPoints.end(), nullptr);
        nmatches = matcher.SearchByProjection(mpCurrentFrame, mpLastFrame, 2*th, false);
    }

    if(nmatches < 20)
        return false;

    Optimizer::PoseOptimization(mpCurrentFrame);

    int nmatchesMap = 0;
    for(int i=0; i<mpCurrentFrame->N; i++){
        if(mpCurrentFrame->mvpMapPoints[i]){
            if(mpCurrentFrame->mvbOutlier[i]){
                std::shared_ptr<MapPoint> pMP = mpCurrentFrame->mvpMapPoints[i];
                mpCurrentFrame->mvpMapPoints[i] = nullptr;
                mpCurrentFrame->mvbOutlier[i] = false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mpCurrentFrame->mnId;
                nmatches--;
            }
            else if(mpCurrentFrame->mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++;
        }
    }
    LOG(INFO) << "Track " << nmatchesMap << " points from last frame" << std::endl;

    return nmatchesMap >= 10;
}

void Tracking::UpdateLastFrame() {
    std::shared_ptr<KeyFrame> pRef = mpLastFrame->mpReferenceKF;
    Eigen::Matrix4d Trl = mlRelativeFramePoses.back();

    mpLastFrame->SetPose(pRef->GetPose()*Trl);
}

bool Tracking::TrackLocalMap() {
    // 更新当前帧的局部地图信息
    UpdateLocalMap();
    // 搜索局部地图点，能否找到相关联的特征点
    SearchLocalPoints();
    // 根据相关联地图点进行优化
    Optimizer::PoseOptimization(mpCurrentFrame);

    mnMatchesInliers = 0;

    for(int i=0; i<mpCurrentFrame->N; ++i){
        if(mpCurrentFrame->mvpMapPoints[i])
        {
            if(!mpCurrentFrame->mvbOutlier[i])
            {
                mpCurrentFrame->mvpMapPoints[i]->IncreaseFound();
                if(mpCurrentFrame->mvpMapPoints[i]->Observations()>0)
                    mnMatchesInliers++;
            }
        }
    }

    LOG(INFO) << "Track " << mnMatchesInliers << " points in the local map" << std::endl;

    if(mnMatchesInliers < 30)
        return false;
    else
        return true;
}

void Tracking::UpdateLocalMap() {
    // 用于可视化
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

// 更新局部的关键帧，找出与当前帧有共视关系的关键帧，以及与关键帧有共视关系的关键帧
void Tracking::UpdateLocalKeyFrames() {
    std::map<std::shared_ptr<KeyFrame>, int> kfCounter;
    for(int i=0; i<mpCurrentFrame->N; ++i){
        std::shared_ptr<MapPoint> pMP = mpCurrentFrame->mvpMapPoints[i];
        if(pMP){
            if(pMP->isBad())
                pMP = nullptr;
            else{
                std::map<std::shared_ptr<KeyFrame>, size_t> observations = pMP->GetObservations();
                for(auto& ob : observations)
                    kfCounter[ob.first]++;
            }
        }
    }

    if(kfCounter.empty())
        return;

    int max=0;
    std::shared_ptr<KeyFrame> pKFmax = nullptr;

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*kfCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(auto& cnt : kfCounter){
        std::shared_ptr<KeyFrame> pKF = cnt.first;

        if(pKF->isBad())
            continue;

        if(cnt.second > max){
            max = cnt.second;
            pKFmax = pKF;
        }

        mvpLocalKeyFrames.push_back(pKF);
        pKF->mnTrackReferenceForFrame = mpCurrentFrame->mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(auto& pKF : mvpLocalKeyFrames){
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size() > 80)
            break;

        std::vector<std::shared_ptr<KeyFrame>> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(auto& pNeighKF : vNeighs){
            if(!pNeighKF->isBad()){
                if(pNeighKF->mnTrackReferenceForFrame != mpCurrentFrame->mnId){
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame = mpCurrentFrame->mnId;
                    break;
                }
            }
        }

        std::set<std::shared_ptr<KeyFrame>> spChilds = pKF->GetChilds();
        for(auto& pChildKF : spChilds){
            if(!pChildKF->isBad()){
                if(pChildKF->mnTrackReferenceForFrame != mpCurrentFrame->mnId){
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame = mpCurrentFrame->mnId;
                    break;
                }
            }
        }

        std::shared_ptr<KeyFrame> pParent = pKF->GetParent();
        if(pParent){
            if(pParent->mnTrackReferenceForFrame != mpCurrentFrame->mnId){
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame = mpCurrentFrame->mnId;
                break;
            }
        }
    }

    if(pKFmax){
        mpReferenceKF = pKFmax;
        mpCurrentFrame->mpReferenceKF = mpReferenceKF;
    }
}

void Tracking::UpdateLocalPoints() {
    mvpLocalMapPoints.clear();

    for(auto& pKF : mvpLocalKeyFrames){
        std::vector<std::shared_ptr<MapPoint>> vpMPs = pKF->GetMapPointMatches();

        for(auto& pMP : vpMPs){
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame == mpCurrentFrame->mnId)
                continue;
            if(!pMP->isBad()){
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame = mpCurrentFrame->mnId;
            }
        }
    }
}

void Tracking::SearchLocalPoints() {
    // Do not search map points already matched
    for(auto pMP : mpCurrentFrame->mvpMapPoints){
        if(pMP){
            if(pMP->isBad())
                pMP = nullptr;
            else{
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mpCurrentFrame->mnId;
                // 这里指已经找到相关联的点，不需要再进行跟踪
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch = 0;

    for(auto& pMP : mvpLocalMapPoints){
        if(pMP->mnLastFrameSeen == mpCurrentFrame->mnId)
            continue;
        if(pMP->isBad())
            continue;
        if(mpCurrentFrame->isInFrustum(pMP, 0.5)){
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch > 0){
        ORBmatcher matcher(0.8);
        int th = 3;
        int nmatchs = matcher.SearchByProjection(mpCurrentFrame, mvpLocalMapPoints, th);
        LOG(INFO) << "Match Local MapPoints " << nmatchs;
    }
}

bool Tracking::NeedNewKeyFrame() {

    if(mpLocalMapping->isStopped() || mpLocalMapping->StopRequested())
        return false;

    int nKFs = mpMap->KeyFramesInMap();

    if(mpCurrentFrame->mnId < mnLastRelocFrameId+mMaxFrames && nKFs > mMaxFrames)
        return false;

    int nMinObs = 3;
    if(nKFs <= 2)
        nMinObs = 2;
    // 参考关键帧跟踪到观察次数多的点的匹配个数
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    bool bLocalMappingIdle = mpLocalMapping->AcceptKeyFrames();

    int nNonTrackedClose = 0;
    int nTrackedClose = 0;

    for (int i = 0; i < mpCurrentFrame->N; ++i) {
        if(mpCurrentFrame->mvDepth[i]>0 && mpCurrentFrame->mvDepth[i]<mThDepth){
            if(mpCurrentFrame->mvpMapPoints[i] && !mpCurrentFrame->mvbOutlier[i])
                nTrackedClose++;
            else
                nNonTrackedClose++;
        }
    }
    // 跟踪成功和丢失的比例
    bool bNeedToInsertClose = nTrackedClose < 100 && nNonTrackedClose > 70;

    float thRefRatio = 0.75;
    if(nKFs<2)
        thRefRatio = 0.4;

    bool c1a = mpCurrentFrame->mnId >= mnLastKeyFrameId+mMaxFrames;
    bool c1b = mpCurrentFrame->mnId >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle;
    bool c1c = mnMatchesInliers < nRefMatches*0.25 || bNeedToInsertClose;

    bool c2 = (mnMatchesInliers < nRefMatches*thRefRatio || bNeedToInsertClose) && mnMatchesInliers > 15;

    if((c1a||c1c) && c2){
        if(bLocalMappingIdle)
            return true;
        else{
            mpLocalMapping->InterruptBA();
            if(mpLocalMapping->KeyframesInQueue()<3)
                return true;
            else
                return false;
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapping->SetNotStop(true))
        return;

    std::shared_ptr<KeyFrame> pKF = std::make_shared<KeyFrame>(mpCurrentFrame, mpMap);

    mpReferenceKF = pKF;
    mpCurrentFrame->mpReferenceKF = pKF;

    mpLocalMapping->InsertKeyFrame(pKF);

    mpLocalMapping->SetNotStop(false);

    mnLastKeyFrameId = mpCurrentFrame->mnId;
    mpLastKeyFrame = pKF;
}