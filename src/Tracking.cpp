#include "Tracking.h"
#include "ORBmatcher.h"
#include "utils/dataprocess_utils.h"

#include <set>
#include <opencv2/core/eigen.hpp>

Tracking::Tracking(const std::string &strSettingPath, std::shared_ptr<Map> pMap, std::shared_ptr<MapDrawer> pMapDrawer,
             std::shared_ptr<FrameDrawer> pFrameDrawer): mState(NO_IMAGES_YET), mpMap(pMap), mpMapDrawer(pMapDrawer),
             mpFrameDrawer(pFrameDrawer), mnLastRelocFrameId(0), mpCurrentFrame(nullptr), mpLastFrame(nullptr),
             mpLastKeyFrame(nullptr), mpReferenceKF(nullptr){
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    double fx = fSettings["Camera.fx"];
    double fy = fSettings["Camera.fy"];
    double cx = fSettings["Camera.cx"];
    double cy = fSettings["Camera.cy"];
    mCalib << fx, 0, cx,
              0, fy, cy,
              0, 0, 1;
    cv::eigen2cv(mCalib, mK);
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

    mVelocity = Eigen::Matrix4d::Identity();
    mbVelocity = false;

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

    mpFrameDrawer->Update(std::shared_ptr<Tracking>(this));
    // TODO 这里先简单将位姿结果作为已知，直接进行赋值，后续需要跟踪图像求解位姿
    mpMapDrawer->SetCurrentCameraPose(pose);

    // TODO 增加关键帧的判断，先简单处理为全部是关键帧
    if(NeedNewKeyFrame()){
        mpCurrentFrame->SetKeyFrame();
        mpMap->AddKeyFrame(mpCurrentFrame);
        UpdateObjectObservation();
        CheckInitialization();
    }
}

void Tracking::GrabPoseAndObjects(const g2o::SE3Quat &pose, const Observations &bbox, const cv::Mat &img_RGB) {

}

void Tracking::GrabRGBDImageAndSingleObject(const cv::Mat &img_RGB, const cv::Mat &depth,
                                         std::shared_ptr<Observation> bbox) {
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

    mpCurrentFrame = std::make_shared<Frame>(mGrayImg, imDepth, mpORBextractor, mK, mDistCoef, mbf, mThDepth);

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
        mpFrameDrawer->Update(std::shared_ptr<Tracking>(this));
        if(mState != OK)
            return;
    }
    else{
        bool bOK;
        if(mState == OK){
            // TODO 这里是Local Mapping可能会改变一些地图点，需要斟酌一下
            CheckReplacedInLastFrame();

            if(!mbVelocity || mpCurrentFrame->mnId < mnLastRelocFrameId+2)
                bOK = TrackReferenceKeyFrame();
            else{
                bOK = TrackWithMotionModel();
                if(!bOK)
                    bOK = TrackReferenceKeyFrame();
            }
        }
    }

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
            (*iter)->mpKeyFrame = mpCurrentFrame;
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
            std::shared_ptr<g2o::Quadric> Q_ptr = mpInitailizeQuadric->BuildQuadric(obs, mCalib);
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

        mpLastFrame = mpCurrentFrame;
        mpLastKeyFrame = pKFinit;
        mnLastKeyFrameId = pKFinit->mnId;

        mvpLocalKeyFrames.push_back(pKFinit);
        mvpLocalMapPoints = mpMap->GetAllMapPoints();
        mpReferenceKF = pKFinit;
        mpCurrentFrame->mpReferenceKF = pKFinit;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
        mpMap->mvpKeyFrameOrigins.push_back(pKFinit);

        mpMapDrawer->SetCurrentCameraPose(mpCurrentFrame->GetPose());

        mState = OK;
    }
}

void Tracking::CheckReplacedInLastFrame() {
    for (int i = 0; i < mpLastFrame->N; ++i) {
        std::shared_ptr<MapPoint> pMP = mpLastFrame->mvpMapPoints[i];
        if(pMP){
            std::shared_ptr<MapPoint> pRep = pMP->GetReplaced();
            if(pRep)
                mpLastFrame->mvpMapPoints[i] = pRep;
        }
    }
}

bool Tracking::TrackReferenceKeyFrame() {
    ORBmatcher matcher(0.7, true);

    int nmatches = matcher.SearchByProjection(mpCurrentFrame, mpReferenceKF);

    if(nmatches < 15)
        return false;

    mpCurrentFrame->SetPose(mpLastFrame->GetPose());


}

bool Tracking::NeedNewKeyFrame() {
    return true;
}