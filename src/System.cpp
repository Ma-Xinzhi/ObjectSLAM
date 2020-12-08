#include "System.h"

System::System(const std::string &strSettingsFile) {
    LOG(INFO) << "ObjectSLAM Project" << std::endl;

    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened()){
        LOG(ERROR) << "Fail to open settings file..." << std::endl;
        fsSettings.release();
        exit(-1);
    }
    fsSettings.release();

    mpMap = std::make_shared<Map>();

    mpFrameDrawer = std::make_shared<FrameDrawer>(mpMap);
    mpMapDrawer = std::make_shared<MapDrawer>(strSettingsFile, mpMap);

    mpTracker = std::make_unique<Tracking>(strSettingsFile, mpMap, mpMapDrawer, mpFrameDrawer);

    mpLocalMapper = std::make_shared<LocalMapping>(mpMap);
    mtLocalMapping = std::thread(&LocalMapping::Run, mpLocalMapper);

    mpViewer = std::make_unique<Viewer>(strSettingsFile, mpMapDrawer, mpFrameDrawer);

    mpTracker->SetLocalMapper(mpLocalMapper);
}

System::~System() {
    mtLocalMapping.join();
}

void System::TrackWithSingleObject(const Eigen::VectorXd &pose, const Eigen::VectorXd &detection, const cv::Mat &im) {
    Eigen::Vector3d t = pose.head(3);
    Eigen::Vector4d q_vec = pose.tail(4);
    Eigen::Quaterniond q(q_vec);
    g2o::SE3Quat pose_se3(q, t);
    std::shared_ptr<Observation> ob_ptr(new Observation);
    ob_ptr->mBbox = detection.head(4);
    ob_ptr->mLabel = detection[4];
    ob_ptr->mProb = detection[5];
    mpTracker->GrabPoseAndSingleObject(pose_se3, ob_ptr, im);
}

void System::TrackWithObjects(const Eigen::VectorXd &pose, const std::vector<Eigen::VectorXd> &detections,
                              const cv::Mat &im) {
    Eigen::Vector3d t = pose.head(3);
    Eigen::Vector4d q_vec = pose.tail(4);
    Eigen::Quaterniond q(q_vec);
    g2o::SE3Quat pose_se3(q, t);
    Observations obs;
    for(int i=0; i<detections.size(); i++){
        Eigen::VectorXd det_vec = detections[i];
        std::shared_ptr<Observation> ob_ptr(new Observation);
        ob_ptr->mBbox = det_vec.head(4);
        ob_ptr->mLabel = det_vec[4];
        ob_ptr->mProb = det_vec[5];
        obs.push_back(ob_ptr);
    }
    // TODO 检测框很多的情况
    mpTracker->GrabPoseAndObjects(pose_se3, obs, im);
}

void System::TrackRGBD(const cv::Mat &img, const cv::Mat &depthmap, const double &timestamp) {
    mpTracker->GrabRGBDImage(img, depthmap, timestamp);
}

void System::ShutDown() {
    mpViewer->RequestFinish();
    while(!mpViewer->IsFinished())
        usleep(5000);
    mpLocalMapper->RequestFinish();
    while(!mpLocalMapper->isFinished())
        usleep(5000);
    pangolin::BindToContext("Map Viewer");
}