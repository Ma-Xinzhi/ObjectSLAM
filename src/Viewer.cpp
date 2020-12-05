#include "Viewer.h"

#include <unistd.h>
#include <pangolin/pangolin.h>

Viewer::Viewer(const std::string &strSettingPath, std::shared_ptr<MapDrawer> pmapdrawer,
               std::shared_ptr<FrameDrawer> pframedrawer): mpMapDrawer(pmapdrawer),mpFrameDrawer(pframedrawer),
               mbFinishRequested(false), mbFinished(false), mbStopped(false), mbStopRequested(false){
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    float fps = fSettings["Camera.fps"];
    if(fps<1)
        fps=30;
    mT = 1e3/fps;

    mImgHeight = fSettings["Camera.height"];
    mImgWidth = fSettings["Camera.width"];
    if(mImgHeight<1 || mImgWidth<1){
        mImgHeight = 480;
        mImgWidth = 640;
    }

    mViewpointX = fSettings["Viewer.ViewpointX"];
    mViewpointY = fSettings["Viewer.ViewpointY"];
    mViewpointZ = fSettings["Viewer.ViewpointZ"];
    mViewpointF = fSettings["Viewer.ViewpointF"];

    fSettings.release();

    mthread = std::thread(&Viewer::Run, this);
}

Viewer::~Viewer() {
    mthread.join();
}

void Viewer::Run() {
    pangolin::CreateWindowAndBind("ObjectSLAM: Map Viewer", 1024, 768);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);
    pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", true, true);
    pangolin::Var<bool> menuShowPoints("menu.Show Points", true, true);
    pangolin::Var<bool> menuShowGraph("menu.Show Graph", true, true);
    pangolin::Var<bool> menuShowEllipsoids("menu.Show Ellipsoids", true, true);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000),
            pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0, 0,-1,0)
    );

    pangolin::Handler3D handler(s_cam);
    pangolin::View &d_cam = pangolin::Display("cam")
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(&handler);

//    pangolin::View &rgb_img = pangolin::Display("rgb").SetBounds(0,0.3,0.2,0.5,double(mImgWidth)/double(mImgHeight))
//            .SetLock(pangolin::LockLeft, pangolin::LockBottom);

//    bool bFollow = true;
    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();

    pangolin::GlTexture imageTexture(mImgWidth, mImgHeight, GL_RGB, false, 0, GL_BGR, GL_UNSIGNED_BYTE);

    cv::namedWindow("Current Frame");

    while(true){
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        Twc = mpMapDrawer->GetCurrentOpenGLMatrix();

        if(menuFollowCamera)
            s_cam.Follow(Twc);
        d_cam.Activate(s_cam);
        glClearColor(1.0, 1.0, 1.0, 1.0);

//        pangolin::glDrawAxis(0.5);
        mpMapDrawer->DrawCurrentCamera(Twc);
        if(menuShowKeyFrames || menuShowGraph)
            mpMapDrawer->DrawKeyFrames(menuShowKeyFrames, menuShowGraph);
        if(menuShowPoints)
            mpMapDrawer->DrawMapPoints();

        if(menuShowEllipsoids)
            mpMapDrawer->DrawEllipsoids();

        pangolin::FinishFrame();

        cv::Mat img = mpFrameDrawer->DrawFrame();
//        if(!img.empty()){
//            imageTexture.Upload(img.data, GL_BGR, GL_UNSIGNED_BYTE);
//            rgb_img.Activate();
//            glColor3f(1.0,1.0,1.0);
//            imageTexture.RenderToViewportFlipY();
//        }
        if(!img.empty()){
            cv::imshow("Current Frame", img);
            cv::waitKey(mT);
        }
//        std::this_thread::sleep_for(std::chrono::duration<int>(5000));
//        usleep(5000);
        if(Stop()){
            while(IsStopped())
                usleep(3000);
        }

        if(CheckFinish())
            break;
    }
    SetFinish();
}

bool Viewer::IsFinished() {
    std::unique_lock<std::mutex> lc(mMutexFinish);
    return mbFinished;
}

bool Viewer::CheckFinish() {
    std::unique_lock<std::mutex> lc(mMutexFinish);
    return mbFinishRequested;
}

void Viewer::SetFinish() {
    std::unique_lock<std::mutex> lc(mMutexFinish);
    mbFinished = true;
}

void Viewer::RequestFinish() {
    std::unique_lock<std::mutex> lc(mMutexFinish);
    mbFinishRequested = true;
}

void Viewer::RequestStop() {
    std::unique_lock<std::mutex> lc(mMutexStop);
    if(!mbStopped)
        mbStopRequested = true;
}

bool Viewer::IsStopped() {
    std::unique_lock<std::mutex> lc(mMutexStop);
    return mbStopped;
}

bool Viewer::Stop() {
    std::unique_lock<std::mutex> lk(mMutexStop);
    std::unique_lock<std::mutex> lk2(mMutexFinish);

    if(mbFinishRequested)
        return false;
    else if(mbStopRequested){
        mbStopped = true;
        mbStopRequested = false;
        return true;
    }
    return false;
}

void Viewer::Release() {
    std::unique_lock<std::mutex> lc(mMutexStop);
    mbStopped = false;
}
