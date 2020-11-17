#include "Viewer.h"

#include <thread>
#include <unistd.h>
#include <pangolin/pangolin.h>

Viewer::Viewer(const std::string &strSettingPath, std::shared_ptr<MapDrawer> pmapdrawer,
               std::shared_ptr<FrameDrawer> pframedrawer): mpMapDrawer(pmapdrawer),
               mpFrameDrawer(pframedrawer){
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    mImgHeight = fSettings["Camera.height"];
    mImgWidth = fSettings["Camera.width"];

    mViewpointX = fSettings["Viewer.ViewpointX"];
    mViewpointY = fSettings["Viewer.ViewpointY"];
    mViewpointZ = fSettings["Viewer.ViewpointZ"];
    mViewpointF = fSettings["Viewer.ViewpointF"];

    mbFinished = false;
    mbFinishRequested = false;

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

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(640,480,mViewpointF,mViewpointF,320,240,0.2,100),
            pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0, 0,-1,0)
            );
    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);
    pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", true, true);
    pangolin::Var<bool> menuShowEllipsoids("menu.Show Ellipsoids", true, true);

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


    while(true){
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        Twc = mpMapDrawer->GetCurrentOpenGLMatrix();

        if(menuFollowCamera){
            s_cam.Follow(Twc);
        }
        d_cam.Activate(s_cam);
        glClearColor(1.0, 1.0, 1.0, 1.0);

        pangolin::glDrawAxis(0.5);

        if(menuShowKeyFrames){
            mpMapDrawer->DrawCurrentCamera(Twc);
            mpMapDrawer->DrawTrajectory();
        }

        if(menuShowEllipsoids)
            mpMapDrawer->DrawEllipsoids();

        cv::Mat img = mpFrameDrawer->DrawFrameAll();
//        if(!img.empty()){
//            imageTexture.Upload(img.data, GL_BGR, GL_UNSIGNED_BYTE);
//            rgb_img.Activate();
//            glColor3f(1.0,1.0,1.0);
//            imageTexture.RenderToViewportFlipY();
//        }
        if(!img.empty()){
            cv::imshow("Current Image", img);
            cv::waitKey(100);
        }

        pangolin::FinishFrame();

//        std::this_thread::sleep_for(std::chrono::duration<int>(5000));
//        usleep(5000);
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
