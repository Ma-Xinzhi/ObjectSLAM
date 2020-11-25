#include "MapDrawer.h"

MapDrawer::MapDrawer(const std::string &strSettingPath, std::shared_ptr<Map> pmap): mpMap(pmap) {
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    mKeyFrameSize = fSettings["Viewer.KeyFrameSize"];
    mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"];
    mGraphLineWidth = fSettings["Viewer.GraphLineWidth"];
    mPointSize = fSettings["Viewer.PointSize"];
    mCameraSize = fSettings["Viewer.CameraSize"];
    mCameraLineWidth = fSettings["Viewer.CameraLineWidth"];

    fSettings.release();
}

void MapDrawer::DrawCurrentCamera(const pangolin::OpenGlMatrix& Twc){

    const float &w = mCameraSize;
    const float h = w * 0.75;
    const float z = w * 0.6;

    glPushMatrix();
    glMultMatrixd(Twc.m);

    glLineWidth(mCameraLineWidth);
    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();

}

void MapDrawer::DrawTrajectory() {
    std::vector<std::shared_ptr<KeyFrame>> all_frame = mpMap->GetAllKeyFrame();
    for (const auto& frame : all_frame) {
        Eigen::Matrix4d Twc = frame->GetPose();

        const float &w = mCameraSize;
        const float h = w * 0.75;
        const float z = w * 0.6;

        glPushMatrix();

        glMultMatrixd(Twc.data());
        glLineWidth(mCameraLineWidth);
        glColor3f(0.0f, 1.0f, 0.0f);
        glBegin(GL_LINES);
        glVertex3f(0,0,0);
        glVertex3f(w,h,z);
        glVertex3f(0,0,0);
        glVertex3f(w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,h,z);

        glVertex3f(w,h,z);
        glVertex3f(w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(-w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(w,h,z);

        glVertex3f(-w,-h,z);
        glVertex3f(w,-h,z);
        glEnd();

        glPopMatrix();
    }

    glLineWidth(mGraphLineWidth);
    glColor4f(0.0f,1.0f,0.0f,0.8f);
    glBegin(GL_LINES);

    for (int i = 0; i < all_frame.size()-1; ++i) {
        Eigen::Vector3d Ow1 = all_frame[i]->GetCameraCenter();
        Eigen::Vector3d Ow2 = all_frame[i+1]->GetCameraCenter();
        glVertex3d(Ow1[0], Ow1[1], Ow1[2]);
        glVertex3d(Ow2[0], Ow2[1], Ow2[2]);
    }

    glEnd();
}

void MapDrawer::DrawEllipsoids(){
    std::vector<std::shared_ptr<g2o::Quadric>> quadrics = mpMap->GetAllQuadric();
    if(quadrics.empty())
        return;
    for(const auto& quadric : quadrics){
        g2o::SE3Quat Twm_SE3 = quadric->GetPose();
        Eigen::Matrix4d Twm = Twm_SE3.to_homogeneous_matrix();
        Eigen::Vector3d scale = quadric->GetScale();

        glPushMatrix();
        glLineWidth(mCameraLineWidth*3/4.0);

        glColor3f(0.0f, 0.0f, 1.0f);

        GLUquadricObj *pObj;
        pObj = gluNewQuadric();
        gluQuadricDrawStyle(pObj, GLU_LINE);
        glMultMatrixd(Twm.data());
        glScaled(scale[0], scale[1], scale[2]);

        gluSphere(pObj, 1.0, 26, 13);// draw a sphere with radius 1.0, center (0,0,0), slices 26, and stacks 13.
        DrawAxisNormal();
        glPopMatrix();
    }
}

void MapDrawer::DrawAxisNormal()
{
    float length = 2.0;

    // x
    glColor3f(1.0,0.0,0.0); // red x
    glBegin(GL_LINES);
    glVertex3f(0.0, 0.0f, 0.0f);
    glVertex3f(length, 0.0f, 0.0f);
    glEnd();

    // y
    glColor3f(0.0,1.0,0.0); // green y
    glBegin(GL_LINES);
    glVertex3f(0.0, 0.0f, 0.0f);
    glVertex3f(0.0, length, 0.0f);

    glEnd();

    // z
    glColor3f(0.0,0.0,1.0); // blue z
    glBegin(GL_LINES);
    glVertex3f(0.0, 0.0f ,0.0f );
    glVertex3f(0.0, 0.0f ,length );

    glEnd();
}

void MapDrawer::SetCurrentCameraPose(const Eigen::Matrix4d &pose) {
    std::unique_lock<std::mutex> lk(mMutexCamera);
    mCameraPose = pose;
}

pangolin::OpenGlMatrix MapDrawer::GetCurrentOpenGLMatrix() {
    std::unique_lock<std::mutex> lk(mMutexCamera);
    return {mCameraPose};
}