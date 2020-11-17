#include "Quadric.h"

namespace g2o{

    Quadric::Quadric() {
        mPose = SE3Quat();
        mScale.setZero();
        mLabel = -1;
        mInstanceID = -1;
    }

    Quadric::Quadric(const Matrix3d& R, const Vector3d& t, const Vector3d& inputscale)
        :mScale(inputscale){
        static int TotalID = 0;
        mPose = SE3Quat(R, t);
        mInstanceID = TotalID++;
        mLabel = -1;
    }

    Quadric::Quadric(const g2o::Quadric &Q) {
        mLabel = Q.mLabel;
        mInstanceID = Q.mInstanceID;
        mPose = Q.mPose;
        mScale = Q.mScale;
    }

    const Quadric& Quadric::operator=(const g2o::Quadric &Q) {
        mLabel = Q.mLabel;
        mInstanceID = Q.mInstanceID;
        mPose = Q.mPose;
        mScale = Q.mScale;
        return Q;
    }

    // v = (t1,t2,t3,theta1,theta2,theta3,s1,s2,s3)
    // xyz roll pitch yaw half_scale
    void Quadric::fromMinimalVector(const Vector9d& v) {
        Eigen::Quaterniond posequat = zyx_euler_to_quat(v(3), v(4), v(5));
        mPose = SE3Quat(posequat, v.head<3>());
        mScale = v.tail<3>();
    }
    // xyz quaternion half_scale
    void Quadric::fromVector(const Vector10d& v) {
        mPose.fromVector(v.head(7));
        mScale = v.tail(3);
    }

    // apply update to current quadric, exponential map
    Quadric Quadric::exp_update(const Vector9d& update) {
        Quadric res(*this);
        res.mPose = res.mPose * SE3Quat::exp(update.head<6>());
        res.mScale = res.mScale + update.tail<3>();
        return res;
    }

    // transform a local ellipsoid to global ellipsoid  Twc is camera pose. from camera
    // to world
    Quadric Quadric::transform_from(const SE3Quat& Twc) const {
        Quadric res;
        res.mPose = Twc * this->mPose;
        res.mScale = this->mScale;
        return res;
    }

    // transform a global ellipsoid to local ellipsoid Twc is camera pose. from camera
    // to world
    Quadric Quadric::transform_to(const SE3Quat& Twc) const {
        Quadric res;
        res.mPose = Twc.inverse() * this->mPose;
        res.mScale = this->mScale;
        return res;
    }

    // xyz roll pitch yaw half_scale
    Vector9d Quadric::toRPYVector() const{
        Vector9d v;
        v.head<6>() = mPose.toXYZPRYVector();
        v.tail<3>() = mScale;
        return v;
    }

    // xyz quaternion, half_scale
    Vector10d Quadric::toVector() const{
        Vector10d v;
        v.head<7>() = mPose.toVector();
        v.tail<3>() = mScale;
        return v;
    }

    Matrix4d Quadric::toSymMatrix() const{
        Matrix4d SymMat;
        Matrix4d centreAtOrigin;
        centreAtOrigin = Eigen::Matrix4d::Identity();
        centreAtOrigin(0, 0) = pow(mScale(0), 2);
        centreAtOrigin(1, 1) = pow(mScale(1), 2);
        centreAtOrigin(2, 2) = pow(mScale(2), 2);
        centreAtOrigin(3, 3) = -1;
        Matrix4d Z;
        Z = mPose.to_homogeneous_matrix();
        SymMat = Z * centreAtOrigin * Z.transpose();
        return SymMat;
    }

    // get rectangles after projection  [topleft, bottomright]

    Matrix3d Quadric::toConic(const SE3Quat& campose_wc, const Matrix3d& calib) const{
        Eigen::Matrix<double, 3, 4> pose_cw;
        pose_cw.block(0, 0, 3, 3) = campose_wc.inverse().rotation().toRotationMatrix();
        pose_cw.col(3).head(3) = campose_wc.inverse().translation();
        Eigen::Matrix<double, 3, 4> P = calib * pose_cw;
        Matrix4d symMat = toSymMatrix();
        Matrix3d conic_star = P * symMat * P.transpose();
        return conic_star.inverse();
    }

    Vector4d Quadric::ProjectOntoImageRectByEquation(const SE3Quat& campose_wc, const Matrix3d& calib) const{
        //    std::cout << "projectOntoImageRect" << std::endl;
        Matrix3d conic = toConic(campose_wc, calib);
        // TODO 归一化？？感觉不会有所影响
        conic = conic / conic(2, 2);
        Vector6d c;
        c << conic(0, 0), conic(0, 1) * 2, conic(1, 1), conic(0, 2) * 2, conic(1, 2) * 2, conic(2, 2);
        Vector2d y, x;

        /// 这里计算的是二次曲线的外接矩形，具体计算思路是
        /// 将x或者y作为一个已知量，另外一个作为参数
        /// 根据矩形和椭圆只有一个交点，b^2 - 4ac = 0建立方程求解
        y(0) = (4 * c(4) * c(0) - 2 * c(1) * c(3) +
                sqrt(pow(2 * c(1) * c(3) - 4 * c(0) * c(4), 2) -
                     4 * (pow(c(1), 2) - 4 * c(0) * c(2)) *
                     (pow(c(3), 2) - 4 * c(0) * c(5)))) /
               (2 * (pow(c(1), 2) - 4 * c(2) * c(0)));

        y(1) = (4 * c(4) * c(0) - 2 * c(1) * c(3) -
                sqrt(pow(2 * c(1) * c(3) - 4 * c(0) * c(4), 2) -
                     4 * (pow(c(1), 2) - 4 * c(0) * c(2)) *
                     (pow(c(3), 2) - 4 * c(0) * c(5)))) /
               (2 * (pow(c(1), 2) - 4 * c(2) * c(0)));

        x(0) = (4 * c(3) * c(2) - 2 * c(1) * c(4) +
                sqrt(pow(2 * c(1) * c(4) - 4 * c(2) * c(3), 2) -
                     4 * (pow(c(1), 2) - 4 * c(0) * c(2)) *
                     (pow(c(4), 2) - 4 * c(2) * c(5)))) /
               (2 * (pow(c(1), 2) - 4 * c(2) * c(0)));

        x(1) = (4 * c(3) * c(2) - 2 * c(1) * c(4) -
                sqrt(pow(2 * c(1) * c(4) - 4 * c(2) * c(3), 2) -
                     4 * (pow(c(1), 2) - 4 * c(0) * c(2)) *
                     (pow(c(4), 2) - 4 * c(2) * c(5)))) /
               (2 * (pow(c(1), 2) - 4 * c(2) * c(0)));
        // TODO: conic at boundary, 如果椭圆不是完整的话，要如何进行处理
        Vector2d bottomright;  // x y
        Vector2d topleft;
        bottomright(0) = x.maxCoeff();
        bottomright(1) = y.maxCoeff();
        topleft(0) = x.minCoeff();
        topleft(1) = y.minCoeff();

        return Vector4d(topleft(0), topleft(1), bottomright(0), bottomright(1));
    }

    // get rectangles after projection  [center, width, height]
    Vector4d Quadric::ProjectOntoImageRectByEllipse(const SE3Quat& campose_wc, const Matrix3d& calib) const{
        Vector5d ellipse = ProjectOntoImageEllipse(campose_wc, calib);
        return GetBboxFromEllipse(ellipse);
    }

    Vector5d Quadric::ProjectOntoImageEllipse(const g2o::SE3Quat &campose_wc, const Eigen::Matrix3d &calib) const {
        Matrix3d conic = toConic(campose_wc, calib);
        // TODO 归一化？？感觉不会有所影响
//        conic = conic / conic(2, 2);

        // matrix to equation coefficients: ax^2+bxy+cy^2+dx+ey+f=0
        double a = conic(0,0);
        double b = conic(0,1)*2;
        double c = conic(1,1);
        double d = conic(0,2)*2;
        double e = conic(2,1)*2;
        double f = conic(2,2);

        // 因为a和b的大小关系，存在两组解
        // 为方便求解，这里的公式默认b>a，这样求解的theta数值就是最终的结果，不需要+Pi/2的处理
        // get x_c, y_c, axis_x, axis_y, theta from coefficients
        double theta = 1 / 2.0 * atan2(b, a-c);
//        double theta = 1 / 2.0 * atan2(b, a-c)+M_PI/2;
        double x_c = (b*e - 2*c*d) / (4*a*c-b*b);
        double y_c = (b*d - 2*a*e) / (4*a*c-b*b);
//        double t_x = x_c*cos(theta)+y_c*sin(theta);
//        double t_y = -x_c*sin(theta)+y_c*cos(theta);
        double a_2 =  2*(a*x_c*x_c+c*y_c*y_c+b*x_c*y_c - f) /(a + c + sqrt((a-c)*(a-c)+b*b));
//        double a_2 =  2*(a*x_c*x_c+c*y_c*y_c+b*x_c*y_c - f) / (2*a + b*tan(theta));
//        double a_2 =  t_x*t_x+(t_y*t_y*(2*c-b*tan(theta))- 2*f) / (2*a + b*tan(theta));
        double b_2 =  2*(a*x_c*x_c+c*y_c*y_c+b*x_c*y_c - f) /( a + c - sqrt((a-c)*(a-c)+b*b));
//        double b_2 =  2*(a*x_c*x_c+c*y_c*y_c+b*x_c*y_c - f) / (2*c-b*tan(theta));
//        double b_2 =  t_y*t_y+(t_x*t_x*(2*a+b*tan(theta))- 2*f) / (2*c-b*tan(theta));
        double axis_x = sqrt(a_2);
        double axis_y = sqrt(b_2);

        Vector5d ellipse;
        ellipse << x_c, y_c, axis_x, axis_y, theta;

        return ellipse;
    }

    Vector4d Quadric::GetBboxFromEllipse(const Vector5d &ellipse) const {
        // TODO: conic at boundary, 如果椭圆不是完整的话，要如何进行处理
        double x = ellipse[0];
        double y = ellipse[1];
        double a = ellipse[2];
        double b = ellipse[3];
        double theta = ellipse[4];

        double cos_theta_2 = cos(theta) * cos(theta);
        double sin_theta_2 = 1 - cos_theta_2;

        double x_limit = sqrt(a*a*cos_theta_2 + b*b*sin_theta_2);
        double y_limit = sqrt(a*a*sin_theta_2 + b*b*cos_theta_2);

        Vector4d bbox;
        bbox << x-x_limit, y-y_limit, x+x_limit, y+y_limit;

        return bbox;
    }

    bool Quadric::CheckObservability(const g2o::SE3Quat &cam_pose) {
        // 蠢！！！求取的是相机坐标系下的向量，不是世界坐标系下的向量
        Vector3d quadric_center = mPose.translation();
        Vector3d camera_center = cam_pose.translation();

        Vector3d quadric_in_camera = cam_pose.inverse() * (quadric_center - camera_center);
        if(quadric_in_camera[2] < 0)
            return false;
        else
            return true;
        // 也可以这样求解，这样求解比较直观吧，直接在位姿T阶段进行求解
//        g2o::SE3Quat rpose = cam_pose.inverse() * mPose;
//        if(rpose.translation()[2] < 0)
//            return false;
//        else
//            return true;

    }
} //g2o