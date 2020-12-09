#ifndef OBJECTSLAM_QUADRIC_H
#define OBJECTSLAM_QUADRIC_H

#include "Object.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "utils/matrix_utils.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <memory>

typedef Eigen::Matrix<double, 5, 1> Vector5d;
typedef Eigen::Matrix<double, 9, 1> Vector9d;
typedef Eigen::Matrix<double, 10, 1> Vector10d;

namespace g2o {

    class Quadric {
    public:
        // 默认构造函数
        Quadric();
        //每创建一个就要增加一个idx
        Quadric(const Matrix3d& R, const Vector3d& t, const Vector3d& inputscale);

        // 拷贝函数不需要对id做增加
        Quadric(const Quadric& Q);
        const Quadric& operator=(const Quadric& Q);

        // v = (t1,t2,t3,theta1,theta2,theta3,s1,s2,s3)
        // xyz roll pitch yaw half_scale
        void fromMinimalVector(const Vector9d& v);

        // xyz quaternion,
        void fromVector(const Vector10d& v);

        g2o::SE3Quat GetPose() const { return mPose; }
        Matrix3d GetRotation() const { return mPose.rotation().toRotationMatrix(); }
        Vector3d GetTranslation() const { return mPose.translation(); }
        Vector3d GetScale() const { return mScale; }
        int GetLabel() const { return mLabel; }
        int GetInstanceID() const { return mInstanceID; }


        void SetTranslation(const Vector3d& t_) { mPose.setTranslation(t_); }
        void SetRotation(const Quaterniond& r_) { mPose.setRotation(r_); }
        void SetRotation(const Matrix3d& R) { mPose.setRotation(Quaterniond(R)); }
        void SetScale(const Vector3d& scale_) { mScale = scale_; }
        void SetLabel(const int& label) { mLabel = label; }

        void SetObservation(const Objects& obs) { mvObservations = obs; }
        void AddObservation(const Object& ob) { mvObservations.push_back(ob); }

        // apply update to current quadric, exponential map
        Quadric exp_update(const Vector9d& update);

        // transform a local cuboid to global cuboid  Twc is camera pose. from camera
        // to world
        Quadric transform_from(const SE3Quat& Twc) const;

        // transform a global cuboid to local cuboid  Twc is camera pose. from camera
        // to world
        Quadric transform_to(const SE3Quat& Twc) const;

        // xyz roll pitch yaw half_scale
        Vector9d toRPYVector() const;

        // xyz quaternion, half_scale
        Vector10d toVector() const;

        Matrix4d toSymMatrix() const;

        // get rectangles after projection  [topleft, bottomright]

        Matrix3d toConic(const SE3Quat& campose_wc, const Matrix3d& calib) const;

        /// 两种方法求解椭球在图像中的投影的bounding box都是正确的，数值接近
        /// 但是，还是觉得恢复出2D平面的椭圆参数，会更直观，后续求解边界问题可能需要用到

        // 通过求解一元二次方程的只有唯一解，b^2-4ac=0进行求解
        Vector4d ProjectOntoImageRectByEquation(const SE3Quat& campose_wc, const Matrix3d& calib) const;
        // 通过矩阵参数求解平面椭圆的参数，根据椭圆参数求解外接矩形作为bbox
        Vector4d ProjectOntoImageRectByEllipse(const SE3Quat& campose_wc, const Matrix3d& calib) const;

        Vector5d ProjectOntoImageEllipse(const SE3Quat& campose_wc, const Matrix3d& calib) const;
        Vector4d GetBboxFromEllipse(const Vector5d& ellipse) const;


        bool CheckObservability(const SE3Quat& cam_pose);
    private:
        int mLabel; //Quadric表示的物体类别
        int mInstanceID; //Quadric的自身ID

//        double mdPro;
        SE3Quat mPose; // 从自身坐标系到世界坐标系 Twq
        Eigen::Vector3d mScale;  // semi-axis a,b,c

        Objects mvObservations;

    };



}  // namespace g2o

#endif //OBJECTSLAM_QUADRIC_H
