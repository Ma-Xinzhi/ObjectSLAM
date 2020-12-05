#include "InitializeQuadric.h"

InitializeQuadric::InitializeQuadric(int height, int width): mImageHeight(height), mImageWidth(width){
    mbResult = false;
}

std::shared_ptr<g2o::Quadric> InitializeQuadric::BuildQuadric(const Observations &obs, const Eigen::Matrix3d &calib) {
    std::vector<g2o::SE3Quat> poses;
    std::vector<Eigen::VectorXd> detections;
    for(const auto& ob : obs){
        Eigen::Matrix4d Twc = ob->mpKeyFrame.lock()->GetPose();
        g2o::SE3Quat pose(Twc.block(0,0,3,3), Twc.col(3).head(3));
        Eigen::VectorXd detection;
        detection.resize(6);
        detection << ob->mBbox, ob->mLabel, ob->mProb;
        poses.push_back(pose);
        detections.push_back(detection);
    }
    std::shared_ptr<g2o::Quadric> Q_ptr = BuildQuadric(poses, detections, calib);
    if (mbResult)
        Q_ptr->SetLabel(obs[0]->mLabel);
    return Q_ptr;
}

std::shared_ptr<g2o::Quadric> InitializeQuadric::BuildQuadric(const std::vector<g2o::SE3Quat>& poses,
        const std::vector<Eigen::VectorXd>& detections, const Eigen::Matrix3d &calib) {
//    CHECK_EQ(poses.size(), detections.size());
    CHECK_GT(poses.size(), 2) << "At least 3 measurements are required.";
    int input_size = poses.size();
    Eigen::MatrixXd planesHomo = GetPlanesHomo(poses, detections, calib);
    int plane_size = planesHomo.size();
    // 这里的检测似乎没有必要，在添加检测结果的时候已经做了一次筛选，这里的平面结果肯定是想要得到的
//    int invalid_plane = 4*input_size - plane_size;
//    if(invalid_plane)
//        LOG(INFO) << "Exist Invalid Plane " << invalid_plane << std::endl;
    if(plane_size < 9){
        std::shared_ptr<g2o::Quadric> Q_ptr(new g2o::Quadric);
        mbResult = false;
        return Q_ptr;
    }

    Eigen::MatrixXd planesVector = GetVectorFromPlanesHomo(planesHomo);
    Eigen::Matrix4d QDual = GetQDualFromVectors(planesVector);

    std::shared_ptr<g2o::Quadric> Q_ptr = BuildQuadricFromQDual(QDual);

    return Q_ptr;
}

std::shared_ptr<g2o::Quadric> InitializeQuadric::BuildQuadricFromQDual(const Eigen::Matrix4d &QDual) {
    // TODO 这里的立方根不知道是否会影响最后的结果，还得保证能够进行求逆，求逆后，差一个尺度
//    Eigen::Matrix4d Q = QDual.inverse() * cbrt(QDual.determinant());
    Eigen::Matrix4d Q = QDual.inverse() * QDual.determinant();
//    Eigen::Matrix4d Q = QDual.inverse();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigen_solver(Q);
    Eigen::Vector4d eigen_values = eigen_solver.eigenvalues();

    // 判断一下特征值正负，椭球的特征值必须是 ---+或者+++-
    int num_pos = int(eigen_values[0]>0) + int(eigen_values[1]>0) + int(eigen_values[2]>0) + int(eigen_values[3]>0);
    int num_neg = int(eigen_values[0]<0) + int(eigen_values[1]<0) + int(eigen_values[2]<0) + int(eigen_values[3]<0);
    if(!(num_pos==3 && num_neg==1) && !(num_pos==1 && num_neg==3)){
        LOG(WARNING) << "Not Ellipsoid : pos/neg " << num_pos << "/" << num_neg << std::endl;
        mbResult = false;
        std::shared_ptr<g2o::Quadric> ptr(new g2o::Quadric);
        return ptr;
    }else
        mbResult = true;
    if(num_pos==1 && num_neg==3)
        Q = -Q;

    Eigen::Matrix3d Q_33 = Q.block(0,0,3,3);
    double det = Q.determinant() / Q_33.determinant();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver_33(Q_33);

    Eigen::Matrix3d rotation = eigen_solver_33.eigenvectors();
    //　保证矩阵为旋转矩阵
    if(fabs(1.0 - rotation.determinant()) > 1e-8)
        rotation = -rotation;
    LOG(INFO) << "The Rotation Matrix of the Constrained Quadric is " << std::endl
              << rotation << std::endl;

    Eigen::Vector3d eigen_values_33 = eigen_solver_33.eigenvalues();
    Eigen::Vector3d eigen_values_33_inverse = eigen_values_33.array().inverse();

    Eigen::Vector3d shape = (-det * eigen_values_33_inverse).array().sqrt();
    LOG(INFO) << "The Shape of the Constrained Quadric is " << std::endl
              << shape.transpose() << std::endl;

    Eigen::Vector3d translation;
    translation = (QDual.col(3)/QDual(3,3)).head(3);
    LOG(INFO) << "The Translation of the Constrained Quadric is " << std::endl
              << translation.transpose() << std::endl;

    //　TODO 另外一种，不确定是否正确，和论文中的公式不太一致，自己推导的
    //  答：应该是-Q_33^-1 * Q.col(3).head(3)的方式去计算位移结果，与论文的公式不一致，可能论文的有点问题
//    Eigen::Vector3d translation_self;
//    translation_self = -Q_33.inverse() * Q.col(3).head(3);
//    LOG(INFO) << "The Translation Self of the Constrained Quadric is " << std::endl
//              << translation_self.transpose() << std::endl;
//    Eigen::Vector3d diff = translation.array() - translation_self.array();
//    LOG(INFO) << "The Different translation of the Constrained Quadric is " << std::endl
//              << diff.transpose() << std::endl;

    std::shared_ptr<g2o::Quadric> Q_ptr(new g2o::Quadric(rotation, translation, shape));
    return Q_ptr;

}

Eigen::MatrixXd InitializeQuadric::GetPlanesHomo(const std::vector<g2o::SE3Quat>& poses, const std::vector<Eigen::VectorXd>& detections,
                                                 const Eigen::Matrix3d &calib) {
    Eigen::MatrixXd planes_all(4, 0);
    for(int i=0; i < poses.size(); i++){
        g2o::SE3Quat pose = poses[i];
        Eigen::VectorXd detection = detections[i];
        // 这里的条件设置，可以不用进行考虑，在添加检测框结果的时候已经做了剔除
//        if( abs(detection[2]-detection[0]) < 10 || abs(detection[3]-detection[1]) < 10)
//            continue;
        Eigen::MatrixXd P = GenerateProjectionMatrix(pose, calib);
        Eigen::MatrixXd lines = FromDetectionToLines(detection);

        Eigen::MatrixXd planes = P.transpose() * lines;

        //　用conservativeResize感觉会比较慢
//        for(int m = 0; m < planes.cols(); ++m){
//            planes_all.conservativeResize(planes_all.rows(), planes_all.cols()+1);
//            planes_all.col(planes_all.cols()-1) = planes.col(m);
//        }

        Eigen::MatrixXd temp(planes_all.rows(), planes_all.cols()+planes.cols());
        temp << planes_all, planes;
        planes_all = temp;
    }
    return planes_all;
}

Eigen::MatrixXd InitializeQuadric::FromDetectionToLines(const Eigen::VectorXd &detection) {
    double x1 = detection[0];
    double y1 = detection[1];
    double x2 = detection[2];
    double y2 = detection[3];

    Eigen::Vector3d line1(1, 0, -x1);
    Eigen::Vector3d line2(0, 1, -y1);
    Eigen::Vector3d line3(1, 0, -x2);
    Eigen::Vector3d line4(0, 1, -y2);

    Eigen::MatrixXd lines_selected(3,0);
    // Todo　考虑检测框边界的位置参数，太靠近图像边界的检测框可能不够准
    if(x1 > 0 && x1 < mImageWidth-1){
        lines_selected.conservativeResize(3, lines_selected.cols()+1);
        lines_selected.col(lines_selected.cols()-1) = line1;
    }
    if(y1 > 0 && y1 < mImageHeight-1){
        lines_selected.conservativeResize(3, lines_selected.cols()+1);
        lines_selected.col(lines_selected.cols()-1) = line2;
    }
    if(x2 > 0 && x2 < mImageWidth-1){
        lines_selected.conservativeResize(3, lines_selected.cols()+1);
        lines_selected.col(lines_selected.cols()-1) = line3;
    }
    if(y2 > 0 && y2 < mImageHeight-1){
        lines_selected.conservativeResize(3, lines_selected.cols()+1);
        lines_selected.col(lines_selected.cols()-1) = line4;
    }
    return lines_selected;

}

Eigen::MatrixXd InitializeQuadric::GenerateProjectionMatrix(const g2o::SE3Quat &campose,
                                                             const Eigen::Matrix3d &calib) {
    g2o::SE3Quat campose_cw = campose.inverse();
    Eigen::MatrixXd identity_left(3, 4);
    identity_left.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
    identity_left.col(3) = Eigen::Vector3d(0,0,0);
    Eigen::MatrixXd proj_mat = calib * identity_left * campose_cw.to_homogeneous_matrix();
    return proj_mat;
}

Eigen::MatrixXd InitializeQuadric::GetVectorFromPlanesHomo(const Eigen::MatrixXd &planes) {
    int cols = planes.cols();
    Eigen::MatrixXd planes_vector(10, cols);
    for(int i=0; i<cols; i++){
        Eigen::VectorXd p = planes.col(i);
        Vector10d v;
        v << p(0)*p(0),2*p(0)*p(1),2*p(0)*p(2),2*p(0)*p(3),
             p(1)*p(1),2*p(1)*p(2),2*p(1)*p(3),p(2)*p(2),
             2*p(2)*p(3),p(3)*p(3);
        planes_vector.block(0,i,10,1) = v;
    }
    return planes_vector;
}

Eigen::Matrix4d InitializeQuadric::GetQDualFromVectors(const Eigen::MatrixXd &planevecs) {
    Eigen::MatrixXd P = planevecs.transpose();

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(P, Eigen::ComputeThinU | Eigen::ComputeThinV);

    Eigen::MatrixXd V = svd.matrixV();
    Eigen::VectorXd Q_vec = V.col(V.cols()-1);

    Eigen::Matrix4d QDual;
    QDual << Q_vec[0], Q_vec[1], Q_vec[2], Q_vec[3],
             Q_vec[1], Q_vec[4], Q_vec[5], Q_vec[6],
             Q_vec[2], Q_vec[5], Q_vec[7], Q_vec[8],
             Q_vec[3], Q_vec[6], Q_vec[8], Q_vec[9];

    return QDual;
}