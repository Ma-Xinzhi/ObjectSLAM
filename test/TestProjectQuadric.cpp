#include "Quadric.h"

#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <glog/logging.h>



int main(int argc, char** argv){
    std::string strSettingPath = "../Example/TUM3.yaml";
    std::string dataset_folder = "../Example/dataset/";
    std::string img_folder = dataset_folder + "rgb/";

    std::string pose_txt = dataset_folder + "groundtruth.txt";
    std::string bbox_txt = dataset_folder + "bbox.txt";

    std::ifstream pose_file(pose_txt);
    std::ifstream bbox_file(bbox_txt);
    if(pose_file.fail() || bbox_file.fail())
        LOG(ERROR) << "Can't open file." << std::endl;

    std::vector<g2o::SE3Quat> poses;
    std::vector<Eigen::VectorXd> detections;
    std::string line_pose, line_bbox;
    while(getline(pose_file, line_pose)){
        std::stringstream ss(line_pose);
        Eigen::Vector3d t;
        Eigen::Vector4d q_v;
        ss >> t[0] >> t[1] >> t[2] >> q_v[0] >> q_v[1] >> q_v[2] >> q_v[3];
        Eigen::Quaterniond q(q_v);
        g2o::SE3Quat pose(q, t);
        poses.push_back(pose);
    }
    while(getline(bbox_file, line_bbox)){
        std::stringstream ss(line_bbox);
        Eigen::VectorXd detection;
        detection.resize(6);
        ss >> detection[0] >> detection[1] >> detection[2] >> detection[3] >> detection[4] >> detection[5];
        detections.push_back(detection);
    }
    CHECK_EQ(poses.size(), detections.size());

    cv::FileStorage fs(strSettingPath, cv::FileStorage::READ);
    double fx = fs["Camera.fx"];
    double fy = fs["Camera.fy"];
    double cx = fs["Camera.cx"];
    double cy = fs["Camera.cy"];

    Eigen::Matrix3d Calib = Eigen::Matrix3d::Identity();
    Calib(0,0) = fx;
    Calib(0,2) = cx;
    Calib(1,1) = fy;
    Calib(1,2) = cy;

    Eigen::Vector3d t;
    t << -1.58291, 0.489966, 0.315858;
    Eigen::Vector3d scale;
    scale << 0.505595, 0.4118, 0.256124;
    Eigen::Quaterniond q(0.00156136, -0.147223, 0.101635, 0.983866);
    Eigen::Matrix3d R = q.toRotationMatrix();
    std::cout << "Before R:\n" << R << std::endl;

    g2o::Quadric quadric_test(R, t, scale);
    std::cout << "Before Q* = \n" << quadric_test.toSymMatrix() << std::endl;


    double tmp = scale[0];
    scale[0] = scale[2];
    scale[2] = tmp;

    std::cout << "After scale: " << scale.transpose() << std::endl;

    Eigen::Vector3d v_tmp = R.col(0);
    R.col(0) = R.col(2);
    R.col(2) = v_tmp;

    if(fabs(1.0 - R.determinant()) > 1e-8)
        R = -R;

    std::cout << "After R:\n" << R << std::endl;

    g2o::Quadric quadric(R, t, scale);

    std::cout << "After Q* = \n" << quadric.toSymMatrix() << std::endl;

    int nums = poses.size();
//    int nums = 10;
    LOG(INFO) << "Frame numbers: " << nums << std::endl;

    char index_c[256];
    int begin = 0;
    for (int i = begin; i < nums; ++i) {
        sprintf(index_c, "%04d", i);
        std::string img_name = img_folder + index_c + "_rgb.jpg";
        cv::Mat img = cv::imread(img_name);
        Eigen::VectorXd det = detections[i];
        cv::rectangle(img, cv::Point(det[0], det[1]), cv::Point(det[2], det[3]), cv::Scalar(0,0,255), 2);
        Vector5d ellipse = quadric.ProjectOntoImageEllipse(poses[i], Calib);
        LOG(INFO) << "Ellipse: " << ellipse.transpose() << std::endl;
        double angle = ellipse[4] * 180 / M_PI;
        cv::ellipse(img, cv::Point(ellipse[0], ellipse[1]), cv::Size(ellipse[2], ellipse[3]),
                   -angle, 0, 360, cv::Scalar(0,255,0), 2);
        Eigen::Vector4d rect1 = quadric.GetBboxFromEllipse(ellipse);
        cv::rectangle(img, cv::Point(rect1[0], rect1[1]), cv::Point(rect1[2], rect1[3]), cv::Scalar(0,255,0), 2);
        LOG(INFO) << "Rectangle by ellipse: " << rect1.transpose() << std::endl;
        Eigen::Vector4d rect2 = quadric.ProjectOntoImageRectByEquation(poses[i], Calib);
        cv::rectangle(img, cv::Point(rect2[0], rect2[1]), cv::Point(rect2[2], rect2[3]), cv::Scalar(255,0,0), 2);
        LOG(INFO) << "Rectangle by equation: " << rect2.transpose() << std::endl;
        cv::imshow("Projection Image", img);
        cv::waitKey();

    }
    LOG(INFO) << "Finish Projection." << std::endl;
    fs.release();
    return 0;
}
