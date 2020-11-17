#include "System.h"
#include "utils/matrix_utils.h"
#include "utils/dataprocess_utils.h"

#include <string>
#include <fstream>
#include <sstream>
#include <unistd.h>

#include <glog/logging.h>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv){
    std::string strSettingPath = "../Example/TUM3.yaml";
    std::string dataset_folder = "../Example/dataset/";
    std::string img_folder = dataset_folder + "rgb/";

    LOG(INFO) << "- Settings File: " << strSettingPath << std::endl
              << "- Dataset Folder: " << dataset_folder << std::endl;

    std::string pose_txt = dataset_folder + "groundtruth.txt";
    std::string bbox_txt = dataset_folder + "bbox.txt";

    std::ifstream pose_file(pose_txt);
    std::ifstream bbox_file(bbox_txt);
    if(pose_file.fail() || bbox_file.fail())
        LOG(ERROR) << "Can't open file." << std::endl;

    std::vector<Eigen::VectorXd> poses;
    std::vector<Eigen::VectorXd> detections;
    std::string line_pose, line_bbox;
    while(getline(pose_file, line_pose)){
        std::stringstream ss(line_pose);
        Eigen::VectorXd pose;
        pose.resize(7);
        ss >> pose[0] >> pose[1] >> pose[2] >> pose[3] >> pose[4] >> pose[5] >> pose[6];
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

    int nums = poses.size();
//    int nums = 10;
    LOG(INFO) << "Frame numbers: " << nums << std::endl;
    System SLAM(strSettingPath);

    char index_c[256];
    int begin = 10;
    for (int i = begin; i < nums; ++i) {
        sprintf(index_c, "%04d", i);
        std::string img_name = img_folder + index_c + "_rgb.jpg";
        cv::Mat img = cv::imread(img_name);
        SLAM.TrackWithSingleObject(poses[i], detections[i], img);
//        cv::imshow("Image", img);
//        cv::waitKey(1000);
        usleep(1e6);

    }
    LOG(INFO) << "Finish all data." << std::endl;
    SLAM.ShutDown();

    return 0;
}

