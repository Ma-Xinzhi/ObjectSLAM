#ifndef OBJECTSLAM_DATASET_H
#define OBJECTSLAM_DATASET_H

#include "Object.h"

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <glog/logging.h>

#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"

class Dataset{
public:
    Dataset(const std::string& datasetdir);
    Dataset(const std::string& datasetdir, const std::string& detectionpath);

    bool LoadDataset();
    void ReadFrame(cv::Mat &rgb, g2o::SE3Quat& pose);

    bool LoadDetection();
    Objects GetDetection();


private:
    std::string mDatasetDir;
    std::string mImageDir;
    std::string mDetectionPath;

};

#endif //OBJECTSLAM_DATASET_H
