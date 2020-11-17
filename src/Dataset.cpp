#include "Dataset.h"

Dataset::Dataset(const std::string &datasetdir) : mDatasetDir(datasetdir){}

Dataset::Dataset(const std::string &datasetdir, const std::string &detectionpath):
    mDatasetDir(datasetdir), mDetectionPath(detectionpath){}

bool Dataset::LoadDataset() {
    LOG(INFO) << "Load dataset from: " << mDatasetDir << std::endl;
    mImageDir = mDatasetDir + "rgb/";
}