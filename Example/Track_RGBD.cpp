#include "System.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <glog/logging.h>
#include <unistd.h>

void LoadImages(const std::string& strAssociationFilename, std::vector<std::string>& vstrImageFilenamesRGB,
                std::vector<std::string>& vstrImageFilenamesD, std::vector<double>& vTimestamps);

int main(int argc, char** argv){
    std::string strAssociationFilename = "/home/maxinzhi/Dataset/TUM/rgbd_dataset_freiburg2_dishes/associate.txt";
    std::string strConfigFile = "../Example/TUM2.yaml";
    std::string strDataset = "/home/maxinzhi/Dataset/TUM/rgbd_dataset_freiburg2_dishes/";

    std::vector<std::string> vstrImageFilenamesRGB, vstrImageFilenamesD;
    std::vector<double> vTimestamps;
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    if(vstrImageFilenamesRGB.empty()){
        LOG(ERROR) << "No images found in provided path..." << std::endl;
        return 1;
    }
    if(vstrImageFilenamesRGB.size() != vstrImageFilenamesD.size()){
        LOG(ERROR) << "Different number of  images for RGB and Depth..." << std::endl;
        return 1;
    }

    int nImages = vstrImageFilenamesRGB.size();

    System SLAM(strConfigFile);
    std::vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    LOG(INFO) << "-------------" << std::endl
              << "Start processing sequence ..." << std::endl
              << "Images in the sequence: " << nImages << std::endl << std::endl;

    cv::Mat imRGB, imD;
    for(int ni=0; ni<nImages; ++ni){
        imRGB = cv::imread(strDataset+vstrImageFilenamesRGB[ni], CV_LOAD_IMAGE_UNCHANGED);
        imD = cv::imread(strDataset+vstrImageFilenamesD[ni], CV_LOAD_IMAGE_UNCHANGED);
        double timestamp = vTimestamps[ni];

        if(imRGB.empty()){
            LOG(ERROR) << "Fail to load image at:" << strDataset << vstrImageFilenamesRGB[ni] << std::endl;
            return 1;
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        SLAM.TrackRGBD(imRGB, imD, timestamp);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double tTrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1).count();
        vTimesTrack[ni] = tTrack;

        double T = 0;
        if(ni < nImages-1)
            T = vTimestamps[ni+1]-timestamp;
        else
            T = timestamp - vTimestamps[ni-1];

        if(tTrack<T)
            usleep((T-tTrack)*1e6);
    }

    SLAM.ShutDown();

    std::sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for (int i = 0; i < nImages; ++i)
        totaltime += vTimesTrack[i];

    LOG(INFO) << "------------" << std::endl
              << "median tracking time: " << vTimesTrack[nImages/2] << std::endl
              << "mean tracking time: " << totaltime/nImages << std::endl;

    return 1;
}

void LoadImages(const std::string& strAssociationFilename, std::vector<std::string>& vstrImageFilenamesRGB,
                std::vector<std::string>& vstrImageFilenamesD, std::vector<double>& vTimestamps){
    std::ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof()){
        std::string s;
        std::getline(fAssociation, s);
        if(!s.empty()){
            std::stringstream ss;
            ss << s;
            double t;
            std::string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);
        }
    }
}