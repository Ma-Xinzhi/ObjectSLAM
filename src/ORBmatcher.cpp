#include "ORBmatcher.h"

// ORB-SLAM中采用了两种保证匹配的鲁棒性的方法
// 第一种是描述子之间的距离最小的与第二小的距离之间满足一定阈值条件
// 第二种是选取满足相似旋转角度的特征点数量前三的特征点作为匹配

const int ORBmatcher::TH_HIGH = 100; // 描述子距离大阈值
const int ORBmatcher::TH_LOW = 50; // 描述子距离小阈值
const int ORBmatcher::HISTO_LENGTH = 30; // 旋转角度的分隔

ORBmatcher::ORBmatcher(float nratio, bool CheckOri): mfNNratio(nratio), mbCheckOrientation(CheckOri) {}

int ORBmatcher::SearchByProjection(const std::shared_ptr<Frame>& pF, const std::vector<std::shared_ptr<MapPoint>> &vpMapPoints,
                                   float th) {
    int nmatches = 0;

    bool bFactor = th!=1.0;

    for(auto& pMP : vpMapPoints){
        if(!pMP->mbTrackInView)
            continue;

        if(pMP->isBad())
            continue;

        int nPredictedLevel = pMP->mnTrackScaleLevel;

        // The size of the window will depend on the viewing direction
        float r = RadiusByViewingCos(pMP->mTrackViewCos);

        if(bFactor)
            r *= th;

        std::vector<size_t> vIndices =
                pF->GetFeaturesInArea(pMP->mTrackProjX, pMP->mTrackProjY, r*pF->mvScaleFactors[nPredictedLevel], nPredictedLevel-1, nPredictedLevel);

        if(vIndices.empty())
            continue;

        cv::Mat MPDescriptor = pMP->GetDescriptor();

        int bestDist = 256;
        int bestLevel = -1;
        int bestDist2 = 256;
        int bestLevel2 = -1;
        int bestIdx = -1;

        for(auto idx : vIndices){
            //如果有对应的观察点，则不进行匹配
            if(pF->mvpMapPoints[idx])
                if(pF->mvpMapPoints[idx]->Observations() > 0)
                    continue;

            // 如果双目情况下，视差大于阈值，说明不匹配
            if(pF->mvuRight[idx] > 0){
                float er = fabs(pMP->mTrackProjX - pF->mvuRight[idx]);
                if(er > r*pF->mvScaleFactors[nPredictedLevel])
                    continue;
            }

            cv::Mat d = pF->mDescriptors.row(idx);

            int dist = ORBmatcher::DescriptorDistance(MPDescriptor, d);

            if(dist < bestDist){
                bestDist2 = bestDist;
                bestDist = dist;
                bestLevel2 = bestLevel;
                bestLevel = pF->mvKeysUn[idx].octave;
                bestIdx = idx;
            }
            else if(dist < bestDist2){
                bestDist2 = dist;
                bestLevel2 = pF->mvKeysUn[idx].octave;
            }
        }

        if(bestDist <= TH_HIGH){
            // 如果匹配的两个最佳匹配点在同一层，需要比较其大小是否满足一定阈值
            if(bestLevel == bestLevel2 && bestDist > mfNNratio*bestDist2)
                continue;
            pF->mvpMapPoints[bestIdx] = pMP;
            nmatches++;
        }
    }

    return nmatches;
}

int ORBmatcher::SearchByProjection(const std::shared_ptr<Frame>& pCurrentFrame, const std::shared_ptr<Frame>& pLastFrame, float th, bool bMono) {
    int nmatches = 0;

    // Rotation Histogram (to check rotation consistency)
    std::vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    Eigen::Matrix3d Rwc = pCurrentFrame->GetRotation();
    Eigen::Matrix3d Rcw = Rwc.transpose();
    Eigen::Vector3d twc = pCurrentFrame->GetTranslation();
    Eigen::Vector3d tcw = -Rcw*twc;

    Eigen::Matrix3d Rwl = pLastFrame->GetRotation();
    Eigen::Vector3d twl = pLastFrame->GetTranslation();

    Eigen::Vector3d tlc = twc - twl;

    // 观察当前帧相对于上一帧是前进还是后退
    bool bForward = tlc[2]>pCurrentFrame->mb && !bMono;
    bool bBackward = -tlc[2]>pCurrentFrame->mb && !bMono;

    for(int i=0; i<pLastFrame->N; i++){
        std::shared_ptr<MapPoint> pMP = pLastFrame->mvpMapPoints[i];
        if(pMP){
            if(!pLastFrame->mvbOutlier[i]){
                Eigen::Vector3d x3Dw = pMP->GetWorldPos();
                Eigen::Vector3d x3Dc = Rcw*x3Dw + tcw;

                float xc = x3Dc[0];
                float yc = x3Dc[1];
                float invzc = 1/x3Dc[2];

                if(invzc<0)
                    continue;

                float u = xc*Frame::fx*invzc + Frame::cx;
                float v = yc*Frame::fy*invzc + Frame::cy;

                if(u<Frame::mnMinX || u>Frame::mnMaxX)
                    continue;
                if(v<Frame::mnMinY || v>Frame::mnMaxY)
                    continue;

                int nLastOctave = pLastFrame->mvKeysUn[i].octave;

                float radius = th*pCurrentFrame->mvScaleFactors[nLastOctave];

                std::vector<size_t> vIndices;

                // 离得近的点可能在金字塔中更高的层中提取到，如果相比上一帧前进的话，可能在更高的层中提取到相应特征点
                // 如果相比上一帧后退的话，只能在该层或者更低的层中提取到相应特征点
                if(bForward)
                    vIndices = pCurrentFrame->GetFeaturesInArea(u, v, radius, nLastOctave);
                else if(bBackward)
                    vIndices = pCurrentFrame->GetFeaturesInArea(u, v, radius, 0, nLastOctave);
                else
                    vIndices = pCurrentFrame->GetFeaturesInArea(u, v, radius, nLastOctave-1, nLastOctave+1);

                if(vIndices.empty())
                    continue;

                cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx = -1;

                for(const auto& index : vIndices){
                    if(pCurrentFrame->mvpMapPoints[index])
                        if(pCurrentFrame->mvpMapPoints[index]->Observations() > 0)
                            continue;

                    if(pCurrentFrame->mvuRight[index] > 0){
                        float ur = u-pCurrentFrame->mbf*invzc;
                        float er = fabs(ur - pCurrentFrame->mvuRight[index]);
                        if(er > radius)
                            continue;
                    }

                    cv::Mat d = pCurrentFrame->mDescriptors.row(index);

                    int dist = DescriptorDistance(dMP, d);

                    if(dist < bestDist){
                        bestDist = dist;
                        bestIdx = index;
                    }
                }

                if(bestDist<=TH_HIGH){
                    pCurrentFrame->mvpMapPoints[bestIdx] = pMP;
                    nmatches++;

                    if(mbCheckOrientation){
                        float rot = pLastFrame->mvKeysUn[i].angle - pCurrentFrame->mvKeysUn[bestIdx].angle;
                        if(rot < 0)
                            rot += 360;
                        int bin = round(rot*factor);
                        // 这里的条件我觉得不可能成立
                        if(bin == HISTO_LENGTH)
                            bin = 0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx);
                    }
                }
            }
        }
    }

    if(mbCheckOrientation){
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; ++i) {
            if(i!=ind1 && i!=ind2 && i!=ind3) {
                for (int j = 0; j < rotHist[i].size(); ++j) {
                    pCurrentFrame->mvpMapPoints[rotHist[i][j]] = nullptr;
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

int ORBmatcher::SearchByProjection(const std::shared_ptr<Frame>& pCurrentFrame, const std::shared_ptr<KeyFrame>& pKF,
                                   const std::set<std::shared_ptr<MapPoint>> &sAlreadyFound, float th, int ORBdist) {
    int nmatches = 0;

    Eigen::Matrix3d Rwc = pCurrentFrame->GetRotation();
    Eigen::Vector3d twc = pCurrentFrame->GetTranslation();
    Eigen::Vector3d tcw = -Rwc.transpose()*twc;

    std::vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    std::vector<std::shared_ptr<MapPoint>> vpMPs = pKF->GetMapPointMatches();

    // vpMPs.size()应该等于N，其中有一些是空指针
    for (int i = 0; i < vpMPs.size(); ++i) {
        std::shared_ptr<MapPoint> pMP = vpMPs[i];
        if(pMP){
            if(!pMP->isBad() && !sAlreadyFound.count(pMP)) {
                Eigen::Vector3d x3Dw = pMP->GetWorldPos();
                Eigen::Vector3d x3Dc = Rwc.transpose() * x3Dw + tcw;

                float xc = x3Dc[0];
                float yc = x3Dc[1];
                float invzc = 1 / x3Dc[2];

                float u = xc * Frame::fx * invzc + Frame::cx;
                float v = yc * Frame::fy * invzc + Frame::cy;

                if (u < Frame::mnMinX || u > Frame::mnMaxX)
                    continue;
                if (v < Frame::mnMinY || v > Frame::mnMaxY)
                    continue;

                Eigen::Vector3d PC = x3Dw - twc;
                float dist3D = PC.norm();

                float maxDistance = pMP->GetMaxDistanceInvariance();
                float minDistance = pMP->GetMinDistanceInvariance();

                if (dist3D < minDistance || dist3D > maxDistance)
                    continue;

                int nPredictedLevel = pMP->PredictScale(dist3D, pCurrentFrame.get());

                float radius = th * pCurrentFrame->mvScaleFactors[nPredictedLevel];

                std::vector<size_t> vIndices = pCurrentFrame->GetFeaturesInArea(u, v, radius, nPredictedLevel - 1,
                                                                              nPredictedLevel + 1);

                if (vIndices.empty())
                    continue;

                cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx = -1;

                for (auto &index : vIndices) {
                    if (pCurrentFrame->mvpMapPoints[index])
                        continue;
                    cv::Mat d = pCurrentFrame->mDescriptors.row(index);

                    int dist = DescriptorDistance(dMP, d);
                    if (dist < bestDist) {
                        bestDist = dist;
                        bestIdx = index;
                    }
                }

                if(bestDist <= ORBdist){
                    pCurrentFrame->mvpMapPoints[bestIdx] = pMP;
                    nmatches++;

                    if(mbCheckOrientation){
                        float rot = pKF->mvKeysUn[i].angle - pCurrentFrame->mvKeysUn[i].angle;
                        if(rot < 0)
                            rot += 360;
                        int bin = round(rot*factor);
                        if(bin == HISTO_LENGTH)
                            bin = 0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx);
                    }
                }
            }
        }
    }

    if(mbCheckOrientation){
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; ++i) {
            if(i!=ind1 && i!=ind2 && i!=ind3){
                for (int j = 0; j < rotHist[i].size(); ++j) {
                    pCurrentFrame->mvpMapPoints[rotHist[i][j]] = nullptr;
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

int ORBmatcher::SearchByProjection(const std::shared_ptr<Frame>& pCurrentFrame, const std::shared_ptr<KeyFrame>& pKF, float th)
{
    int nmatches = 0;

    Eigen::Matrix3d Rwc = pCurrentFrame->GetRotation();
    Eigen::Vector3d twc = pCurrentFrame->GetTranslation();
    Eigen::Vector3d tcw = -Rwc.transpose()*twc;

    std::vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    std::vector<std::shared_ptr<MapPoint>> vpMPs = pKF->GetMapPointMatches();

    // vpMPs.size()应该等于N，其中有一些是空指针
    for (int i = 0; i < vpMPs.size(); ++i) {
        std::shared_ptr<MapPoint> pMP = vpMPs[i];
        if(pMP){
            if(!pMP->isBad()) {
                Eigen::Vector3d x3Dw = pMP->GetWorldPos();
                Eigen::Vector3d x3Dc = Rwc.transpose() * x3Dw + tcw;

                float xc = x3Dc[0];
                float yc = x3Dc[1];
                float invzc = 1 / x3Dc[2];

                float u = xc * Frame::fx * invzc + Frame::cx;
                float v = yc * Frame::fy * invzc + Frame::cy;

                if (u < Frame::mnMinX || u > Frame::mnMaxX)
                    continue;
                if (v < Frame::mnMinY || v > Frame::mnMaxY)
                    continue;

                Eigen::Vector3d PC = x3Dw - twc;
                float dist3D = PC.norm();

                float maxDistance = pMP->GetMaxDistanceInvariance();
                float minDistance = pMP->GetMinDistanceInvariance();

                if (dist3D < minDistance || dist3D > maxDistance)
                    continue;

                int nPredictedLevel = pMP->PredictScale(dist3D, pCurrentFrame.get());

                float radius = th * pCurrentFrame->mvScaleFactors[nPredictedLevel];

                std::vector<size_t> vIndices = pCurrentFrame->GetFeaturesInArea(u, v, radius, nPredictedLevel - 1,
                                                                                nPredictedLevel + 1);

                if (vIndices.empty())
                    continue;

                cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx = -1;

                for (auto &index : vIndices) {
                    if (pCurrentFrame->mvpMapPoints[index])
                        continue;
                    cv::Mat d = pCurrentFrame->mDescriptors.row(index);

                    int dist = DescriptorDistance(dMP, d);
                    if (dist < bestDist) {
                        bestDist = dist;
                        bestIdx = index;
                    }
                }

                if(bestDist <= TH_HIGH){
                    pCurrentFrame->mvpMapPoints[bestIdx] = pMP;
                    nmatches++;

                    if(mbCheckOrientation){
                        float rot = pKF->mvKeysUn[i].angle - pCurrentFrame->mvKeysUn[i].angle;
                        if(rot < 0)
                            rot += 360;
                        int bin = round(rot*factor);
                        if(bin == HISTO_LENGTH)
                            bin = 0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx);
                    }
                }
            }
        }
    }

    if(mbCheckOrientation){
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; ++i) {
            if(i!=ind1 && i!=ind2 && i!=ind3){
                for (int j = 0; j < rotHist[i].size(); ++j) {
                    pCurrentFrame->mvpMapPoints[rotHist[i][j]] = nullptr;
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

// TODO ORB中利用DBOW做的关键帧间的匹配，为保证匹配的准确性，利用了极线约束
//  这里不使用DBOW，我觉得可以利用语义信息做关联，或许比DBOW会好点

/// 先用特征点匹配进行三角化的关联
int ORBmatcher::SearchForTriangulation(const std::shared_ptr<KeyFrame>& pKF1, const std::shared_ptr<KeyFrame>& pKF2, const Eigen::Matrix3d& F12,
                                       std::vector<std::pair<size_t, size_t>> &vMatchedPairs) {

}

int ORBmatcher::SearchBySim3(const std::shared_ptr<KeyFrame>& pKF1, const std::shared_ptr<KeyFrame>& pKF2,
                             std::vector<std::shared_ptr<MapPoint>> &vpMatches12, const float &s12, const Eigen::Matrix3d &R12,
                             const Eigen::Vector3d &t12, const float th) {
    float fx = pKF1->fx;
    float fy = pKF1->fy;
    float cx = pKF1->cx;
    float cy = pKF1->cy;

    Eigen::Matrix3d Rwc1 = pKF1->GetRotation();
    Eigen::Vector3d twc1 = pKF1->GetTranslation();
    Eigen::Vector3d tc1w = -Rwc1.transpose()*twc1;

    Eigen::Matrix3d Rwc2 = pKF2->GetRotation();
    Eigen::Vector3d twc2 = pKF2->GetTranslation();
    Eigen::Vector3d tc2w = -Rwc2.transpose()*twc2;

    Eigen::Matrix3d sR12 = s12 * R12;
    Eigen::Matrix3d sR21 = (1.0/s12) * R12.transpose();
    Eigen::Vector3d t21 = -sR21 * t12;

    std::vector<std::shared_ptr<MapPoint>> vpMapPoints1 = pKF1->GetMapPointMatches();
    std::vector<std::shared_ptr<MapPoint>> vpMapPoints2 = pKF2->GetMapPointMatches();

    int N1 = vpMapPoints1.size();
    int N2 = vpMapPoints2.size();

    std::vector<bool> vbAlreadyMatched1(N1, false);
    std::vector<bool> vbAlreadyMatched2(N2, false);

    for (int i = 0; i < N1; ++i) {
        std::shared_ptr<MapPoint> pMP = vpMatches12[i];
        if(pMP){
            vbAlreadyMatched1[i] = true;
            int idx2 = pMP->GetIndexInKeyFrame(pKF2);
            if(idx2>=0 && idx2<N2)
                vbAlreadyMatched2[idx2] = true;
        }
    }

    std::vector<int> vnMatch1(N1, -1);
    std::vector<int> vnMatch2(N2, -1);

    for (int i1 = 0; i1 < N1; ++i1) {
        std::shared_ptr<MapPoint> pMP = vpMapPoints1[i1];

        if(!pMP || pMP->isBad() || vbAlreadyMatched1[i1])
            continue;

        Eigen::Vector3d p3Dw = pMP->GetWorldPos();
        Eigen::Vector3d p3Dc1 = Rwc1.transpose()*p3Dw+tc1w;
        Eigen::Vector3d p3Dc2 = sR21*p3Dc1+t21;

        if(p3Dc2[2] < 0)
            continue;

        float x = p3Dc2[0];
        float y = p3Dc2[1];
        float invz = 1/p3Dc2[2];

        float u = x*fx*invz + cx;
        float v = y*fy*invz + cy;

        if(!pKF2->IsInImage(u, v))
            continue;

        float maxDistance = pMP->GetMaxDistanceInvariance();
        float minDistance = pMP->GetMinDistanceInvariance();

        float dist3D = p3Dc2.norm();
        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist3D, pKF2.get());

        float radius = th*pKF2->mvScaleFactors[nPredictedLevel];

        std::vector<size_t> vIndices = pKF2->GetFeaturesInArea(u, v, radius, nPredictedLevel-1, nPredictedLevel+1);

        if(vIndices.empty())
            continue;

        cv::Mat dMP = pKF1->mDescriptors.row(i1);

        int bestDist = 256;
        int bestIdx = -1;

        for(auto& idx : vIndices){
            cv::Mat dKF2 = pKF2->mDescriptors.row(idx);

            int dist = DescriptorDistance(dMP, dKF2);

            if(dist<bestDist){
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist <= TH_HIGH)
            vnMatch1[i1] = bestIdx;
    }

    for (int i2 = 0; i2 < N2; ++i2) {
        std::shared_ptr<MapPoint> pMP = vpMapPoints2[i2];

        if(!pMP || pMP->isBad() || vbAlreadyMatched2[i2])
            continue;

        Eigen::Vector3d p3Dw = pMP->GetWorldPos();
        Eigen::Vector3d p3Dc2 = Rwc2.transpose()*p3Dw + tc2w;
        Eigen::Vector3d p3Dc1 = sR12*p3Dc2+t12;

        if(p3Dc1[2]<0)
            continue;

        float x = p3Dc1[0];
        float y = p3Dc1[1];
        float invz = 1/p3Dc1[2];

        float u = x*invz*fx + cx;
        float v = y*invz*fy + cy;

        if(!pKF1->IsInImage(u,v))
            continue;

        float maxDistance = pMP->GetMaxDistanceInvariance();
        float minDistance = pMP->GetMinDistanceInvariance();

        float dist3D = p3Dw.norm();

        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist3D, pKF1.get());

        float radius = th*pKF1->mvScaleFactors[nPredictedLevel];

        std::vector<size_t> vIndices = pKF1->GetFeaturesInArea(u, v, radius, nPredictedLevel-1, nPredictedLevel+1);

        cv::Mat dMP = pKF2->mDescriptors.row(i2);

        int bestDist = 256;
        int bestIdx = -1;

        for(auto& idx : vIndices){
            cv::Mat dKF1 = pKF1->mDescriptors.row(idx);

            int dist = DescriptorDistance(dMP, dKF1);

            if(dist<bestDist){
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<TH_HIGH)
            vnMatch2[i2] = bestIdx;
    }

    int nFound = 0;

    for (int i = 0; i < N1; ++i) {
        int idx2 = vnMatch1[i];
        if(idx2>=0){
            int idx1 = vnMatch2[idx2];
            if(i == idx1){
                vpMatches12[i] = vpMapPoints2[idx2];
                nFound++;
            }
        }
    }

    return nFound;
}

int ORBmatcher::Fuse(const std::shared_ptr<KeyFrame>& pKF, const std::vector<std::shared_ptr<MapPoint>> &vpMapPoints, float th)
{
    Eigen::Matrix3d Rwc = pKF->GetRotation();
    Eigen::Vector3d twc = pKF->GetTranslation();
    Eigen::Vector3d tcw = -Rwc.transpose()*twc;

    float fx = pKF->fx;
    float fy = pKF->fy;
    float cx = pKF->cx;
    float cy = pKF->cy;
    float bf = pKF->mbf;

    Eigen::Vector3d Ow = pKF->GetCameraCenter();

    int nFused = 0;

    int nMPs = vpMapPoints.size();

    for (int i = 0; i < nMPs; ++i) {
        std::shared_ptr<MapPoint> pMP = vpMapPoints[i];

        if(!pMP || pMP->isBad() || pMP->IsInKeyFrame(pKF))
            continue;

        Eigen::Vector3d p3Dw = pMP->GetWorldPos();
        Eigen::Vector3d p3Dc = Rwc.transpose()*p3Dw + tcw;

        if(p3Dc[2]<0)
            continue;

        float x = p3Dc[0];
        float y = p3Dc[1];
        float invz = p3Dc[2];

        float u = x*fx*invz+cx;
        float v = y*fy*invz+cy;

        if(!pKF->IsInImage(u,v))
            continue;

        float ur = u-bf*invz;

        float maxDistance = pMP->GetMaxDistanceInvariance();
        float minDistance = pMP->GetMinDistanceInvariance();
        Eigen::Vector3d PO = p3Dw - Ow;
        float dist3D = PO.norm();

        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // 该点的观察和平均观察之间的角度应该小于60度
        Eigen::Vector3d Pn = pMP->GetNormal();
        if(PO.dot(Pn) < 0.5*dist3D)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist3D, pKF.get());

        float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        std::vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius, nPredictedLevel-1, nPredictedLevel+1);

        if(vIndices.empty())
            continue;

        cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;

        for(auto& index : vIndices){
            cv::KeyPoint kp = pKF->mvKeysUn[index];
            int kpLevel = kp.octave;
            if(pKF->mvuRight[index]>0){
                float kpx = kp.pt.x;
                float kpy = kp.pt.y;
                float kpr = pKF->mvuRight[index];
                float ex = u - kpx;
                float ey = v - kpy;
                float er = ur - kpr;
                float e2 = ex*ex+ey*ey+er*er;

                if(e2*pKF->mvInvLevelSigma2[kpLevel] > 7.8)
                    continue;
            }
            else{
                float kpx = kp.pt.x;
                float kpy = kp.pt.y;
                float ex = u - kpx;
                float ey = v - kpy;
                float e2 = ex*ex+ey*ey;

                if(e2*pKF->mvInvLevelSigma2[kpLevel] > 5.99)
                    continue;
            }

            cv::Mat dKF = pKF->mDescriptors.row(index);

            int dist = DescriptorDistance(dMP, dKF);

            if(dist < bestDist){
                bestDist = dist;
                bestIdx = index;
            }
        }
        if(bestDist <= TH_LOW){
            std::shared_ptr<MapPoint> pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF){
                if(!pMPinKF->isBad()){
                    if(pMPinKF->Observations() > pMP->Observations())
                        pMP->Replace(pMPinKF);
                    else
                        pMPinKF->Replace(pMP);
                }
            }
            else{
                pMP->AddObservation(pKF, bestIdx);
                pKF->AddMapPoint(pMP, bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}

int ORBmatcher::Fuse(const std::shared_ptr<KeyFrame>& pKF, Eigen::Matrix4d Scw,
                     const std::vector<std::shared_ptr<MapPoint>> &vpPoints, float th,
                     std::vector<std::shared_ptr<MapPoint>> &vpReplacePoint) {
    float fx = pKF->fx;
    float fy = pKF->fy;
    float cx = pKF->cx;
    float cy = pKF->cy;

    Eigen::Matrix3d sRcw = Scw.block(0,0,3,3);
    float scw = sRcw.row(0).norm();
    Eigen::Matrix3d Rcw = sRcw/scw;
    Eigen::Vector3d tcw = Scw.col(3).head(3)/scw;
    Eigen::Vector3d Ow = -Rcw.transpose()*tcw;

    std::set<std::shared_ptr<MapPoint>> spAlreadyFound = pKF->GetMapPoints();

    int nFused = 0;

    int nPoints = vpPoints.size();

    for (int iMP = 0; iMP < nPoints; ++iMP) {
        std::shared_ptr<MapPoint> pMP = vpPoints[iMP];

        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        Eigen::Vector3d p3Dw = pMP->GetWorldPos();
        Eigen::Vector3d p3Dc = Rcw*p3Dw+tcw;

        if(p3Dc[2]<0)
            continue;

        float x = p3Dc[0];
        float y = p3Dc[1];
        float invz = 1/p3Dc[2];

        float u = x*invz*fx+cx;
        float v = y*invz*fy+cy;

        if(!pKF->IsInImage(u,v))
            continue;

        float maxDistance = pMP->GetMaxDistanceInvariance();
        float minDistance = pMP->GetMinDistanceInvariance();
        Eigen::Vector3d PO = p3Dw - Ow;
        float dist3D = PO.norm();

        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        Eigen::Vector3d Pn = pMP->GetNormal();
        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist3D, pKF.get());

        float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        std::vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius, nPredictedLevel-1, nPredictedLevel+1);
        if(vIndices.empty())
            continue;

        cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;

        for(auto& index : vIndices){
            cv::Mat dKF = pKF->mDescriptors.row(index);

            int dist = DescriptorDistance(dMP, dKF);

            if(dist < bestDist){
                bestDist = dist;
                bestIdx = index;
            }
        }

        if(bestDist <= TH_LOW){
            std::shared_ptr<MapPoint> pMPinKF = pKF->GetMapPoint(bestIdx);

            if(pMPinKF){
                if(!pMPinKF->isBad())
                    vpReplacePoint[iMP] = pMPinKF;
            }
            else{
                pKF->AddMapPoint(pMP, bestIdx);
                pMP->AddObservation(pKF, bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}

float ORBmatcher::RadiusByViewingCos(float viewCos) {
    if(viewCos > 0.998)
        return 2.5;
    else
        return 4.0;
}

bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const Eigen::Matrix3d &F12,
                                       const std::shared_ptr<KeyFrame> pKF) {
    float a = kp1.pt.x*F12(0,0)+kp1.pt.y*F12(1,0)+F12(2,0);
    float b = kp1.pt.x*F12(0,1)+kp1.pt.y*F12(1,1)+F12(2,1);
    float c = kp1.pt.x*F12(0,2)+kp1.pt.y*F12(1,2)+F12(2,2);

    float num = a*kp2.pt.x+b*kp2.pt.y+c;

    float d = a*a+b*b;

    if(d == 0)
        return false;

    float dsqr = num*num/d;

    return dsqr < 3.84*pKF->mvLevelSigma2[kp2.octave];
}

void ORBmatcher::ComputeThreeMaxima(std::vector<int>* hist, int L, int &ind1, int &ind2, int &ind3) {
    int max1 = 0;
    int max2 = 0;
    int max3 = 0;

    for(int i=0; i<L; i++){
        int s = hist[i].size();
        if(s>max1){
            max3 = max2;
            max2 = max1;
            max1 = s;
            ind3 = ind2;
            ind2 = ind1;
            ind1 = i;
        }
        else if(s>max2){
            max3 = max2;
            max2 = s;
            ind3 = ind2;
            ind2 = i;
        }
        else if(s>max3){
            max3 = s;
            ind3 = i;
        }
    }

    if(max2 < 0.1*(float)max1)
        ind2 = ind3 = -1;
    else if(max3 < 0.1*(float)max2)
        ind3 = -1;
}

// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}