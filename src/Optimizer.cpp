#include "Optimizer.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"

//#include <Eigen/StdVector>

#include <mutex>

int Optimizer::PoseOptimization(std::shared_ptr<Frame> pFrame) {
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondances = 0;

    g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(toSE3Quat(pFrame->GetPose().inverse()));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    int N = pFrame->N;

    std::vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    vpEdgesStereo.reserve(N);
    std::vector<size_t> vnIndexEdgeStereo;
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);
    {
        std::unique_lock<std::mutex> lk(MapPoint::mGlobalMutex);

        for (int i = 0; i < N; ++i) {
            std::shared_ptr<MapPoint> pMP = pFrame->mvpMapPoints[i];
            if(pMP){
                /// 先只考虑RGB-D情况
                if(pFrame->mvuRight[i] > 0){
                    nInitialCorrespondances++;
                    pFrame->mvbOutlier[i] = false;

                    Eigen::Vector3d obs;
                    cv::KeyPoint kpUn = pFrame->mvKeysUn[i];
                    float kp_ur = pFrame->mvuRight[i];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                    e->setVertex(0, optimizer.vertex(0));
                    e->setMeasurement(obs);
                    float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    // 这里的原因是特征点在不同层中提取得到的
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaStereo);

                    e->fx = Frame::fx;
                    e->fy = Frame::fy;
                    e->cx = Frame::cx;
                    e->cy = Frame::cy;
                    e->bf = pFrame->mbf;

                    Eigen::Vector3d Pw = pMP->GetWorldPos();
                    e->Xw = Pw;

                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);
                    vnIndexEdgeStereo.push_back(i);
                }
            }
        }
    }

    if(nInitialCorrespondances < 3)
        return 0;

    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815,7.815};
    const int its[4]={10,10,10,10};

    int nBad = 0;
    for(size_t it=0; it<4; it++){
        vSE3->setEstimate(toSE3Quat(pFrame->GetPose().inverse()));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad = 0;
        for(size_t i=0; i<vpEdgesStereo.size(); i++){
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

            size_t idx = vnIndexEdgeStereo[i];

            // 这里这样理解，outlier点未加入到优化过程，其error参数未进行更新，还是之前的数值
            // 而inlier点的error不需要再进行计算了，之前的优化过程已经得到了最新的error值
            // 但是位姿随着迭代更新，位姿数据发生了变化，之前的outlier点用新的位姿计算error可能就不是outlier了
            // 这里做一次检查，看看是否存在这种情况
            if(pFrame->mvbOutlier[idx])
                e->computeError();

            float chi2 = e->chi2();

            if(chi2 > chi2Stereo[it]){
                pFrame->mvbOutlier[idx] = true;
                e->setLevel(1);
                nBad++;
            }
            else{
                e->setLevel(0);
                pFrame->mvbOutlier[idx] = false;
            }

            if(it == 2)
                e->setRobustKernel(nullptr);
        }
        if(optimizer.edges().size()<10)
            break;
    }
    g2o::VertexSE3Expmap* vSE3_opt = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3Quat_opt = vSE3_opt->estimate();
    Eigen::Matrix4d Twc = SE3Quat_opt.to_homogeneous_matrix().inverse();
    pFrame->SetPose(Twc);

    return nInitialCorrespondances-nBad;
}

g2o::SE3Quat Optimizer::toSE3Quat(const Eigen::Matrix4d &pose) {
    Eigen::Matrix3d R = pose.block(0,0,3,3);
    Eigen::Vector3d t = pose.col(3).head(3);
    return g2o::SE3Quat(R, t);
}

void Optimizer::LocalBundleAdjustment(std::shared_ptr<KeyFrame> pKF, bool *pbStopFlag, std::shared_ptr<Map> pMap) {
    std::list<std::shared_ptr<KeyFrame>> lLocalKeyFrames;

    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    std::vector<std::shared_ptr<KeyFrame>> vpNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for(auto& pNeighKF : vpNeighKFs){
        if(!pKF->isBad()){
            pNeighKF->mnBALocalForKF = pKF->mnId;
            lLocalKeyFrames.push_back(pNeighKF);
        }
    }

    std::list<std::shared_ptr<MapPoint>> lLocalMapPoints;
    for(auto& pLocalKF : lLocalKeyFrames){
        std::vector<std::shared_ptr<MapPoint>> vpMPs = pLocalKF->GetMapPointMatches();
        for(auto& pMP : vpMPs){
            if(pMP){
                if(!pMP->isBad()){
                    if(pMP->mnBALocalForKF != pKF->mnId){
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF = pKF->mnId;
                    }
                }
            }
        }
    }

    std::list<std::shared_ptr<KeyFrame>> lFixedKeyFrames;
    for(auto& pMP : lLocalMapPoints){
        std::map<std::shared_ptr<KeyFrame>, size_t> obs = pMP->GetObservations();
        for(auto& ob : obs){
            std::shared_ptr<KeyFrame> pKFi = ob.first;
            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId){
                if(!pKFi->isBad()){
                    pKFi->mnBAFixedForKF = pKF->mnId;
                    if(!pKFi->isBad())
                        lFixedKeyFrames.push_back(pKFi);
                }
            }
        }
    }

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    for(auto& pLocalKF : lLocalKeyFrames){
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap;
        vSE3->setEstimate(toSE3Quat(pLocalKF->GetPose().inverse()));
        vSE3->setId(pLocalKF->mnId);
        vSE3->setFixed(pLocalKF->mnId == 0);
        optimizer.addVertex(vSE3);
        if(pLocalKF->mnId > maxKFid)
            maxKFid = pLocalKF->mnId;
    }

    for(auto& pFixedKF : lFixedKeyFrames){
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap;
        vSE3->setEstimate(toSE3Quat(pFixedKF->GetPose().inverse()));
        vSE3->setId(pFixedKF->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pFixedKF->mnId > maxKFid)
            maxKFid = pFixedKF->mnId;
    }

    int nExpectedSize = (lLocalKeyFrames.size()+lFixedKeyFrames.size())*lLocalMapPoints.size();

    std::vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    std::vector<std::shared_ptr<KeyFrame>> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    std::vector<std::shared_ptr<MapPoint>> vpEdgeMapPointStereo;
    vpEdgeMapPointStereo.reserve(nExpectedSize);

    float thHuberStereo = sqrt(7.815);

    for(auto& pMP : lLocalMapPoints){
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ;
        vPoint->setEstimate(pMP->GetWorldPos());
        int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        std::map<std::shared_ptr<KeyFrame>, size_t> obs = pMP->GetObservations();
        for(auto& ob : obs){
            std::shared_ptr<KeyFrame> pKFi = ob.first;
            if(!pKFi->isBad()){
                cv::KeyPoint kpUn = pKFi->mvKeysUn[ob.second];
                if(pKFi->mvuRight[ob.second]>0){
                    Eigen::Vector3d kp_uvr;
                    float ur = pKFi->mvuRight[ob.second];
                    kp_uvr << kpUn.pt.x, kpUn.pt.y, ur;

                    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ;

                    e->setVertex(0, optimizer.vertex(id));
                    e->setVertex(1, optimizer.vertex(pKFi->mnId));
                    e->setMeasurement(kp_uvr);

                    float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpEdgeMapPointStereo.push_back(pMP);
                }
            }
        }
    }

    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    bool bDoMore = true;

    if(pbStopFlag)
        if(*pbStopFlag)
            bDoMore = false;

    if(bDoMore){
        for (int i = 0; i < vpEdgesStereo.size(); ++i) {
            g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
            std::shared_ptr<MapPoint> pMP = vpEdgeMapPointStereo[i];

            if(pMP->isBad())
                continue;

            if(e->chi2() > 7.815 || !e->isDepthPositive())
                e->setLevel(1);

            e->setRobustKernel(nullptr);
        }

        optimizer.initializeOptimization(0);
        optimizer.optimize(10);
    }

    std::vector<std::pair<std::shared_ptr<KeyFrame>, std::shared_ptr<MapPoint>>> vToErase;
    vToErase.resize(vpEdgesStereo.size());

    for (int i = 0; i < vpEdgesStereo.size(); ++i) {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        std::shared_ptr<MapPoint> pMP = vpEdgeMapPointStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive()){
            std::shared_ptr<KeyFrame> pKFi = vpEdgeKFStereo[i];
            vToErase.emplace_back(std::make_pair(pKFi, pMP));
        }
    }

    std::unique_lock<std::mutex> lk(pMap->mMutexMapUpdate);

    if(!vToErase.empty()){
        for(auto& pair : vToErase){
            std::shared_ptr<KeyFrame> pKFi = pair.first;
            std::shared_ptr<MapPoint> pMPi = pair.second;

            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    for(auto& pKFi : lLocalKeyFrames){
        g2o::VertexSE3Expmap* vSE3 = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
        Eigen::Matrix4d Twc = vSE3->estimate().to_homogeneous_matrix().inverse();
        pKFi->SetPose(Twc);
    }

    for(auto& pMP : lLocalMapPoints){
        g2o::VertexSBAPointXYZ* vPoint = dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(maxKFid+pMP->mnId+1));
        pMP->SetWorldPos(vPoint->estimate());
        pMP->UpdateNormalAndDepth();
    }
}