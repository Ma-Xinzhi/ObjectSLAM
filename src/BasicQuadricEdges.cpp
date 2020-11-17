#include "BasicQuadricEdges.h"

namespace g2o{

    void VertexQuadric::setToOriginImpl() { _estimate = Quadric(); }

    void VertexQuadric::oplusImpl(const double* update_) {
        Eigen::Map<const Vector9d> update(update_);
        setEstimate(_estimate.exp_update(update));
    }

    bool VertexQuadric::read(std::istream& is) {
        Vector9d est;
        for (int i = 0; i < 9; i++)
            is >> est[i];
        Quadric oneQuadric;
        oneQuadric.fromMinimalVector(est);
        setEstimate(oneQuadric);
        return true;
    }

    bool VertexQuadric::write(std::ostream& os) const {
        Vector10d lv = _estimate.toVector();
        for (int i = 0; i < lv.rows(); i++) {
            os << lv[i] << " ";
        }
        return os.good();
    }

    bool EdgeSE3QuadricProj::read(std::istream& is) { return true; };

    bool EdgeSE3QuadricProj::write(std::ostream& os) const { return os.good(); };

    void EdgeSE3QuadricProj::computeError() {
        //    std::cout << "EdgeSE3QuadricProj computeError" << std::endl;
        const VertexSE3Expmap* SE3Vertex = dynamic_cast<const VertexSE3Expmap*>(_vertices[0]);  //  world to camera pose
        const VertexQuadric* quadricVertex = dynamic_cast<const VertexQuadric*>(_vertices[1]);  //  object pose to world

        SE3Quat cam_pose_wc = SE3Vertex->estimate();
        Quadric global_quadric = quadricVertex->estimate();

        Vector4d rect_project = global_quadric.ProjectOntoImageRectByEllipse(cam_pose_wc, calib);  // center, width, height

        _error = rect_project - _measurement;
    }
}