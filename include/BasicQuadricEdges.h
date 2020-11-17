#ifndef OBJECTSLAM_BASICQUADRICEDGES_H
#define OBJECTSLAM_BASICQUADRICEDGES_H

#include "Quadric.h"
#include "Thirdparty/g2o/g2o/core/base_multi_edge.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"


namespace g2o{
    // NOTE  this vertex stores object pose to world
    class VertexQuadric : public BaseVertex<9, Quadric>{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexQuadric(){};

        virtual void setToOriginImpl();

        virtual void oplusImpl(const double* update_);

        virtual bool read(std::istream& is);

        virtual bool write(std::ostream& os) const;
    };

// camera -object 2D projection error, rectangle difference, could also change
// to iou

    class EdgeSE3QuadricProj : public BaseBinaryEdge<4, Vector4d, VertexSE3Expmap, VertexQuadric> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeSE3QuadricProj(){};

        virtual bool read(std::istream& is);

        virtual bool write(std::ostream& os) const;

        void computeError();

        Eigen::Matrix3d calib;
    };
}

#endif //OBJECTSLAM_BASICQUADRICEDGES_H
