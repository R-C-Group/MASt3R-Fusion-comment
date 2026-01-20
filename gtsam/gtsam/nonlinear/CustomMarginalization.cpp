/**
* This file is modified from DM-VIO, which is written by Vladyslav Usenko for the paper "Direct Visual-Inertial Odometry with Stereo Cameras".
* See https://github.com/lukasvst/dm-vio/blob/master/src/GTSAMIntegration/Marginalization.cpp
*/

#include <gtsam/base/SymmetricBlockMatrix.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include "CustomMarginalization.h"
#include <gtsam/inference/Symbol.h>

gtsam::LinearContainerFactor
gtsam::marginalizeOut(const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& values,
                      const gtsam::FastVector<gtsam::Key>& keysToMarginalize)
{
    if(keysToMarginalize.empty())
    {
        std::cout << "WARNING: Calling marginalizeOut with empty keysToMarginalize." << std::endl;
        return gtsam::LinearContainerFactor();
    }

    std::shared_ptr<gtsam::NonlinearFactorGraph> newGraph(new gtsam::NonlinearFactorGraph);

    gtsam::NonlinearFactorGraph marginalizedOutGraph;

    gtsam::FastSet<gtsam::Key> setOfKeysToMarginalize(keysToMarginalize);
    gtsam::FastSet<gtsam::Key> connectedKeys;

    extractKeysToMarginalize(graph, *newGraph, marginalizedOutGraph, setOfKeysToMarginalize, connectedKeys);

    gtsam::GaussianFactorGraph::shared_ptr linearizedFactorsToMarginalize = marginalizedOutGraph.linearize(values);
    std::map<gtsam::Key, size_t> keyDimMap = linearizedFactorsToMarginalize->getKeyDimMap();

    int mSize = 0;
    int aSize = 0;

    gtsam::Ordering ordering;

    gtsam::Ordering connectedOrdering;
    gtsam::FastVector<size_t> connectedDims;
    for(const gtsam::Key& k : setOfKeysToMarginalize)
    {
        ordering.push_back(k);
        mSize += keyDimMap[k];
    }
    for(const gtsam::Key& k : connectedKeys)
    {
        ordering.push_back(k);
        connectedOrdering.push_back(k);
        connectedDims.push_back(keyDimMap[k]);
        aSize += keyDimMap[k];
    }

    gtsam::Matrix hessian = linearizedFactorsToMarginalize->augmentedHessian(ordering);

    gtsam::Matrix HAfterSchurComplement = computeSchurComplement(hessian, mSize, aSize);

    gtsam::SymmetricBlockMatrix sm(connectedDims, true);
    sm.setFullMatrix(HAfterSchurComplement);

    gtsam::LinearContainerFactor lcf=gtsam::LinearContainerFactor(
            gtsam::HessianFactor(connectedOrdering, sm), values);

    return lcf;
}

void gtsam::extractKeysToMarginalize(const gtsam::NonlinearFactorGraph& graph, gtsam::NonlinearFactorGraph& newGraph,
                                     gtsam::NonlinearFactorGraph& marginalizedOutGraph,
                                     gtsam::FastSet<gtsam::Key>& setOfKeysToMarginalize,
                                     gtsam::FastSet<gtsam::Key>& connectedKeys)
{
    for(size_t i = 0; i < graph.size(); i++)
    {
        gtsam::NonlinearFactor::shared_ptr factor = graph.at(i);

        gtsam::FastSet<gtsam::Key> set_of_factor_keys(factor->keys());

        gtsam::FastSet<gtsam::Key> intersection;

        std::set_intersection(setOfKeysToMarginalize.begin(), setOfKeysToMarginalize.end(),
                              set_of_factor_keys.begin(), set_of_factor_keys.end(),
                              std::inserter(intersection, intersection.begin()));

        if(!intersection.empty())
        {
            std::set_difference(set_of_factor_keys.begin(), set_of_factor_keys.end(),
                                setOfKeysToMarginalize.begin(), setOfKeysToMarginalize.end(),
                                std::inserter(connectedKeys, connectedKeys.begin()));

            marginalizedOutGraph.add(factor);
        }else
        {
            newGraph.add(factor);
        }
    }
}

gtsam::Matrix gtsam::computeSchurComplement(const gtsam::Matrix& augmentedHessian, int mSize, int aSize)
{
    int n = augmentedHessian.rows() - 1;
    auto pair = std::pair<gtsam::Matrix, gtsam::Vector>(augmentedHessian.block(0, 0, n, n),
                                                        augmentedHessian.block(0, n, n, 1));

    // Preconditioning like in DSO code.
    gtsam::Vector SVec = (pair.first.diagonal().cwiseAbs() +
                          gtsam::Vector::Constant(pair.first.cols(), 10)).cwiseSqrt();
    gtsam::Vector SVecI = SVec.cwiseInverse();

    gtsam::Matrix hessianScaled = SVecI.asDiagonal() * pair.first * SVecI.asDiagonal();
    gtsam::Vector bScaled = SVecI.asDiagonal() * pair.second;

    gtsam::Matrix Hmm = hessianScaled.block(0, 0, mSize, mSize);
    gtsam::Matrix Hma = hessianScaled.block(0, mSize, mSize, aSize);
    gtsam::Matrix Haa = hessianScaled.block(mSize, mSize, aSize, aSize);

    gtsam::Vector bm = bScaled.segment(0, mSize);
    gtsam::Vector ba = bScaled.segment(mSize, aSize);

    // Compute inverse.
    gtsam::Matrix HmmInv = Hmm.completeOrthogonalDecomposition().pseudoInverse();

    gtsam::Matrix HaaNew = Haa - Hma.transpose() * HmmInv * Hma;
    gtsam::Vector baNew = ba - Hma.transpose() * HmmInv * bm;

    // Unscale
    gtsam::Vector SVecUpdated = SVec.segment(mSize, aSize);
    gtsam::Matrix HNewUnscaled = SVecUpdated.asDiagonal() * HaaNew * SVecUpdated.asDiagonal();
    gtsam::Matrix bNewUnscaled = SVecUpdated.asDiagonal() * baNew;

    // Make Hessian symmetric for numeric reasons.
    HNewUnscaled = 0.5 * (HNewUnscaled.transpose() + HNewUnscaled).eval();

    gtsam::Matrix augmentedHRes(aSize + 1, aSize + 1);
    augmentedHRes.setZero();
    augmentedHRes.topLeftCorner(aSize, aSize) = HNewUnscaled;
    augmentedHRes.topRightCorner(aSize, 1) = bNewUnscaled;
    augmentedHRes.bottomLeftCorner(1, aSize) = bNewUnscaled.transpose();
    augmentedHRes(aSize,aSize) = 0.0;

    return augmentedHRes;
}

gtsam::Matrix gtsam::BA2GTSAM(const gtsam::Matrix& H,
                              const gtsam::Vector& v,
                              const gtsam::Pose3& Tbc) {
    gtsam::Matrix A = -Tbc.inverse().AdjointMap();
    gtsam::Matrix Ap = A;
    Ap.block(0,0,3,6) = A.block(3,0,3,6);
    Ap.block(3,0,3,6) = A.block(0,0,3,6);
    
    int ss = H.rows()/6;
    gtsam::Matrix Hnew(ss*6,ss*6+1);
    for(int i = 0; i<ss;i++)
    {
    for(int j = 0; j<ss;j++)
        Hnew.block(i*6,j*6,6,6) = Ap.transpose() * H.block(i*6,j*6,6,6) * Ap;
    Hnew.block(i*6,ss*6,6,1) = Ap.transpose() * v.segment(i*6,6);
    }
    return Hnew;
}

gtsam::Vector gtsam::GTSAM2BA(const gtsam::Vector& x, const gtsam::Pose3& Tbc) {
    gtsam::Matrix A = -Tbc.inverse().AdjointMap();
    gtsam::Matrix Ap = A;
    Ap.block(0,0,3,6) = A.block(3,0,3,6);
    Ap.block(3,0,3,6) = A.block(0,0,3,6);

    int ss = x.rows()/6;
    gtsam::Vector xnew = x;
    for(int i=0;i<ss;i++)
    {
        xnew.segment(i*6,6) = Ap * x.segment(i*6,6);
    }
    return xnew;
}


gtsam::LinearContainerFactor gtsam::CustomHessianFactor(
    const gtsam::KeyVector& symbols_in,
    const gtsam::Values& values,
    const gtsam::Matrix& H,
    const gtsam::Vector& v) {

    gtsam::Matrix info_expand(H.rows() + 1, H.cols() + 1);
    info_expand << H, v, v.transpose(), 100.0;

    gtsam::FastVector<std::uint64_t> dims;
    for (const auto& sym : symbols_in) {
        if (sym >= gtsam::Symbol('x', 0) && sym - gtsam::Symbol('x', 0) < 100000) {
            dims.push_back(6);
        } else if (sym >=  gtsam::Symbol('s', 0) && sym -  gtsam::Symbol('s', 0) < 100000) {
            dims.push_back(1);
        }
    }
    // for(int i=0;i<dims.size();i++)
    // std::cerr<<dims[i];
    // std::cerr<<std::endl;
    // std::cerr<<info_expand.rows()<<std::endl;
    // std::cerr<<info_expand.cols()<<std::endl;
    // std::cerr<<info_expand<<std::endl;

    auto linearContainerFactor = gtsam::LinearContainerFactor(gtsam::HessianFactor(symbols_in, dims, info_expand),values);
    
    return linearContainerFactor;
}

gtsam::FastVector<gtsam::LinearContainerFactor> gtsam::Align2GTSAM_factors(
    const gtsam::JacobianVector& H11,
    const gtsam::JacobianVector& v11,
    const gtsam::JacobianVector& wTcs,
    const gtsam::FastVector<double>& ss,
    const gtsam::FastVector<int>& ii,
    const gtsam::FastVector<int>& jj,
    const int pin)
    {
        gtsam::FastVector<gtsam::LinearContainerFactor> factors;

        for (int idx = 0; idx < ii.size(); ++idx) {

                int i = ii[idx] - pin;
                int j = jj[idx] - pin;

                Eigen::Matrix4d Xi = wTcs[i];
                Xi.block<3,3>(0,0) *= ss[i];

                Eigen::Matrix4d Xj = wTcs[j];
                Xj.block<3,3>(0,0) *= ss[j];

                Eigen::Matrix4d Xij = Xi.inverse() * Xj;

                double s = std::cbrt(Xij.block<3,3>(0,0).determinant());
                Eigen::Matrix3d R = Xij.block<3,3>(0,0) / s;
                Eigen::Vector3d t = Xij.block<3,1>(0,3);

                Eigen::Matrix<double,7,7> pXij_pXj = Eigen::Matrix<double,7,7>::Zero();
                pXij_pXj.block<3,3>(0,0) = s * R;
                pXij_pXj.block<3,3>(0,3) = skewSymmetric(t) * R;
                pXij_pXj.block<3,1>(0,6) = -t;
                pXij_pXj.block<3,3>(3,3) = R;
                pXij_pXj(6,6) = 1;

                s = ss[j];
                Eigen::Matrix<double,7,7> pXj_pXj = Eigen::Matrix<double,7,7>::Zero();
                pXj_pXj(0,6) = s;
                pXj_pXj.block<3,3>(4,0) = s * Eigen::Matrix3d::Identity();
                pXj_pXj.block<3,3>(1,3) = Eigen::Matrix3d::Identity();

                Eigen::Matrix<double,7,7> pXij_pXj_ = pXij_pXj * pXj_pXj.inverse();

                Eigen::Matrix<double,7,7> pXij_pXi = -Eigen::Matrix<double,7,7>::Identity();
                s = ss[i];
                Eigen::Matrix<double,7,7> pXi_pXi = Eigen::Matrix<double,7,7>::Zero();
                pXi_pXi(0,6) = s;
                pXi_pXi.block<3,3>(4,0) = s * Eigen::Matrix3d::Identity();
                pXi_pXi.block<3,3>(1,3) = Eigen::Matrix3d::Identity();

                Eigen::Matrix<double,7,7> pXij_pXi_ = pXij_pXi * pXi_pXi.inverse();

                Eigen::Matrix<double,7,14> J;
                J.block<7,7>(0, 0) = pXij_pXi_;
                J.block<7,7>(0, 7) = pXij_pXj_;

                Eigen::Matrix<double,7,7> H = H11[idx];
                Eigen::Matrix<double,7,1> v = v11[idx];

                Eigen::Matrix<double,14,14> HHH = J.transpose() * H * J;
                Eigen::Matrix<double,14,1> vvv = J.transpose() * v;

                gtsam::FastVector<Key> symbols;
                symbols.push_back(gtsam::Symbol('s', i));
                symbols.push_back(gtsam::Symbol('x', i));
                symbols.push_back(gtsam::Symbol('s', j));
                symbols.push_back(gtsam::Symbol('x', j));
                gtsam::Values initials;

                initials.insert(gtsam::Symbol('s', i), ss[i]);
                initials.insert(gtsam::Symbol('s', j), ss[j]);
                initials.insert(gtsam::Symbol('x', i), gtsam::Pose3(wTcs[i]));
                initials.insert(gtsam::Symbol('x', j), gtsam::Pose3(wTcs[j]));
                gtsam::Matrix HHHm = HHH;
                gtsam::Vector vvvm = vvv;

                factors.push_back(CustomHessianFactor(symbols,initials,HHH/1e6,-vvv/1e6));

            // HHH 和 vvv 可用于后续线性化或求解
        }
            //     for idx in range(ii.shape[0]):
            // i = ii[idx] - pin
            // j = jj[idx] - pin

            // Xi = np.copy(wTcs[i])
            // Xi[0:3,0:3] *= ss[i]
            // Xj = np.copy(wTcs[j])
            // Xj[0:3,0:3] *= ss[j]
            // Xij = np.linalg.inv(Xi) @ Xj

            // s = np.power(np.linalg.det(Xij[0:3,0:3]),1.0/3)
            // R = Xij[0:3,0:3]/s
            // t = Xij[0:3,3]
            // pXij_pXj = np.zeros([7,7])
            // pXij_pXj[0:3,0:3] = s * R
            // pXij_pXj[0:3,3:6] = skew_sym(t) @ R
            // pXij_pXj[0:3,6] = -t
            // pXij_pXj[3:6,3:6] = R
            // pXij_pXj[6,6] = 1

            // s = ss[j]
            // pXj_pXj = np.zeros([7,7])
            // pXj_pXj[0,6] = s
            // pXj_pXj[4:7,0:3] = s*np.eye(3,3)
            // pXj_pXj[1:4,3:6] = np.eye(3,3)
            // pXij_pXj_ = pXij_pXj@np.linalg.inv(pXj_pXj)

            // pXij_pXi = -np.eye(7,7)
            // s = ss[i]
            // pXi_pXi = np.zeros([7,7])
            // pXi_pXi[0,6] = s
            // pXi_pXi[4:7,0:3] = s*np.eye(3,3)
            // pXi_pXi[1:4,3:6] = np.eye(3,3)
            // pXij_pXi_ = pXij_pXi@np.linalg.inv(pXi_pXi)

            // J = np.hstack([pXij_pXi_,pXij_pXj_])
            // H = H11[0,idx,:,:]
            // v = v11[0,idx,:]
            // HHH = J.T @ H @ J
            // vvv = J.T @ v

        //     symbols = [S(i),X(i),S(j),X(j)]
        //     initials = gtsam.Values()
        //     initials.insert(S(i),ss[i])
        //     initials.insert(S(j),ss[j])
        //     initials.insert(X(i),gtsam.Pose3(wTcs[i]))
        //     initials.insert(X(j),gtsam.Pose3(wTcs[j]))
        //     # factors.append(CustomHessianFactor(symbols,initials,HHH/1e6,-vvv/1e6))
        //     factors.append(gtsam.CustomHessianFactor(symbols,initials,HHH/1e6,-vvv/1e6))
        // return factors

        // return gtsam::FastVector<gtsam::LinearContainerFactor>();
        return factors;
    }