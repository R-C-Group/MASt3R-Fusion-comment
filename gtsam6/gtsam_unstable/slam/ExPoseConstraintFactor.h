#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace gtsam {

class ExPoseConstraintFactor : public NoiseModelFactor3<Pose3, Pose3, Pose3> {
public:

  /// shorthand for base class type
  typedef NoiseModelFactor3<Pose3, Pose3, Pose3> Base;

  // Provide access to the Matrix& version of evaluateError:
  using Base::evaluateError;
  
  typedef ExPoseConstraintFactor This;

  /// shorthand for a smart pointer to a factor
  typedef std::shared_ptr<This> shared_ptr;

 public:
  ExPoseConstraintFactor(const Key key0, const Key key1, const Key key2, const SharedNoiseModel& model)
      : Base(model, key0, key1, key2) {}

  Vector evaluateError(const Pose3& Twi,
                       const Pose3& Twc,
                       const Pose3& Tic,
                       OptionalMatrixType H0, 
                       OptionalMatrixType H1, 
                       OptionalMatrixType H2) const override {
    /**
     * Assuming self == wTa, takes a pose aTb in local coordinates
     * and transforms it to world coordinates wTb = wTa * aTb.
     * This is identical to compose.
     */
    // Pose3 transformPoseFrom(const Pose3& aTb, OptionalJacobian<6, 6> Hself = boost::none,
    //                                           OptionalJacobian<6, 6> HaTb = boost::none) const;

    Matrix66 Htcw_twc;
    Pose3 Tcw = Twc.inverse(Htcw_twc);

    Matrix66 Htcie_tcw, Htcie_twi;
    Pose3 Tci_est = Tcw.transformPoseFrom(Twi, Htcie_tcw, Htcie_twi);

    Matrix66 Herr_tic, Herr_tcie;
    Pose3 Terr = Tic.transformPoseFrom(Tci_est, Herr_tic, Herr_tcie);

    Matrix66 Hlog;
    Vector6 error = Pose3::Logmap(Terr, Hlog);

    // 组合雅可比矩阵
    if (H0) *H0 = Hlog * Herr_tcie * Htcie_twi;
    if (H1) *H1 = Hlog * Herr_tcie * Htcie_tcw * Htcw_twc;
    if (H2) *H2 = Hlog * Herr_tic;

    return error;
  }
};
}  // namespace gtsam