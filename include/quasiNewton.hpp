#ifndef QUASI_NEWTON_HPP
#define QUASI_NEWTON_HPP
#include <Eigen/Dense>

/*
 * Implementations of a selection of quasi-Newton update formulas. 
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     11/14/2019
 */
using namespace Eigen;

template <typename T>
Matrix<T, Dynamic, Dynamic> updateBFGSInv(const Ref<const Matrix<T, Dynamic, Dynamic> >& H,
                                          const Ref<const Matrix<T, Dynamic, 1> >& s,
                                          const Ref<const Matrix<T, Dynamic, 1> >& y)
{
    /*
     * Compute the BFGS update formula (Nocedal and Wright, Eqn. 6.17).
     */
    Matrix<T, Dynamic, Dynamic> I = Matrix<T, Dynamic, Dynamic>::Identity(H.rows(), H.cols());
    Matrix<T, Dynamic, 1> rho = 1.0 / y.dot(s);
    return (I - rho * s * y.transpose()) * H * (I - rho * y * s.transpose()) + rho * s * s.transpose();
}

template <typename T>
Matrix<T, Dynamic, Dynamic> updateBFGS(const Ref<const Matrix<T, Dynamic, Dynamic> >& B,
                                       const Ref<const Matrix<T, Dynamic, 1> >& s,
                                       const Ref<const Matrix<T, Dynamic, 1> >& y)
{
    /*
     * Compute the BFGS update formula (Nocedal and Wright, Eqn. 6.19).
     */
    return B - ((B * s * s.transpose() * B) / (s.dot(B * s)) + ((y * y.transpose()) / (y.dot(s)));
}

template <typename T>
LLT<Matrix<T, Dynamic, Dynamic> > updateBFGSCholesky(const Ref<const LLT<Matrix<T, Dynamic, Dynamic> > >& dec,
                                                     const Ref<const Matrix<T, Dynamic, 1> >& s,
                                                     const Ref<const Matrix<T, Dynamic, 1> >& y)
{
    /*
     * Compute the BFGS update formula (Nocedal and Wright, Eqn. 6.19) 
     * in terms of the given Cholesky (LLT) decomposition.
     */
    return dec.rankUpdate(dec.matrixLLT() * s, -1 / (s.dot(B * s))).rankUpdate(y, y.dot(s));
}

template <typename T>
LDLT<Matrix<T, Dynamic, Dynamic> > updateBFGSCholesky(const Ref<const LDLT<Matrix<T, Dynamic, Dynamic> > >& dec,
                                                      const Ref<const Matrix<T, Dynamic, 1> >& s,
                                                      const Ref<const Matrix<T, Dynamic, 1> >& y)
{
    /*
     * Compute the BFGS update formula (Nocedal and Wright, Eqn. 6.19) 
     * in terms of the given Cholesky (LDLT) decomposition.
     */
    return dec.rankUpdate(dec.matrixLDLT() * s, -1 / (s.dot(B * s))).rankUpdate(y, y.dot(s));
}

template <typename T>
Matrix<T, Dynamic, Dynamic> updateSR1Inv(const Ref<const Matrix<T, Dynamic, Dynamic> >& H,
                                         const Ref<const Matrix<T, Dynamic, 1> >& s,
                                         const Ref<const Matrix<T, Dynamic, 1> >& y)
{
    /*
     * Compute the SR1 update formula (Nocedal and Wright, Eqn. 6.25).
     */
    return H + ((s - H * y) * (s - H * y).transpose() / (s - H * y).dot(y);
}

template <typename T>
Matrix<T, Dynamic, Dynamic> updateSR1(const Ref<const Matrix<T, Dynamic, Dynamic> >& B,
                                      const Ref<const Matrix<T, Dynamic, 1> >& s,
                                      const Ref<const Matrix<T, Dynamic, 1> >& y)
{
    /*
     * Compute the SR1 update formula (Nocedal and Wright, Eqn. 6.24).
     */
    return B + ((y - B * s) * (y - B * s).transpose() / (y - B * s).dot(s);
}

#endif
