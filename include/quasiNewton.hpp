#ifndef QUASI_NEWTON_HPP
#define QUASI_NEWTON_HPP
#include <Eigen/Dense>

/*
 * Implementations of a selection of quasi-Newton update formulas. 
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     3/15/2021
 */
using namespace Eigen;

template <typename T>
Matrix<T, Dynamic, Dynamic> updateBFGSInv(Matrix<T, Dynamic, Dynamic>& H,
                                          const Ref<const Matrix<T, Dynamic, 1> >& s,
                                          const Ref<const Matrix<T, Dynamic, 1> >& y)
{
    /*
     * Compute the BFGS update formula (Nocedal and Wright, Eqn. 6.17).
     */
    Matrix<T, Dynamic, Dynamic> I = Matrix<T, Dynamic, Dynamic>::Identity(H.rows(), H.cols());
    T rho = 1.0 / y.dot(s);
    Matrix<T, Dynamic, Dynamic> J = (I - rho * s * y.transpose()) * H * (I - rho * y * s.transpose());
    return J.template selfadjointView<Lower>().rankUpdate(s, rho);
}

template <typename T>
Matrix<T, Dynamic, Dynamic> updateBFGS(Matrix<T, Dynamic, Dynamic>& B,
                                       const Ref<const Matrix<T, Dynamic, 1> >& s,
                                       const Ref<const Matrix<T, Dynamic, 1> >& y)
{
    /*
     * Compute the BFGS update formula (Nocedal and Wright, Eqn. 6.19).
     */
    return B.template selfadjointView<Lower>().rankUpdate(B * s, -1.0 / s.dot(B * s)).rankUpdate(y, 1.0 / y.dot(s));
}

template <typename T>
Matrix<T, Dynamic, Dynamic> updateBFGSDamped(Matrix<T, Dynamic, Dynamic>& B,
                                             const Ref<const Matrix<T, Dynamic, 1> >& s,
                                             const Ref<const Matrix<T, Dynamic, 1> >& y)
{
    /*
     * Compute the damped BFGS update formula (Nocedal and Wright, Eqn. 18.16).
     */
    double theta;
    if (s.dot(y) >= 0.2 * s.dot(B * s))
        theta = 1.0;
    else
        theta = (0.8 * s.dot(B * s)) / (s.dot(B * s) - s.dot(y));

    Matrix<T, Dynamic, 1> r = theta * y + (1 - theta) * B * s; 
    return B.template selfadjointView<Lower>().rankUpdate(B * s, -1.0 / s.dot(B * s)).rankUpdate(r, 1.0 / s.dot(r)); 
}

template <typename T>
LLT<Matrix<T, Dynamic, Dynamic> > updateBFGSCholesky(LLT<Matrix<T, Dynamic, Dynamic> >& dec,
                                                     const Ref<const Matrix<T, Dynamic, 1> >& s,
                                                     const Ref<const Matrix<T, Dynamic, 1> >& y)
{
    /*
     * Compute the BFGS update formula (Nocedal and Wright, Eqn. 6.19) 
     * in terms of the given Cholesky (LLT) decomposition.
     */
    return dec.rankUpdate(dec.matrixLLT() * s, -1.0 / (s.dot(dec.matrixLLT() * s))).rankUpdate(y, 1.0 / y.dot(s));
}

template <typename T>
LDLT<Matrix<T, Dynamic, Dynamic> > updateBFGSCholesky(LDLT<Matrix<T, Dynamic, Dynamic> >& dec,
                                                      const Ref<const Matrix<T, Dynamic, 1> >& s,
                                                      const Ref<const Matrix<T, Dynamic, 1> >& y)
{
    /*
     * Compute the BFGS update formula (Nocedal and Wright, Eqn. 6.19) 
     * in terms of the given Cholesky (LDLT) decomposition.
     */
    return dec.rankUpdate(dec.matrixLDLT() * s, -1.0 / (s.dot(dec.matrixLDLT() * s))).rankUpdate(y, 1.0 / y.dot(s));
}

template <typename T>
Matrix<T, Dynamic, Dynamic> updateSR1Inv(Matrix<T, Dynamic, Dynamic>& H,
                                         const Ref<const Matrix<T, Dynamic, 1> >& s,
                                         const Ref<const Matrix<T, Dynamic, 1> >& y)
{
    /*
     * Compute the SR1 update formula (Nocedal and Wright, Eqn. 6.25).
     */
    return H.template selfadjointView<Lower>().rankUpdate(s - H * y, 1.0 / (s - H * y).dot(y));
}

template <typename T>
Matrix<T, Dynamic, Dynamic> updateSR1(Matrix<T, Dynamic, Dynamic>& B,
                                      const Ref<const Matrix<T, Dynamic, 1> >& s,
                                      const Ref<const Matrix<T, Dynamic, 1> >& y)
{
    /*
     * Compute the SR1 update formula (Nocedal and Wright, Eqn. 6.24).
     */
    return B.template selfadjointView<Lower>().rankUpdate(y - B * s, 1.0 / (y - B * s).dot(s));
}

#endif
