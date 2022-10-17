/**
 * An implementation of a line-search procedure for choosing step-sizes in 
 * nonlinear optimization that satisfy the (strong) Wolfe conditions, given by
 * Eqs. 3.6 and 3.7 in Nocedal and Wright.
 *
 * The optimization problem being considered is assumed to be *unconstrained*
 * within the given maximum step-size, i.e., even if the problem is constrained
 * as a whole, the constraints are always satisfied for every choice of
 * step-size between 0 and the maximum. This may be assumed if, for instance, 
 * the constraints define a convex region in the input space. 
 *
 * This implementation is an imitation of `scipy.optimize.line_search()`.
 *
 * **Authors:** 
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * 
 * **Last updated:**
 *     10/16/2022
 */

#ifndef SQP_OPTIMIZER_LINE_SEARCH_HPP
#define SQP_OPTIMIZER_LINE_SEARCH_HPP

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <Eigen/Dense>
#include <CGAL/QP_models.h>
#include <CGAL/QP_functions.h>
#include <boost/multiprecision/gmp.hpp>

using namespace Eigen;
using boost::multiprecision::mpq_rational;

/**
 * Test whether the Armijo condition (Nocedal and Wright, Eq. 3.6a) is 
 * satisfied by the given pair of old (pre-step) and new (post-step) solutions.
 *
 * @param dir      Candidate direction for updating the old solution.
 * @param stepsize Candidate step-size for updating the old solution.
 * @param f_old    Pre-computed value of objective at the old solution.
 * @param f_new    Pre-computed value of objective at the new solution.
 * @param grad_old Pre-computed value of gradient at the old solution.
 * @param c        Constant multiplier. 
 */
template <typename T>
bool wolfeArmijo(const Ref<const Matrix<T, Dynamic, 1> >& dir,
                 const T stepsize, const T f_old, const T f_new, 
                 const Ref<const Matrix<T, Dynamic, 1> >& grad_old, const T c)
{
    return (f_new <= f_old + c * stepsize * grad_old.dot(dir)); 
}

/**
 * Test whether the curvature condition (Nocedal and Wright, Eq. 3.6b) is
 * satisfied by the given pair of old (pre-step) and new (post-step) solutions.
 *
 * @param dir      Candidate direction for updating the old solution.
 * @param grad_old Pre-computed value of gradient at the old solution.
 * @param grad_new Pre-computed value of gradient at the new solution.
 * @param c        Constant multiplier. 
 */
template <typename T>
bool wolfeCurvature(const Ref<const Matrix<T, Dynamic, 1> >& dir,
                    const Ref<const Matrix<T, Dynamic, 1> >& grad_old,
                    const Ref<const Matrix<T, Dynamic, 1> >& grad_new,
                    const T c)
{
    return (grad_new.dot(dir) >= c * grad_old.dot(dir));
}

/**
 * Test whether the *strong* curvature condition (Nocedal and Wright, Eq. 3.7b)
 * is satisfied by the given pair of old (pre-step) and new (post-step) solutions.
 *
 * @param dir      Candidate direction for updating the old solution.
 * @param grad_old Pre-computed value of gradient at the old solution.
 * @param grad_new Pre-computed value of gradient at the new solution.
 * @param c        Constant multiplier. 
 */
template <typename T>
bool wolfeStrongCurvature(const Ref<const Matrix<T, Dynamic, 1> >& dir,
                          const Ref<const Matrix<T, Dynamic, 1> >& grad_old,
                          const Ref<const Matrix<T, Dynamic, 1> >& grad_new,
                          const T c)
{
    using std::abs;
    using boost::multiprecision::abs; 

    return (abs(grad_new.dot(dir)) <= c * abs(grad_old.dot(dir))); 
}

/**
 * Find the minimum for an interpolating quadratic polynomial that goes through
 * `(a, f_a)` and `(b, f_b)` with derivative at `a` of `df_a`.
 *
 * @param a    First x-value.
 * @param f_a  Value of interpolated function at `a`.
 * @param df_a Derivative of interpolated function at `a`.
 * @param b    Second x-value.
 * @param f_b  Value of interpolated function at `b`.
 * @returns Input value at which the quadratic interpolant is minimized.
 * @throws std::runtime_error If `a == b` or `f_b - f_a == df_a * (b - a)`. 
 */
template <typename T>
T minQuadraticInterpolant(T a, T f_a, T df_a, T b, T f_b)
{
    T D = f_a;
    T C = df_a;
    T delta = b - a;
    if (delta == 0)
    {
        throw std::runtime_error(
            "Quadratic interpolation failed: points have the same x-value"
        );
    }
    T B = (f_b - D - C * delta) / (delta * delta);
    if (B == 0)
    {
        throw std::runtime_error(
            "Quadratic interpolation failed: f_b - f_a - df_a * (b - a) == 0"
        );
    }
    return a - C / (2 * B);
}

/**
 * Find the minimum for an interpolating cubic polynomial that goes through
 * `(a, f_a)`, `(b, f_b)`, and `(c, f_c)` with derivative at `a` of `df_a`.
 *
 * @param a    First x-value.
 * @param f_a  Value of interpolated function at `a`.
 * @param df_a Derivative of interpolated function at `a`.
 * @param b    Second x-value.
 * @param f_b  Value of interpolated function at `b`.
 * @param c    Third x-value.
 * @param f_c  Value of interpolated function at `c`.
 * @returns Input value at which the cubic interpolant is minimized.
 * @throws std::runtime_error If `a == b` or `b == c` or `a == c` or a more
 *                            complicated internal divide-by-zero is encountered
 *                            (see below).
 */
template <typename T>
T minCubicInterpolant(T a, T f_a, T df_a, T b, T f_b, T c, T f_c)
{
    using std::sqrt;
    using boost::multiprecision::sqrt;

    T C = df_a; 
    T delta1 = b - a;
    T delta2 = c - a;
    T delta1_squared = delta1 * delta1;
    T delta2_squared = delta2 * delta2;
    T denom = (delta1_squared * delta2_squared) * (delta1 - delta2);
    if (denom == 0)
    {
        throw std::runtime_error(
            "Cubic interpolation failed: at least two points have the same x-value"
        );
    }

    Matrix<T, 2, 2> M;
    M << delta2_squared, -delta1_squared, -delta2_squared * delta2, delta1_squared * delta1;
    Matrix<T, 2, 1> v;
    v << f_b - f_a - C * delta1, f_c - f_a - C * delta2; 
    Matrix<T, 2, 1> prod = (M * v) / denom;
    T A = prod(0);  
    T B = prod(1); 
    T radical = B * B - 3 * A * C;
    if (A == 0)
    {
        throw std::runtime_error("Cubic interpolation failed due to divide-by-zero");
    }

    return a + (-B + sqrt(radical)) / (3 * A);
}

/**
 * An implementation of the zoom function (Algorithm 3.6 in Nocedal and Wright),
 * which chooses a stepsize that satisfies the strong Wolfe conditions.
 *
 * @param func         Objective function.
 * @param gradient     Function that evaluates the gradient of the objective.
 * @param x_curr       Input vector.
 * @param f_curr       Pre-computed value of objective at `x_curr`.
 * @param grad_curr    Pre-computed value of gradient at `x_curr`.
 * @param dir          Step vector.
 * @param stepsize_lo  Min/max allowed stepsize, together with `stepsize_hi`.
 * @param stepsize_hi  Min/max allowed stepsize, together with `stepsize_lo`.
 * @param stepsize_tol Tolerance between `stepsize_hi` and `stepsize_lo` (as
 *                     they are iteratively updated) for terminating stepsize
 *                     search. 
 * @param c1           Constant multiplier in Armijo condition.
 * @param c2           Constant multiplier in strong curvature condition.
 * @param max_iter     Maximum number of stepsize search iterations.
 * @param verbose      If true, output intermittent messages to `stdout`.
 * @returns New stepsize, together with indicators as to whether the 
 *          stepsize satisfies the Armijo and strong curvature conditions.
 */
template <typename T>
std::tuple<T, bool, bool> zoom(std::function<T(const Ref<const Matrix<T, Dynamic, 1> >&)> func,
                               std::function<Matrix<T, Dynamic, 1>(const Ref<const Matrix<T, Dynamic, 1> >&)> gradient,
                               const Ref<const Matrix<T, Dynamic, 1> >& x_curr, const T f_curr,
                               const Ref<const Matrix<T, Dynamic, 1> >& grad_curr,
                               const Ref<const Matrix<T, Dynamic, 1> >& dir,
                               T stepsize_lo, T stepsize_hi, const T stepsize_tol, 
                               const T c1, const T c2, const int max_iter,
                               const bool verbose)
{
    bool satisfies_armijo = false; 
    bool satisfies_curvature = false;
    T delta = stepsize_hi - stepsize_lo;
    T abs_delta = abs(delta);
    const T cubic_interpolate_tol_factor = 0.2; 
    const T quadratic_interpolate_tol_factor = 0.1;
    T stepsize_prev = 0; 
    
    int i = 0;
    while (abs_delta > stepsize_tol && i < max_iter)
    {
        if (verbose)
        {
            std::cout << "... zooming into stepsize interval between "
                      << stepsize_lo << " and " << stepsize_hi << std::endl;
        }

        // Determine new stepsize as some interpolant of the two bracketing
        // stepsizes
        //
        // We do this by interpolating \phi(\alpha) = func(x + \alpha * dir),
        // where \alpha is the stepsize, between the bracketing stepsizes
        //
        // If at least one iteration has been performed (i > 0), then first 
        // try cubic interpolation of \phi(\alpha) and check that the result
        // is not too close to either stepsize_lo or stepsize_hi
        //
        // If on the first iteration *or* the cubic interpolation above results
        // in a stepsize too close to either stepsize_lo or stepsize_hi, then 
        // try quadratic interpolation of \phi(\alpha) and check that that
        // result is not too close to either stepsize_lo or stepsize_hi
        //
        // If the quadratic interpolation above (also) results in a stepsize 
        // too close to either stepsize_lo or stepsize_hi, then simply take the 
        // average of stepsize_lo and stepsize_hi (bisection)
        T new_stepsize, bracket_tol;
        T phi_lo = func(x_curr + stepsize_lo * dir); 
        T phi_hi = func(x_curr + stepsize_hi * dir);
        T phi_prev = (i == 0 ? f_curr : func(x_curr + stepsize_prev * dir));
        T dphi_lo = dir.dot(gradient(x_curr + stepsize_lo * dir));
        if (i == 0)
        {
            try
            {
                new_stepsize = minQuadraticInterpolant(
                    stepsize_lo, phi_lo, dphi_lo, stepsize_hi, phi_hi
                );
                bracket_tol = quadratic_interpolate_tol_factor * abs_delta;
                if (abs(new_stepsize - stepsize_lo) < bracket_tol ||
                    abs(new_stepsize - stepsize_hi) < bracket_tol)
                {
                    new_stepsize = (stepsize_hi + stepsize_lo) / 2;
                }
            }
            catch (const std::runtime_error& e)
            {
                new_stepsize = (stepsize_hi + stepsize_lo) / 2;
            }
        }
        else
        {
            int interpolation_mode;   // 0 for bisection, 1 for quadratic, 2 for cubic
            try
            { 
                new_stepsize = minCubicInterpolant(
                    stepsize_lo, phi_lo, dphi_lo, stepsize_hi, phi_hi,
                    stepsize_prev, phi_prev
                );
                interpolation_mode = 2;
            }
            catch (const std::runtime_error& e)
            {
                try
                {
                    new_stepsize = minQuadraticInterpolant(
                        stepsize_lo, phi_lo, dphi_lo, stepsize_hi, phi_hi
                    );
                    interpolation_mode = 1;
                }
                catch (const std::runtime_error& e)
                {
                    new_stepsize = (stepsize_lo + stepsize_hi) / 2;
                    interpolation_mode = 0;
                }
            }
            if (interpolation_mode == 2)
            {
                bracket_tol = cubic_interpolate_tol_factor * abs_delta;
                if (abs(new_stepsize - stepsize_lo) < bracket_tol ||
                    abs(new_stepsize - stepsize_hi) < bracket_tol)
                {
                    try
                    {
                        new_stepsize = minQuadraticInterpolant(
                            stepsize_lo, phi_lo, dphi_lo, stepsize_hi, phi_hi
                        );
                        bracket_tol = quadratic_interpolate_tol_factor * abs_delta;
                        if (abs(new_stepsize - stepsize_lo) < bracket_tol ||
                            abs(new_stepsize - stepsize_hi) < bracket_tol)
                        {
                            new_stepsize = (stepsize_hi + stepsize_lo) / 2;
                        }
                    }
                    catch (const std::runtime_error& e)
                    {
                        new_stepsize = (stepsize_lo + stepsize_hi) / 2;
                    }
                }
            }
            else if (interpolation_mode == 1)
            {
                bracket_tol = quadratic_interpolate_tol_factor * abs_delta;
                if (abs(new_stepsize - stepsize_lo) < bracket_tol ||
                    abs(new_stepsize - stepsize_hi) < bracket_tol)
                {
                    new_stepsize = (stepsize_lo + stepsize_hi) / 2;
                }
            }
        }
        T phi_new = func(x_curr + new_stepsize * dir);
        T dphi_zero = dir.dot(grad_curr);
        if (verbose)
        {
            std::cout << "...... candidate stepsize = " << new_stepsize << std::endl;
        }
        
        // The quantities defined by Nocedal and Wright below are 
        // given here by:
        //
        // \alpha_j = new_stepsize
        // \alpha_{lo} = stepsize_lo
        // \alpha_{hi} = stepsize_hi
        // \phi(0) = f_curr
        // \phi'(0) = dphi_zero = dir.dot(grad_curr)
        // \phi(\alpha_j) = phi_new
        // \phi(\alpha_{lo}) = func(x_curr + stepsize_lo * dir) = phi_lo
        satisfies_armijo = wolfeArmijo<T>(dir, new_stepsize, f_curr, phi_new, grad_curr, c1);
        if (!satisfies_armijo)    // If the Armijo condition is not satisfied 
        {
            if (verbose)
            {
                std::cout << "...... violates Armijo condition" << std::endl;
            }
            stepsize_prev = stepsize_hi;
            stepsize_hi = new_stepsize;
        }
        else if (phi_new >= phi_lo)   // If the step yields an increase in the
        {                             // objective value relative to stepsize_lo
            if (verbose)
            {
                std::cout << "...... causes increase in objective relative to "
                          << "stepsize_lo = " << stepsize_lo << std::endl;
            }
            stepsize_prev = stepsize_hi;
            stepsize_hi = new_stepsize;
        }
        else 
        {
            Matrix<T, Dynamic, 1> grad_new = gradient(x_curr + new_stepsize * dir); 
            satisfies_curvature = wolfeStrongCurvature<T>(dir, grad_curr, grad_new, c2); 
            if (satisfies_curvature)   // If the strong curvature condition is satisfied
            {
                if (verbose)
                {
                    std::cout << "...... found stepsize satisfying both Armijo "
                              << "and strong curvature conditions" << std::endl;
                }
                return std::make_tuple(new_stepsize, true, true);   // Loop exits here
            }
            if (verbose)
            {
                std::cout << "...... violates strong curvature condition" << std::endl;
            }
            if (dir.dot(grad_new) * (stepsize_hi - stepsize_lo) >= 0)
            {
                stepsize_prev = stepsize_hi;
                stepsize_hi = stepsize_lo;
            }
            else 
            {
                stepsize_prev = stepsize_lo;
            }
            stepsize_lo = new_stepsize;
        }
        delta = stepsize_hi - stepsize_lo;
        abs_delta = abs(delta);
        i++;
    }

    // If stepsize has not yet been returned, take the average of stepsize_lo
    // and stepsize_hi (which should be relatively close or less than stepsize_tol
    // at this point) as the final stepsize
    T final_stepsize = (stepsize_hi + stepsize_lo) / 2; 
    T phi_final = func(x_curr + final_stepsize * dir);
    Matrix<T, Dynamic, 1> grad_final = gradient(x_curr + final_stepsize * dir); 
    satisfies_armijo = wolfeArmijo<T>(dir, final_stepsize, f_curr, phi_final, grad_curr, c1);
    satisfies_curvature = wolfeStrongCurvature<T>(dir, grad_curr, grad_final, c2);
    if (verbose)
    {
        std::cout << "... terminating zoom (interval between " << stepsize_lo
                  << " and " << stepsize_hi << ")" << std::endl
                  << "... final stepsize = " << final_stepsize
                  << "; Armijo = " << satisfies_armijo 
                  << "; strong curvature = " << satisfies_curvature << std::endl; 
    }
    return std::make_tuple(final_stepsize, satisfies_armijo, satisfies_curvature);
}

/**
 * An implementation of a line-search algorithm for choosing stepsizes 
 * (Nocedal and Wright, Algorithms 3.5 and 3.6).
 *
 * This algorithm identifies a stepsize between a minimum stepsize and a
 * maximum stepsize (usually some epsilon > 0 and 1.0, respectively) that
 * satisfies the *strong* Wolfe conditions, by taking two candidate stepsizes,
 * `stepsize0` and `stepsize1`, and choosing as the stepsize:
 * - `stepsize1` itself if `stepsize1` satisfies the strong Wolfe conditions;
 * - a value in `[stepsize0, stepsize1]` if *any* of the three conditions are
 *   satisfied:
 *   1) `stepsize1` violates the Armijo condition, 
 *   2) `func(x_curr + stepsize1 * dir) >= func(x_curr + stepsize0 * dir)`,
 *      meaning that `stepsize1` leads to an increased objective value relative
 *      to `stepsize0`, or
 *   3) the descent direction at `stepsize1` is nonnegative; *or*
 * - a value in (a subinterval of) `[stepsize1, 1.0]` by repeating the above
 *   checks with `stepsize1 <- min(2 * stepsize1, max_stepsize)` and
 *   `stepsize0 <- stepsize1` up to the given maximum number of iterations.
 *
 * @param func
 * @param gradient
 * @param x_curr
 * @param f_curr
 * @param f_prev
 * @param grad_curr
 * @param dir
 * @param min_stepsize
 * @param max_stepsize
 * @param c1
 * @param c2
 * @param max_iter
 * @param zoom_max_iter
 * @param verbose
 * @param zoom_verbose
 */
template <typename T>
std::tuple<T, bool, bool> lineSearch(std::function<T(const Ref<const Matrix<T, Dynamic, 1> >&)> func,
                                     std::function<Matrix<T, Dynamic, 1>(const Ref<const Matrix<T, Dynamic, 1> >&)> gradient,
                                     const Ref<const Matrix<T, Dynamic, 1> >& x_curr,
                                     const T f_curr, const T f_prev,
                                     const Ref<const Matrix<T, Dynamic, 1> >& grad_curr, 
                                     const Ref<const Matrix<T, Dynamic, 1> >& dir,
                                     T min_stepsize, T max_stepsize, const T c1,
                                     const T c2, const int max_iter,
                                     const int zoom_max_iter, const bool verbose,
                                     const bool zoom_verbose)
{
    using std::abs;
    using boost::multiprecision::abs;
    using std::min;
    using boost::multiprecision::min;
    using std::isnan;
    using boost::multiprecision::isnan;

    bool satisfies_armijo = false;
    bool satisfies_curvature = false;

    // Given the objective function f and direction vector p, Nocedal and
    // Wright define the following:
    //
    // - For any given stepsize \alpha, \phi(\alpha) is the function
    //   f(x + \alpha * p)
    // - Therefore, the derivative of \phi w.r.t. \alpha, \phi'(\alpha), is 
    //   given (through the multivariable chain rule) by the dot product 
    //   of p and the gradient of f at x + \alpha * p
    // - Therefore, \phi'(0) is simply the dot product of p and the gradient
    //   of f at x
    T dphi0 = dir.dot(grad_curr);

    // Initialize the bracketing stepsizes 
    T stepsize0 = min_stepsize;
    T stepsize1, stepsize;
    if (isnan(f_prev))
        stepsize1 = max_stepsize;
    else
    {
        T trial_stepsize = 1.01 * 2 * (f_curr - f_prev) / dphi0;
        if (trial_stepsize > 0)
            stepsize1 = trial_stepsize; 
        else 
            stepsize1 = max_stepsize;
    }

    T phi0 = func(x_curr + stepsize0 * dir);
    T phi1 = func(x_curr + stepsize1 * dir);
    for (int i = 1; i <= max_iter; ++i)
    {
        if (verbose)
        {
            std::cout << "... searching for stepsize between "
                      << stepsize0 << " and " << stepsize1 << std::endl;
        }

        // Does stepsize1 satisfy the Armijo condition? 
        if (!wolfeArmijo<T>(dir, stepsize1, f_curr, phi1, grad_curr, c1))
        {
            // If it does *not*, then the interval between stepsize0 and
            // stepsize1 must contain stepsizes that satisfy both strong
            // Wolfe conditions
            //
            // In this case, stepsize0 should still satisfy the Armijo
            // condition (and hence can be passed first as "\alpha_lo")
            auto result = zoom(
                func, gradient, x_curr, f_curr, grad_curr, dir, stepsize0,
                stepsize1, min_stepsize, c1, c2, zoom_max_iter, zoom_verbose
            );
            stepsize = std::get<0>(result);
            satisfies_armijo = std::get<1>(result); 
            satisfies_curvature = std::get<2>(result);  
            return std::make_tuple(stepsize, satisfies_armijo, satisfies_curvature);
        }
        // Or does stepsize1 lead to an increased objective value relative
        // to stepsize0?
        else if (phi1 >= phi0 && i > 1)
        {
            // If it does, then again the interval between stepsize0 and 
            // stepsize1 must contain stepsizes that satisfy both strong
            // Wolfe conditions
            //
            // In this case, both stepsize0 and stepsize1 should satisfy
            // the Armijo condition (and hence either can be passed first
            // as "\alpha_lo")
            auto result = zoom(
                func, gradient, x_curr, f_curr, grad_curr, dir, stepsize0,
                stepsize1, min_stepsize, c1, c2, zoom_max_iter, zoom_verbose
            );
            stepsize = std::get<0>(result);
            satisfies_armijo = std::get<1>(result); 
            satisfies_curvature = std::get<2>(result);  
            return std::make_tuple(stepsize, satisfies_armijo, satisfies_curvature);
        }
        
        // Otherwise, does stepsize1 satisfy the strong curvature condition?
        Matrix<T, Dynamic, 1> grad_new = gradient(x_curr + stepsize1 * dir);
        if (wolfeStrongCurvature<T>(dir, grad_curr, grad_new, c2))
        {
            // If it does, then seeing as stepsize1 must also satisfy
            // the Armijo condition (see above), stepsize1 satisfies 
            // both strong Wolfe conditions 
            return std::make_tuple(stepsize1, true, true);
        }

        // Otherwise, is the descent direction at x_curr + stepsize1 * dir
        // nonnegative?
        if (dir.dot(grad_new) >= 0)
        {
            // If it is, then the interval between stepsize0 and stepsize1
            // must contain stepsizes that satisfy both strong Wolfe
            // conditions
            //
            // In this case, stepsize1 should satisfy the Armijo condition
            // (and hence can be passed first as "\alpha_lo")
            auto result = zoom(
                func, gradient, x_curr, f_curr, grad_curr, dir, stepsize1,
                stepsize0, min_stepsize, c1, c2, zoom_max_iter, zoom_verbose
            );
            stepsize = std::get<0>(result);
            satisfies_armijo = std::get<1>(result); 
            satisfies_curvature = std::get<2>(result);  
            return std::make_tuple(stepsize, satisfies_armijo, satisfies_curvature);
        }

        // Update stepsize0 and stepsize1 in the same manner as in
        // scipy.optimize.line_search() (or, more specifically, see 
        // scipy.optimize._linesearch.scalar_search_wolfe2())
        T stepsize2 = min(2 * stepsize1, max_stepsize);
        stepsize0 = stepsize1; 
        stepsize1 = stepsize2;

        // Re-evaluate func(x_curr + stepsize0 * dir) and func(x_curr + stepsize1 * dir)
        phi0 = func(x_curr + stepsize0 * dir);
        phi1 = func(x_curr + stepsize1 * dir);
    }

    // Return stepsize1 if a stepsize has not yet been returned
    satisfies_armijo = wolfeArmijo<T>(dir, stepsize1, f_curr, phi1, grad_curr, c1);
    satisfies_curvature = wolfeStrongCurvature<T>(
        dir, grad_curr, gradient(x_curr + stepsize1 * dir), c2
    );
    return std::make_tuple(stepsize1, satisfies_armijo, satisfies_curvature);
}

#endif 
