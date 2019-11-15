#ifndef DUAL_NUMBERS_HPP
#define DUAL_NUMBERS_HPP

#include <iostream>
#include <cmath>
#include <stdexcept>

/*
 * A lightweight implementation of dual numbers with 64-bit components.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     11/14/2019
 */

namespace Duals {

class DualNumber
{
    private:
        double x_;    // Real part
        double d_;    // Multiplier of epsilon (nilpotent part)

    public:
        DualNumber()
        {
            /*
             * Empty constructor; initialize to zero.
             */
            this->x_ = 0.0;
            this->d_ = 0.0;
        }

        DualNumber(double x)
        {
            /*
             * Constructor from double; initialize nilpotent part to zero.
             */
            this->x_ = x;
            this->d_ = 0.0;
        }

        DualNumber(double x, double d)
        {
            /*
             * Trivial constructor. 
             */
            this->x_ = x;
            this->d_ = d;
        }

        ~DualNumber()
        {
            /*
             * Trivial destructor.
             */
        }

        double x() const
        {
            /*
             * Return the real part.
             */
            return this->x_;
        }

        double d() const
        {
            /*
             * Return the nilpotent part.
             */
            return this->d_;
        }

        DualNumber& operator=(DualNumber v)
        {
            /*
             * Trivial assignment operator.
             */
            this->x_ = v.x();
            this->d_ = v.d();
            return *this;
        }

        DualNumber& operator=(double v)
        {
            /*
             * Assignment from double.
             */
            this->x_ = v;
            this->d_ = 0.0;
            return *this;
        }

        DualNumber operator+(const DualNumber& v) const
        {
            /*
             * Return the result of adding v.
             */
            return DualNumber(this->x_ + v.x(), this->d_ + v.d());
        }

        DualNumber operator-(const DualNumber& v) const
        {
            /*
             * Return the result of subtracting v.
             */
            return DualNumber(this->x_ - v.x(), this->d_ - v.d());
        }

        DualNumber operator*(const DualNumber& v) const
        {
            /*
             * Return the result of multiplying by v.
             */
            return DualNumber(this->x_ * v.x(), this->x_ * v.d() + this->d_ * v.x());
        }

        DualNumber operator/(const DualNumber& v) const
        {
            /*
             * Return the result of dividing by v.
             */
            if (v.x() == 0)
                throw std::domain_error("Division by zero");
            return DualNumber(this->x_ / v.x(), (this->d_ * v.x() - this->x_ * v.d()) / (v.x() * v.x()));
        }

        DualNumber& operator+=(const DualNumber& v)
        {
            /*
             * In-place addition by v.
             */
            this->x_ += v.x();
            this->d_ += v.d();
            return *this;
        }

        DualNumber& operator-=(const DualNumber& v)
        {
            /*
             * In-place subtraction by v.
             */
            this->x_ -= v.x();
            this->d_ -= v.d();
            return *this;
        }

        DualNumber& operator*=(const DualNumber& v)
        {
            /*
             * In-place multiplication by v.
             */
            this->x_ *= v.x();
            this->d_ = this->x_ * v.d() + this->d_ * v.x();
            return *this;
        }

        DualNumber& operator/=(const DualNumber& v)
        {
            /*
             * In-place division by v.
             */
            if (v.x() == 0)
                throw std::domain_error("Division by zero");
            this->x_ /= v.x();
            this->d_ = (this->d_ * v.x() - this->x_ * v.d()) / (v.x() * v.x());
            return *this;
        }
};

int sign(double x)
{
    /*
     * Return the sign of x.
     */
    if (x > 0)      return 1;
    else if (x < 0) return -1;
    else            return 0;
}

DualNumber pow(const DualNumber& v, const double p)
{
    /*
     * Return v raised to the power of p.
     */
    return DualNumber(std::pow(v.x(), p), p * std::pow(v.x(), p - 1.0) * v.d());
}

DualNumber exp(const DualNumber& v)
{
    /*
     * Return exp(v).
     */
    return DualNumber(std::exp(v.x()), std::exp(v.x()) * v.d());
}

DualNumber log(const DualNumber& v)
{
    /*
     * Return the natural log of v.
     */
    return DualNumber(std::log(v.x()), v.d() / v.x());
}

DualNumber sqrt(const DualNumber& v)
{
    /*
     * Return the square root of v.
     */
    return DualNumber(std::sqrt(v.x()), 0.5 * v.d() / std::sqrt(v.x()));
}

DualNumber abs(const DualNumber& v)
{
    /*
     * Return the absolute value of v.
     */
    return DualNumber(std::abs(v.x()), sign(v.x()) * v.d()); 
}

DualNumber sin(const DualNumber& v)
{
    /*
     * Return the sine of v.
     */
    return DualNumber(std::sin(v.x()), std::cos(v.x()) * v.d());
}

DualNumber cos(const DualNumber& v)
{
    /*
     * Return the cosine of v.
     */
    return DualNumber(std::cos(v.x()), (-std::sin(v.x())) * v.d());
}

DualNumber tan(const DualNumber& v)
{
    /*
     * Return the tangent of v.
     */
    return DualNumber(std::tan(v.x()), v.d() / (std::cos(v.x()) * std::cos(v.x())));
}

DualNumber real(const DualNumber& v)
{
    /*
     * Return v itself.
     */
    return DualNumber(v.x(), v.d());
}

DualNumber imag(const DualNumber& v)
{
    /*
     * Return zero.
     */
    return DualNumber(0.0, 0.0);
}

DualNumber conj(const DualNumber& v)
{
    /*
     * Return v itself.
     */
    return DualNumber(v.x(), v.d());
}

DualNumber abs2(const DualNumber& v)
{
    /*
     * Return v squared.
     */
    return DualNumber(v.x() * v.x(), 2 * v.x() * v.d());
}

}   // namespace Duals

#endif
