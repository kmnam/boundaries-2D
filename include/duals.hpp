/**
 * An implementation of a dual number class for forward-mode automatic 
 * differentiation. 
 *
 * **Authors**:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated**:
 *     5/5/2022
 */
#ifndef DUAL_NUMBERS_HPP
#define DUAL_NUMBERS_HPP
#include <iostream>
#include <cmath>
#include <algorithm>
#include <boost/multiprecision/gmp.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <Eigen/Dense>

/**
 * A template-based implementation of a dual number system. 
 */
template <typename T>
class Dual
{
    public:
        T a;   /** Real part of the dual number (publicly accessible).       */ 
        T b;   /** Derivative part of the dual number (publicly accessible). */

        /**
         * Empty constructor; initialize both parts to zero.
         */
        Dual()
        {
            this->a = 0; 
            this->b = 1; 
        }

        /**
         * Constructor that specifies only the real part; set `this->b = 0`.
         *
         * @param a Real part.
         */
        Dual(const T a)
        {
            this->a = a; 
            this->b = 1; 
        }

        /**
         * Constructor that specifies both real and derivative parts.
         *
         * @param a Real part.
         * @param b Derivative part. 
         */
        Dual(const T a, const T b)
        {
            this->a = a;
            this->b = b;
        }

        /**
         * Copy constructor.
         *
         * @param x Dual number to be copied over. 
         */
        Dual(const Dual<T>& x) : Dual<T>(x.a, x.b)
        {
        }

        /**
         * Trivial destructor.
         */
        ~Dual()
        {
        }

        // TODO Implement move constructor
        
        /**
         * Copy assignment operator.
         *
         * @param x Another dual number (to which `this` is to be assigned).
         * @returns Reference to `this`.
         */
        Dual<T>& operator=(const Dual<T>& x)
        {
            this->a = x.a; 
            this->b = x.b;
            return *this; 
        }

        /**
         * Compound addition by another dual number (with potentially different
         * scalar type).
         *
         * @param x Another dual number.
         * @returns Reference to `this`.
         */
        template <typename U>
        Dual<T>& operator+=(const Dual<U>& x)
        {
            this->a += x.a; 
            this->b += x.b;
            return *this; 
        }

        /**
         * Compound addition by a real-valued scalar with type T.
         *
         * @param x Real-valued scalar. 
         * @returns Reference to `this`.
         */
        Dual<T>& operator+=(const T& x)
        {
            this->a += x; 
            return *this;
        }

        /**
         * Compound subtraction by another dual number (with potentially different
         * scalar type).
         *
         * @param x Another dual number.
         * @returns Reference to `this`.
         */
        template <typename U>
        Dual<T>& operator-=(const Dual<U>& x)
        {
            this->a -= x.a; 
            this->b -= x.b;
            return *this; 
        }

        /**
         * Compound subtraction by a real-valued scalar with type T. 
         *
         * @param x Real-valued scalar. 
         * @returns Reference to `this`.
         */
        Dual<T>& operator-=(const T& x)
        {
            this->a -= x; 
            return *this;
        }

        /**
         * Negation.
         *
         * @returns Reference to `this`.
         */
        Dual<T>& operator-()
        {
            this->a = -this->a;
            this->b = -this->b;
            return *this;
        }

        /**
         * Compound multiplication by another dual number (with potentially
         * different scalar type).
         *
         * @param x Another dual number. 
         * @returns Reference to `this`. 
         */
        template <typename U>
        Dual<T>& operator*=(const Dual<U>& x)
        {
            this->a *= x.a; 
            this->b = (this->a * x.b) + (this->b * x.a); 
            return *this; 
        }

        /**
         * Compound multiplication by a real-valued scalar with type T.
         *
         * @param x Real-valued scalar.
         * @returns Reference to `this`. 
         */
        Dual<T>& operator*=(const T& x)
        {
            this->a *= x;
            this->b *= x; 
            return *this; 
        }

        /**
         * Compound division by another dual number (with potentially different
         * scalar type).
         *
         * @param x Another dual number.
         * @returns Reference to `this`.
         * @throws std::runtime_error If `x.a` is zero. 
         */
        template <typename U>
        Dual<T>& operator/=(const Dual<U>& x)
        {
            if (x.a == 0)
                throw std::runtime_error("Division by zero"); 
            this->a /= x.a; 
            this->b = ((this->b * x.a) - (this->a * x.b)) / (x.a * x.a); 
            return *this; 
        }

        /**
         * Compound division by a real-valued scalar with type T.
         *
         * @param x Real-valued scalar.
         * @returns Reference to `this`.
         * @throws std::runtime_error If `x` is zero. 
         */
        Dual<T>& operator/=(const T& x)
        {
            if (x == 0)
                throw std::runtime_error("Division by zero"); 
            this->a /= x;
            this->b /= x;
            return *this; 
        }

        /**
         * Exponentiation.
         *
         * @returns The exponential of `this`.
         */
        Dual<T> exp()
        {
            using std::exp;
            using boost::multiprecision::exp;

            T e = exp(this->a); 
            return Dual<T>(e, this->b * e); 
        }

        /**
         * Natural logarithm.
         *
         * @returns The natural logarithm of `this`. 
         */
        Dual<T> log()
        {
            using std::log;
            using boost::multiprecision::log;

            return Dual<T>(log(this->a), this->b / this->a); 
        }

        /**
         * Base-10 logarithm.
         *
         * @returns The base-10 logarithm of `this`.
         */
        Dual<T> log10()
        {
            using std::log; 
            using std::log10;
            using boost::multiprecision::log;  
            using boost::multiprecision::log10;

            T ten = 10; 
            return Dual<T>(log10(this->a), this->b / (this->a * log(ten))); 
        }

        /**
         * Square root.
         *
         * @returns The square root of `this`.
         */
        Dual<T> sqrt()
        {
            using std::sqrt;
            using boost::multiprecision::sqrt;

            T s = sqrt(this->a);
            return Dual<T>(s, this->b / (2 * s)); 
        }

        /**
         * Sine function.
         *
         * @returns The sine of `this`.
         */
        Dual<T> sin()
        {
            using std::sin;
            using std::cos;
            using boost::multiprecision::sin;
            using boost::multiprecision::cos; 

            return Dual<T>(sin(this->a), this->b * cos(this->a));
        }

        /**
         * Cosine function.
         *
         * @returns The cosine of `this`.
         */
        Dual<T> cos()
        {
            using std::sin;
            using std::cos;
            using boost::multiprecision::sin;
            using boost::multiprecision::cos; 

            return Dual<T>(cos(this->a), -this->b * sin(this->a)); 
        }

        /**
         * Tangent function.
         *
         * @returns The tangent of `this`.
         */
        Dual<T> tan()
        {
            using std::tan;
            using std::cos;
            using boost::multiprecision::tan;
            using boost::multiprecision::cos; 

            T c = cos(this->a); 
            return Dual<T>(tan(this->a), this->b / (c * c)); 
        }

        /**
         * Absolute value.
         *
         * @returns The absolute value of `this`.
         * @throws std::runtime_error If `this->a` is zero. 
         */
        Dual<T> abs()
        {
            if (this->a == 0)
                throw std::runtime_error("Absolute value of zero has undefined derivative"); 
            else if (this->a > 0)
                return Dual<T>(this->a, this->b);
            else
                return Dual<T>(-this->a, -this->b);  
        }
};

/**
 * Addition by another dual number (with potentially different scalar type).
 *
 * @param x One dual number.
 * @param y Another dual number. 
 * @returns The sum of `x` and `y`. 
 */
template <typename T, typename U>
inline Dual<T> operator+(const Dual<T>& x, const Dual<U>& y)
{
    Dual<T> z(x);
    z += y;
    return z;  
}

/**
 * Addition by a real-valued scalar with type T.
 *
 * @param x Dual number.
 * @param y Real-valued scalar.
 * @returns The sum of `x` and `y`. 
 */
template <typename T>
inline Dual<T> operator+(const Dual<T>& x, const T& y) 
{
    Dual<T> z(x);
    z += y;
    return z;  
}

/**
 * Addition *to* a real-valued scalar with type T.
 *
 * @param x Real-valued scalar.
 * @param y Dual number.
 * @returns The sum of `x` and `y`.
 */
template <typename T>
inline Dual<T> operator+(const T& x, const Dual<T>& y)
{
    Dual<T> z(y); 
    z += x; 
    return z;
}

/**
 * Subtraction by another dual number (with potentially different scalar type).
 *
 * @param x One dual number.
 * @param y Another dual number. 
 * @returns The difference of `x` and `y`. 
 */
template <typename T, typename U>
inline Dual<T> operator-(const Dual<T>& x, const Dual<U>& y)
{
    Dual<T> z(x);
    z -= y;
    return z;  
}

/**
 * Subtraction by a real-valued scalar with type T.
 *
 * @param x Dual number.
 * @param y Real-valued scalar.
 * @returns The difference of `x` and `y`. 
 */
template <typename T>
inline Dual<T> operator-(const Dual<T>& x, const T& y) 
{
    Dual<T> z(x);
    z -= y; 
    return z; 
}

/**
 * Subtraction *from* a real-valued scalar with type T.
 *
 * @param x Real-valued scalar.
 * @param y Dual number.
 * @returns The difference of `x` and `y`. 
 */
template <typename T>
inline Dual<T> operator-(const T& x, const Dual<T>& y) 
{
    Dual<T> z; 
    z.a = x - y.a;
    z.b = -y.b; 
    return z; 
}

/**
 * Multiplication by another dual number (with potentially different scalar type).
 *
 * @param x One dual number.
 * @param y Another dual number.
 * @param The product of `x` and `y`.
 */
template <typename T, typename U>
inline Dual<T> operator*(const Dual<T>& x, const Dual<U>& y)
{
    Dual<T> z(x);
    z *= y;
    return z; 
}

/**
 * Multiplication by a real-valued scalar with type T. 
 *
 * @param x Dual number.
 * @param y Real-valued scalar.
 * @returns The product of `x` and `y`. 
 */
template <typename T>
inline Dual<T> operator*(const Dual<T>& x, const T& y) 
{
    Dual<T> z(x);
    z *= y;
    return z; 
}

/**
 * Multiplication *of* a real-valued scalar with type T.
 *
 * @param x Real-valued scalar.
 * @param y Dual number.
 * @returns The product of `x` and `y`. 
 */
template <typename T>
inline Dual<T> operator*(const T& x, const Dual<T>& y) 
{
    Dual<T> z(y);
    z *= x;
    return z; 
}

/**
 * Division by another dual number (with potentially different scalar type). 
 *
 * @param x One dual number.
 * @param y Another dual number. 
 * @returns The quotient `x / y`.
 */
template <typename T, typename U>
inline Dual<T> operator/(const Dual<T>& x, const Dual<U>& y)
{
    Dual<T> z(x);
    z /= y; 
    return z; 
}

/**
 * Division by a real-valued scalar with type T.
 *
 * @param x Dual number.
 * @param y Real-valued scalar.
 * @returns The quotient `x / y`.
 */
template <typename T>
inline Dual<T> operator/(const Dual<T>& x, const T& y) 
{
    Dual<T> z(x);
    z /= y;
    return z;  
}

/**
 * Division *of* a real-valued scalar with type T.
 *
 * @param x Real-valued scalar.
 * @param y Dual number.
 * @returns The quotient `x / y`.
 */
template <typename T>
inline Dual<T> operator/(const T& x, const Dual<T>& y)
{
    Dual<T> z;
    z.a = x / y.a;
    z.b = -(y.b * x / (y.a * y.a));
    return z; 
}

/**
 * Equality operator.
 *
 * @param x One dual number.
 * @param y Another dual number.
 * @returns True if both parts are equal, false otherwise.
 */
template <typename T>
inline bool operator==(const Dual<T>& x, const Dual<T>& y)
{
    return (x.a == y.a && x.b == y.b); 
}

/**
 * Inequality operator.
 *
 * @param x One dual number.
 * @param y Another dual number.
 * @returns True if one or both parts are not equal, false otherwise.
 */
template <typename T>
inline bool operator!=(const Dual<T>& x, const Dual<T>& y)
{
    return !(x == y); 
}

/**
 * Greater-than operator.
 *
 * @param x One dual number.
 * @param y Another dual number.
 * @returns True if the real part of `x` is greater than the real part of `y`,
 *          false otherwise.
 */
template <typename T>
inline bool operator>(const Dual<T>& x, const Dual<T>& y)
{
    return (x.a > y.a); 
}

/**
 * Less-than operator.
 *
 * @param x One dual number.
 * @param y Another dual number.
 * @returns True if the real part of `x` is less than the real part of `y`,
 *          false otherwise.
 */
template <typename T>
inline bool operator<(const Dual<T>& x, const Dual<T>& y)
{
    return (y > x); 
}

/**
 * Greater-than-or-equal-to operator.
 *
 * @param x One dual number.
 * @param y Another dual number.
 * @returns True if the real part of `x` is greater than or equal to the real
 *          part of `y`, false otherwise.
 */
template <typename T>
inline bool operator>=(const Dual<T>& x, const Dual<T>& y)
{
    return (x == y || x > y); 
}

/**
 * Less-than-or-equal-to operator.
 *
 * @param x One dual number.
 * @param y Another dual number.
 * @returns True if the real part of `x` is less than or equal to the real part
 *          of `y`, false otherwise.
 */
template <typename T>
inline bool operator<=(const Dual<T>& x, const Dual<T>& y)
{
    return (x == y || y > x); 
}

/**
 * Exponentiation of a dual number.
 *
 * @param x Dual number.
 * @returns The exponential of `x`.
 */
template <typename T>
inline Dual<T> exp(Dual<T> x)
{
    return x.exp(); 
}

/**
 * Raise (dual number) `x` to the power of (scalar) `y`.
 *
 * @param x Dual number.
 * @param y Real-valued scalar.
 * @returns `x` raised to the power of `y`.
 */
template <typename T, typename U>
Dual<T> pow(Dual<T> x, U y)
{
    using std::pow;
    using boost::multiprecision::pow;

    T power = static_cast<T>(y); 
    return Dual<T>(pow(x.a, power), x.b * power * pow(x.a, power - 1));
}

/**
 * Raise (scalar) `x` to the power of (dual number) `y`.
 *
 * @param x Real-valued scalar. 
 * @param y Dual number.
 * @returns `x` raised to the power of `y`.
 */
template <typename T, typename U>
Dual<T> pow(U x, Dual<T> y)
{
    using std::pow;
    using std::log; 
    using boost::multiprecision::pow;
    using boost::multiprecision::log; 

    T base = static_cast<T>(x); 
    T p = pow(base, y.a);
    return Dual<T>(p, y.b * log(base) * p); 
}

/**
 * Natural logarithm of a dual number.
 *
 * @param x Dual number.
 * @returns The natural logarithm of `x`. 
 */
template <typename T>
inline Dual<T> log(Dual<T> x)
{
    return x.log();
}

/**
 * Base-10 logarithm of a dual number.
 *
 * @param x Dual number.
 * @returns The base-10 logarithm of `x`.
 */
template <typename T>
inline Dual<T> log10(Dual<T> x)
{
    return x.log10(); 
}

/**
 * Square root of a dual number.
 *
 * @param x Dual number.
 * @returns The square root of `x`.
 */
template <typename T>
inline Dual<T> sqrt(Dual<T> x)
{
    return x.sqrt(); 
}

/**
 * Sine of a dual number.
 *
 * @param x Dual number.
 * @returns The sine of `x`.
 */
template <typename T>
inline Dual<T> sin(Dual<T> x)
{
    return x.sin(); 
}

/**
 * Cosine of a dual number.
 *
 * @param x Dual number.
 * @returns The cosine of `x`.
 */
template <typename T>
inline Dual<T> cos(Dual<T> x)
{
    return x.cos(); 
}

/**
 * Tangent of a dual number.
 *
 * @param x Dual number.
 * @returns The tangent of `x`.
 */
template <typename T>
inline Dual<T> tan(Dual<T> x)
{
    return x.tan(); 
}

/**
 * Absolute value of a dual number.
 *
 * @param x Dual number.
 * @returns The absolute value of `x`.
 */
template <typename T>
inline Dual<T> abs(Dual<T> x)
{
    return x.abs(); 
}

/**
 * Eigen `NumTraits` and `ScalarBinaryOpTraits` specializations for various
 * `Dual` types.
 */
namespace Eigen {

template <>
struct NumTraits<Dual<float> > : NumTraits<float>
{
    typedef Dual<float> Real; 
    typedef Dual<float> NonInteger; 
    typedef Dual<float> Nested;

    enum
    {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 3,
        MulCost = 3
    }; 
};

template <>
struct NumTraits<Dual<double> > : NumTraits<double>
{
    typedef Dual<double> Real; 
    typedef Dual<double> NonInteger; 
    typedef Dual<double> Nested;

    enum
    {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 3,
        MulCost = 3
    }; 
};

template <>
struct NumTraits<Dual<long double> > : NumTraits<long double>
{
    typedef Dual<long double> Real; 
    typedef Dual<long double> NonInteger; 
    typedef Dual<long double> Nested;

    enum
    {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 3,
        MulCost = 3
    }; 
};

template <int Precision, boost::multiprecision::expression_template_option ExpressionTemplates>
struct NumTraits<Dual<boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<Precision>, ExpressionTemplates> > >
    : NumTraits<boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<Precision>, ExpressionTemplates> >
{
    typedef Dual<boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<Precision>, ExpressionTemplates> > Real; 
    typedef Dual<boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<Precision>, ExpressionTemplates> > NonInteger;
    typedef Dual<boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<Precision>, ExpressionTemplates> > Nested; 

    enum
    {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 2,    // Double that of boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<Precision>, ...>
        AddCost = 8,
        MulCost = 16
    }; 
};

template <>
struct NumTraits<Dual<boost::multiprecision::mpq_rational> > : NumTraits<boost::multiprecision::mpq_rational>
{
    typedef Dual<boost::multiprecision::mpq_rational> Real; 
    typedef Dual<boost::multiprecision::mpq_rational> NonInteger;
    typedef Dual<boost::multiprecision::mpq_rational> Nested; 

    enum
    {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 2,    // Double that of boost::multiprecision::mpq_rational
        AddCost = 8,
        MulCost = 16
    }; 
};

template <typename T, typename BinaryOp>
struct ScalarBinaryOpTraits<Dual<T>, T, BinaryOp> { typedef Dual<T> ReturnType; }; 

template <typename T, typename BinaryOp>
struct ScalarBinaryOpTraits<T, Dual<T>, BinaryOp> { typedef Dual<T> ReturnType; }; 

}   // namespace Eigen

#endif
