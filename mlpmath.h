#ifndef MLPMATH
#define MLPMATH

#include "types.h"
#include <cmath>

inline double norm(const EigenMatrix &input)
{
	return (input.transpose() * input).trace() / input.cols();
}


inline double norm2(const EigenMatrix &input)
{
	return input.array().square().sum();
}


inline double sigmoid(double const &value)
{
	return 1 / ( 1 + exp(-value) );
}


inline double sigmoidDerivative(double const &value)
{
	return sigmoid(value) * ( 1 - sigmoid(value) );
}


inline double sigmoidDerivativeA(double const &value)
{
	return value * (1 - value);
}


inline double tanH(double const &value)
{
	return 1.7159 * tanh(value * 2 / 3);
}


inline double tanHDerivative(double const &value)
{
	return 1.7159 * 2 / 3 * ( 1 - pow(tanh(value * 2 / 3), 2) );
}


inline double chooseConstFunction(double (*function)(double const &), double value)
{
	return (*function)(value);
}


inline double chooseFunction(double (*function)(double), double value)
{
	return (*function)(value);
}


inline EigenMatrix addBias(const EigenMatrix &matrix)
{
	EigenMatrix newMatrix( matrix.rows() + 1, matrix.cols() );
	newMatrix.block( 0, 0, matrix.rows(), matrix.cols() ) = matrix;
	newMatrix.block( matrix.rows(), 0, 1, matrix.cols() ) = -EigenVector::Ones( matrix.cols() ).transpose();
	return newMatrix;
}


inline EigenMatrix activation(const EigenMatrix &weights, const EigenMatrix &inputVector)
{
	return ( weights * addBias(inputVector) ).unaryExpr( ptr_fun(sigmoid) );
}


inline EigenMatrix activation( const EigenMatrix &weights, const EigenMatrix &inputVector, double (*function)(double const &) )
{
	return ( weights * addBias(inputVector) ).unaryExpr( ptr_fun(function) );
}


#endif // MLPMATH

