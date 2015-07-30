#ifndef MLPMATH
#define MLPMATH

#include "types.h"
#include <cmath>

inline realnumber norm(const EigenMatrix &input)
{
	return (input.transpose() * input).trace() / input.cols();
}


inline realnumber norm2(const EigenMatrix &input)
{
	return input.array().square().sum();
}


inline realnumber sigmoid(realnumber const &value)
{
	return 1 / ( 1 + expf(-value) );
}


inline realnumber sigmoidDerivative(realnumber const &value)
{
	return sigmoid(value) * ( 1 - sigmoid(value) );
}


inline realnumber sigmoidDerivativeA(realnumber const &value)
{
	return value * (1 - value);
}


inline realnumber tanH(realnumber const &value)
{
    return 1.7159 * tanhf(value * 0.66666);
}


inline realnumber tanHDerivative(realnumber const &value)
{
    return 1.1439 * ( 1 - powf(tanhf(value * 0.66666), 2) );
}


inline realnumber chooseConstFunction(realnumber (*function)(realnumber const &), realnumber value)
{
	return (*function)(value);
}


inline realnumber chooseFunction(realnumber (*function)(realnumber), realnumber value)
{
	return (*function)(value);
}


inline EigenMatrix addBias(const EigenMatrix &matrix)
{
	integer rows = matrix.rows(), cols = matrix.cols();
	EigenMatrix newMatrix(rows + 1, cols);
	newMatrix.block(0, 0, rows, cols) = matrix;
	newMatrix.block(rows, 0, 1, cols) = -EigenVector::Ones( matrix.cols() ).transpose();
	return newMatrix;
}


inline EigenMatrix activation(const EigenMatrix &weights, const EigenMatrix &inputVector)
{
	return ( weights * addBias(inputVector) ).unaryExpr( ptr_fun(sigmoid) );
}


inline EigenMatrix activation( const EigenMatrix &weights, const EigenMatrix &inputVector, realnumber (*function)(realnumber const &) )
{
	return ( weights * addBias(inputVector) ).unaryExpr( ptr_fun(function) );
}


#endif // MLPMATH

