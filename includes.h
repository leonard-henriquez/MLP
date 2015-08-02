#ifndef TYPES_H
#define TYPES_H

#include <iostream>
#include <vector>
#include <time.h>
#include <cmath>
#include <Eigen/Dense>

// Typedef
using namespace Eigen;
using namespace std;
typedef long long int integer;
typedef double realnumber;
typedef vector<realnumber> STLVector;
typedef Matrix<realnumber, Dynamic, 1> EigenVector;
typedef Matrix<realnumber, Dynamic, Dynamic> EigenMatrix;

const realnumber CLOCKS_PER_SEC_INV = 1 / realnumber(CLOCKS_PER_SEC);



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


inline void display(const string & str)
{
	cout << str << endl;
}


#endif // TYPES_H
