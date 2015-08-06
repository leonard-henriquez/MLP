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
typedef array<realnumber, 3> tab3;
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
	newMatrix.block(   0, 0, rows, cols) = matrix;
	newMatrix.block(rows, 0, 1,	   cols) = -EigenVector::Ones( matrix.cols() ).transpose();
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


class arrayOfLayers : private vector<EigenMatrix>
{
public:
	arrayOfLayers( const vector<integer> &structure = vector<integer>() ) : vector<EigenMatrix>(), _size(0), _last(-1)
	{
		set(structure);
	}
	void set( const vector<integer> &structure = vector<integer>() )
	{
		if (structure.size() > 1)
		{
			arrayOfLayers::resize(structure.size() - 1);
			for (integer i = 0; i < size(); ++i)
				at(i).resize(structure[i + 1], structure[i] + 1);
		}
		else
		{
			vector<EigenMatrix>::clear();
		}
	}
	vector<integer> get() const
	{
		vector<integer> vect;
		vect.push_back(at(0).cols() - 1);
		for (integer j = 0; j < size(); ++j)
			vect.push_back( at(j).rows() );
		return vect;
	}
	void clear()
	{
		vector<EigenMatrix>::clear();
	}
	void resize(integer size)
	{
		_size = size;
		_last = _size - 1;
		vector<EigenMatrix>::resize(_size);
	}
	integer size() const
	{
		return _size;
	}
	integer last() const
	{
		return _last;
	}
	arrayOfLayers& operator =(const arrayOfLayers &other)
	{
		arrayOfLayers::resize( other.size() );
		for (integer i = 0; i < _size; ++i)
			vector<EigenMatrix>::operator [](i) = other[i];
		return *this;
	}
	EigenMatrix& operator [](integer i)
	{
		return vector<EigenMatrix>::operator [](i);
	}
	EigenMatrix const operator [](integer i) const
	{
		return at(i);
	}
private:
	integer _size;
	integer _last;
};


struct functions
{
	functions(realnumber(*aF)(realnumber const &) = tanH, realnumber(*dAF)(realnumber const &) = tanHDerivative) :
		activation(aF),
		derivativeActivation(dAF)
	{}

	realnumber (*activation)(realnumber const&);
	realnumber (*derivativeActivation)(realnumber const&);
};


struct learningParameters
{
	learningParameters(realnumber ME = 0.001, realnumber MT = 60, realnumber LR = 1, bool ALR = 1, realnumber L0 = 0, realnumber L1 = 0, realnumber L2 = 0) :
		iteration(0),
		startingTime(0),
		refreshTime(1),
		nextDisplayTime(0),
		mqe(0),
		maxError(ME),
		maxTime(MT),
		learningRate(LR),
		adaptativeLearningRate(ALR),
		lambda({
		{L0, L1, L2}
	})
	{}

	int iteration;
	clock_t startingTime;
	clock_t refreshTime;
	clock_t nextDisplayTime;
	realnumber mqe;
	realnumber maxError;
	realnumber maxTime;
	realnumber learningRate;
	bool adaptativeLearningRate;
	tab3 lambda;
};


class learningData
{
public:
	learningData( EigenMatrix I = EigenMatrix(), EigenMatrix O = EigenMatrix() ) :
		input(),
		output(),
		mean(),
		sigma(0)
	{
		setInput(I);
		setOutput(O);
		input.resize( input.rows(), examples() );
		output.resize( output.rows(), examples() );
	}
	learningData& operator =(const learningData &other)
	{
		input = other.input;
		output = other.output;
		mean = other.mean;
		sigma = other.sigma;
		return *this;
	}
	EigenMatrix getOriginalInput() const
	{
		return input * sigma + mean * ( EigenVector::Ones( input.cols() ) ).transpose();
	}
	EigenMatrix getInput() const
	{
		return input;
	}
	EigenMatrix getInput(integer i) const
	{
		return input.col(i);
	}
	void setInput( EigenMatrix I = EigenMatrix() )
	{
		input = I;
		mean.resize( input.rows() );
		for (integer i = 0; i < input.rows(); ++i)
			mean(i) = input.row(i).sum() / input.cols();
		EigenMatrix mean_resized = mean * ( EigenVector::Ones( input.cols() ) ).transpose();
		sigma = sqrt( norm(input - mean_resized) );
		if (sigma != 0)
			input = (input - mean_resized) / sigma;
		else
			input = EigenMatrix( input.rows(), input.cols() );
	}
	EigenMatrix getOutput() const
	{
		return output;
	}
	EigenMatrix getOutput(integer i) const
	{
		return output.col(i);
	}
	void setOutput( EigenMatrix O = EigenMatrix() )
	{
		output = O;
	}
	EigenMatrix getMean(integer i = 1) const
	{
		return mean * ( EigenVector::Ones(i) ).transpose();
	}
	realnumber getSigma() const
	{
		return sigma;
	}
	integer examples() const
	{
		return input.cols();
	}
	integer inputs() const
	{
		return input.rows();
	}
	integer outputs() const
	{
		return output.rows();
	}
private:
	EigenMatrix input;
	EigenMatrix output;
	EigenVector mean;
	realnumber sigma;
};




#endif	// TYPES_H
