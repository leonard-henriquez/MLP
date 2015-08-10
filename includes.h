#ifndef TYPES_H
#define TYPES_H

#include <iostream>
#include <vector>
#include <time.h>
#include <cmath>
#include <Eigen/Dense>
#include <algorithm>

// Typedef
using namespace Eigen;
using namespace std;
typedef long long int integer;
typedef double realnumber;
typedef array<realnumber, 3> decayArray;
typedef vector<realnumber> STLVector;
typedef Matrix<realnumber, Dynamic, 1> EigenVector;
typedef Matrix<realnumber, Dynamic, Dynamic> EigenMatrix;

const realnumber CLOCKS_PER_SEC_INV = 1 / realnumber(CLOCKS_PER_SEC);

enum mode {LEARNING, VALIDATION, TEST};
enum initFlag {NOT_INIT, INIT, SET_ZERO};
enum resetFlag {NOT_RESET, RESET};



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
	return 1 / (1 + expf(-value));
}


inline realnumber sigmoidDerivative(realnumber const &value)
{
	return sigmoid(value) * (1 - sigmoid(value));
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
	return 1.1439 * (1 - powf(tanhf(value * 0.66666), 2));
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
	newMatrix.block(rows, 0, 1,	   cols) = -EigenVector::Ones(matrix.cols()).transpose();
	return newMatrix;
}


inline EigenMatrix activation(const EigenMatrix &weights, const EigenMatrix &inputVector)
{
	return (weights * addBias(inputVector)).unaryExpr(ptr_fun(sigmoid));
}


inline EigenMatrix activation(const EigenMatrix &weights, const EigenMatrix &inputVector, realnumber (*function)(realnumber const &))
{
	return (weights * addBias(inputVector)).unaryExpr(ptr_fun(function));
}


/******************************************************* display function ***********************************************************/


inline void display(const string & str)
{
	cout << str << endl;
}


/****************************************************** vector of matrices **********************************************************/


class layerType : private vector<EigenMatrix>
{

public:


	layerType(const integer &size = 0) : vector<EigenMatrix>(size), _size(size), _last(size - 1) {}

	layerType(const vector<integer> &structure, initFlag init = SET_ZERO) : vector<EigenMatrix>(), _size(0), _last(-1)
	{
		if (init == SET_ZERO)
			set(structure);
		else
			resize(structure);
	}

	layerType(const layerType &other)
	{
		clone(other);
	}

	void clone(const layerType &other)
	{
		layerType::resize(other.size());
		for (integer i = 0; i < _size; ++i)vector<EigenMatrix>::operator [](i) = other[i];
	}

	void set(const vector<integer> &structure, const initFlag &init = SET_ZERO)
	{
		if (structure.size() > 1)
		{
			layerType::resize(structure.size() - 1);
			if (init == SET_ZERO)
			{
				for (integer i = 0; i < size(); ++i)
				{
					vector<EigenMatrix>::operator [](i) = EigenMatrix::Zero(structure[i + 1], structure[i] + 1);


				}
			}
			else
			{
				for (integer i = 0; i < size(); ++i)
				{
					vector<EigenMatrix>::operator [](i).resize(structure[i + 1], structure[i] + 1);


				}
			}
		}
		else
		{
			vector<EigenMatrix>::clear();
		}
	}

	vector<integer>getStructure() const
	{
		vector<integer> vect;
		if (_size > 0 && at(0).cols() > 0)
		{
			vect.push_back(at(0).cols() - 1);
			for (integer j = 0; j < _size; ++j)
			{
				vect.push_back(at(j).rows());
			}
		}
		return vect;
	}

	void clear()
	{
		vector<EigenMatrix>::clear();
	}

	void resize(const vector<integer> &structure = vector<integer>())
	{
		if (structure.size() > 1)
		{
			layerType::resize(structure.size() - 1);
			for (integer i = 0; i < size(); ++i)
			{
				vector<EigenMatrix>::operator [](i).resize(structure[i + 1], structure[i] + 1);


			}
		}
		else
		{
			vector<EigenMatrix>::clear();
		}
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

	layerType& operator =(const layerType &other)
	{
		clone(other);
		return *this;
	}

	layerType& operator +=(const layerType &other)
	{
		for (integer i = 0; i < _size; ++i)
		{
			vector<EigenMatrix>::operator [](i) += other[i];


		}

		return *this;
	}

	layerType& operator *=(const float &f)
	{
		for (integer i = 0; i < _size; ++i)
		{
			vector<EigenMatrix>::operator [](i) *= f;


		}

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



inline layerType operator +(layerType one, const layerType &other)
{
	return one += other;
}


inline layerType operator *(layerType layer, const float &f)
{
	return layer *= f;
}


class deltaType : private vector<EigenVector>
{

public:


	deltaType() : vector<EigenVector>() {}

	deltaType(const deltaType &other)
	{
		clone(other);
	}

	deltaType(const vector<integer> &structure) : vector<EigenVector>()
	{
		set(structure);
	}

	deltaType(const layerType &layers) : vector<EigenVector>()
	{
		set(layers.getStructure());
	}

	void clone(const deltaType &other)
	{
		vector<EigenVector>::resize(other.vector<EigenVector>::size());
		for (integer i = 0; i < vector<EigenVector>::size(); ++i)
		{
			vector<EigenVector>::operator [](i) = other[i];


		}
	}

	void set(const vector<integer> &structure)
	{
		vector<EigenVector>::resize(max((unsigned long) 0, structure.size() - 1));
		for (integer i = 0; i < structure.size() - 1; ++i)
		{
			vector<EigenVector>::operator [](i) = EigenVector::Zero(structure[i + 1]);


		}
	}

	deltaType& operator =(const deltaType &other)
	{
		clone(other);
		return *this;
	}

	EigenVector& operator [](integer i)
	{
		return vector<EigenVector>::operator [](i);
	}

	EigenVector const operator [](integer i) const
	{
		return at(i);
	}

};



/****************************************************** learning functions **********************************************************/


struct functions
{
	functions(realnumber(*aF)(realnumber const &) = tanH, realnumber(*dAF)(realnumber const &) = tanHDerivative) :
		activation(aF),
		derivativeActivation(dAF)
	{}

	realnumber (*activation)(realnumber const&);
	realnumber (*derivativeActivation)(realnumber const&);
};


/****************************************************** learning parameters *********************************************************/


union universalClass
{
public:


	struct gD
	{
		gD() : backup(), delta(), gradient() {}

		gD(layerType layers) : backup(layers), delta(deltaType(layers))
		{
			gradient.set(layers.getStructure());
		}

		layerType backup;
		deltaType delta;
		layerType gradient;
	};

	universalClass() : gradientDescent() {}
	~universalClass() {}

	gD gradientDescent;
private:


	universalClass(const universalClass &other);
	universalClass& operator =(const universalClass &other);


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
		algorithmSpecific()
	{
		lambda = {L0, L1, L2};
	}

	int iteration;
	clock_t startingTime;
	clock_t refreshTime;
	clock_t nextDisplayTime;
	realnumber mqe;
	realnumber maxError;
	realnumber maxTime;
	realnumber learningRate;
	bool adaptativeLearningRate;
	decayArray lambda;
	universalClass algorithmSpecific;
};


/********************************************************* learning data ************************************************************/


class learningData
{

public:


	learningData() :
		sigma(0), nbExamples(0), batchSize(-1)
	{}

	learningData(const EigenMatrix &I, const EigenMatrix &O) :
		batchSize(-1)
	{
		setIO(I, O);
	}

	learningData(const learningData &other)
	{
		clone(other);
	}

	learningData& operator =(const learningData &other)
	{
		clone(other);
		return *this;
	}

	void clone(const learningData &other)
	{
		input = other.input;
		output = other.output;
		mean = other.mean;
		sigma = other.sigma;
		nbExamples = other.nbExamples;
		batchSize = other.batchSize;
		batchSet = other.batchSet;
		learningSet = other.learningSet;
		validationSet = other.validationSet;
		testSet = other.testSet;
	}

	void setIO(const EigenMatrix &I, const EigenMatrix &O)
	{
		input = I;
		output = O;
		nbExamples = min(input.cols(), output.cols());
		input.resize(input.rows(), nbExamples);
		output.resize(output.rows(), nbExamples);
		setBatchSize(-1);

		mean.resize(input.rows());
		for (integer i = 0; i < input.rows(); ++i)
			mean(i) = input.row(i).sum() / input.cols();

		EigenMatrix mean_resized = mean * (EigenVector::Ones(input.cols())).transpose();
		sigma = sqrt(norm(input - mean_resized));
		if (sigma != 0)
			input = (input - mean_resized) / sigma;
		else
			input = EigenMatrix(input.rows(), input.cols());
	}

	EigenMatrix getOriginalInput() const
	{
		return input * sigma + mean * (EigenVector::Ones(input.cols())).transpose();
	}

	EigenMatrix getInput() const
	{
		return input;
	}

	EigenMatrix getInput(const integer &i) const
	{
		return input.col(i);
	}

	EigenMatrix getInput(const mode &lvt) const
	{
		const vector<integer>& indexes = batch(lvt);

		EigenMatrix permutation = EigenMatrix::Zero(nbExamples, indexes.size());
		for (integer i = 0; i < indexes.size(); ++i)
		{
			permutation(indexes[i], i) = 1;
		}

		return input * permutation;
	}

	EigenMatrix getOutput() const
	{
		return output;
	}

	EigenMatrix getOutput(const integer &i) const
	{
		return output.col(i);
	}

	EigenMatrix getOutput(const mode &lvt) const
	{
		const vector<integer>& indexes = batch(lvt);

		EigenMatrix permutation = EigenMatrix::Zero(nbExamples, indexes.size());
		for (integer i = 0; i < indexes.size(); ++i)
		{
			permutation(indexes[i], i) = 1;
		}

		return output * permutation;
	}

	EigenMatrix getMean(const integer &i = 1) const
	{
		return mean * (EigenVector::Ones(i)).transpose();
	}

	realnumber getSigma() const
	{
		return sigma;
	}

	void setBatchSize(const integer &size)
	{
		// at least 5 learning batch because it's useless to have a too big validation set
		batchSize = min(nbExamples / 7, size);
		integer vtSize = (batchSize >= 0) ? batchSize : nbExamples / 10;

		learningSet.clear();
		validationSet.clear();
		testSet.clear();
		for (integer i = 0; i < nbExamples; ++i)
			learningSet.push_back(i);

		random_shuffle(learningSet.begin(), learningSet.end());


		while (validationSet.size() != vtSize)
		{
			validationSet.push_back(learningSet.back());
			learningSet.pop_back();
		}

		while (testSet.size() != vtSize)
		{
			testSet.push_back(learningSet.back());
			learningSet.pop_back();
		}

		newBatch();
	}

	integer getBatchSize() const
	{
		return batchSize;
	}

	vector<integer> newBatch()
	{
		if (batchSize >= 0)
		{
			if (toBatch.empty())
			{
				toBatch = learningSet;
			}

			batchSet.clear();
			while (batchSet.size() != batchSize)
			{
				batchSet.push_back(toBatch.back());
				toBatch.pop_back();
			}

			return batchSet;
		}
		else
		{
			batchSet = learningSet;
			return batchSet;
		}
	}

	vector<integer> batch() const
	{
		return batchSet;
	}

	vector<integer> batch(const mode &lvt) const
	{
		switch (lvt)
		{
		case LEARNING:
			return batchSet;
			break;
		case VALIDATION:
			return validationSet;
			break;
		case TEST:
			return testSet;
			break;
		}
		return vector<integer>();
	}

	integer examples() const
	{
		return nbExamples;
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
	integer nbExamples;
	integer batchSize;
	vector<integer> batchSet;
	vector<integer> toBatch;
	vector<integer> validationSet;
	vector<integer> learningSet;
	vector<integer> testSet;
};



/************************************************************* else ****************************************************************/



#endif	// TYPES_H
