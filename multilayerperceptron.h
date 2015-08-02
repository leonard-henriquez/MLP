#ifndef MULTILAYERPERCEPTRON_H
#define MULTILAYERPERCEPTRON_H

#include "includes.h"

typedef array<realnumber, 3> tab3;

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
		return min( input.cols(), output.cols() );
	}
private:
	EigenMatrix input;
	EigenMatrix output;
	EigenVector mean;
	realnumber sigma;
};




class MLP
{
public:
	enum initialise {NOT_INIT, INIT};
	enum resetOrNot {NOT_RESET, RESET};
	enum normaliseInput {NOT_NORMALISE, NORMALISE, NORMALISE_WITHOUT_RECALC};

	MLP(void(*dispFunc)(const string &) = display);
	MLP &operator =(const MLP &);

	MLP (const MLP &other);
	virtual ~MLP();
	bool isSet () const;
	void set (const arrayOfLayers &futurLayers);
	arrayOfLayers get () const;
	void setStructure (const vector<integer> &str, const initialise &init = INIT, const resetOrNot &overrideIfAlreadySet = RESET);
	vector<integer> getStructure () const;
	void setActivationFunction (integer i); // 0 sig, 1 tanh
	integer getActivationFunction () const;
	void setLearningData (learningData &data);
	learningData getLearningData () const;
	void setDisplayFunction ( void (*displayFunction)(string const&) );

	EigenMatrix run () const;
	EigenMatrix run (const integer &exampleIndex, const integer &layer) const;
	realnumber MQE (const learningParameters &parameters) const;

	void gradientDescent (learningParameters &parameters);

	virtual void displayInfo (const learningParameters &parameters) const;

protected:
	void clone (const MLP &);
	void clear ();

	virtual bool displayMQE (learningParameters &parameters) const;

	// apprentissage
	void weightDecay (const learningParameters &parameters);
	realnumber weightCost (const learningParameters &parameters) const;
	EigenVector modifyDelta (const EigenVector &yj, const EigenVector &output, const integer &exampleIndex, arrayOfLayers &delta);
	void modifyWeights (const learningParameters &data, const integer &exampleIndex, arrayOfLayers &delta);
	void modifyLearningRate (learningParameters &data, const arrayOfLayers &layers_backup);
	void saveWeights (arrayOfLayers &layers_backup) const;
	void restoreWeights (const arrayOfLayers &layers_backup);

	// structure du MLP
	arrayOfLayers layers;
	functions func;

	// données entres/sorties
	learningData io;

	void (*displayFunction)(string const&);
};

#endif // MULTILAYERPERCEPTRON_H
