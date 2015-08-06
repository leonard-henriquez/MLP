#ifndef MULTILAYERPERCEPTRON_H
#define MULTILAYERPERCEPTRON_H

#include "includes.h"

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
	void setActivationFunction (integer i);	// 0 sig, 1 tanh
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

#endif	// MULTILAYERPERCEPTRON_H
