#ifndef MULTILAYERPERCEPTRON_H
#define MULTILAYERPERCEPTRON_H

#include "mlp_includes.h"

class MLP
{
public:


	MLP(void(*dispFunc)(const string &) = display);
	MLP & operator =(const MLP &);


	MLP (const MLP &other);
	virtual ~MLP();
	bool isSet () const;

	void restoreWeights (const layerType &layers_backup);
	layerType getWeights () const;

	void setStructure (const vector<integer> &str, const initFlag &init = INIT, const resetFlag &overrideIfAlreadySet = RESET);


	vector<integer>getStructure () const;
	void setActivationFunction (integer i);			// 0 sig, 1 tanh
	integer getActivationFunction () const;
	void setLearningData (learningData &data);
	learningData getLearningData () const;

	void setDisplayFunction (void (*displayFunction)(string const&));

	EigenMatrix run () const;
	EigenMatrix run (const integer &exampleIndex, const integer &layer) const;
	EigenMatrix run (const mode &lvt) const;

	realnumber MQE (const learningParameters &parameters, const mode &lvt = VALIDATION) const;
	layerType gradient (const integer &exampleIndex);
	layerType meanGradient ();
	void gradientDescent (learningParameters &parameters);

	virtual void displayInfo (const learningParameters &parameters) const;


protected:


	void clone (const MLP &);
	void clear ();
	void saveWeights (learningParameters &parameters) const;
	void restoreWeights (const learningParameters &parameters);
	virtual bool displayMQE (learningParameters &parameters) const;


	// apprentissage
	void weightDecay (const learningParameters &parameters);
	realnumber weightCost (const learningParameters &parameters) const;
    EigenVector modifyDelta (deltaType &delta, const EigenVector &yj, const EigenVector &yo, const integer &layer);
    void update (const layerType &gradient, learningParameters &parameters);


	// structure du MLP
	layerType layers;
	functions func;

	// données entres/sorties
	learningData io;

	void (*displayFunction)(string const&);
};



#endif	// MULTILAYERPERCEPTRON_H
