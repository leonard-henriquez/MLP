#include "multilayerperceptron.h"

MLP::MLP(void(*dispFunc)(string const &)) :
	layers(),
	func(),
	io(),
	displayFunction(dispFunc)
{
	srand(0);	// srand(time(NULL));
}


void MLP::clone(const MLP & other)
{
	setStructure(other.getStructure(), NOT_INIT, RESET);
	layers = other.layers;
}


void MLP::clear()
{
	if (isSet())
	{
		layers.clear();
	}
}


MLP& MLP::operator =(const MLP & other)
{
	clone(other);
	return *this;
}


MLP::MLP (const MLP & other) : MLP()
{
	clone(other);
}


MLP::~MLP()
{
	clear();
}


bool MLP::isSet() const
{
	return (layers.size() > 0);
}


void MLP::setStructure(const vector<integer> &str, const initFlag &init, const resetFlag &overrideIfAlreadySet)
{
	if (isSet() && overrideIfAlreadySet == RESET)
	{
		clear();
	}

	if (!isSet())
	{
		layers.set(str, init);
	}

	if (init == INIT)
	{
		// then initialise random
		integer I = str[0], O = str[layers.last() + 1];
		const float factor = sqrtf((float) 6 / (I + O));
		for (integer j = 0; j <= layers.last(); ++j)
		{
			layers[j].setRandom(str[j + 1], str[j] + 1);
			if (I + O > 0)
				layers[j] *= factor;
		}
	}
}


vector<integer> MLP::getStructure() const
{
	return layers.getStructure();
}


void MLP::setLearningData(learningData &data)
{
	io = data;
}


learningData MLP::getLearningData() const
{
	return io;
}


void MLP::gradientDescent(learningParameters &parameters)
// learn permet de réaliser l'apprentissage du MLP
{

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*                                                                                                                           *
*                                                  A IMPLEMENTER                                                            *
*                                                                                                                           *
*      normaliser les données d'entrainement                                                                                *
*      erreur en dessous de laquelle un exemple n'est plus traité                                                           *
*      weight decay                                                                                                         *
*      OK: variation du taux d'apprentissage (algo de Vogl) OU poids distinct pour chaque connexion (Sanossian & Evans)     *
*      élagage                                                                                                              *
*      injection de bruit                                                                                                   *
*      ensemble de validation                                                                                               *
*      early stop                                                                                                           *
*                                                                                                                           *
*                                                                                                                           *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/*
 * ME = MAX_ERROR
 * MT = MAX_TIME
 * LR = LEARNING_RATE
 * ALR = ADAPTATIVELR (adaptative learning rate)
 */

	if (isSet())
    {
		parameters.iteration = 0;
		parameters.nextDisplayTime = 0;
		parameters.mqe = MQE(parameters);
        parameters.startTime = system_clock::now();
		parameters.algorithmSpecific.gradientDescent = universalClass::gD(layers);

		displayInfo(parameters);
		display("learning starting...");

        while (parameters.mqe > parameters.maxError && timeElapsed(parameters.startTime) < parameters.maxTime)
		{
			// affiche "mqe" et "learningRate" si le dernier affichage date de plus d'une seconde
			displayMQE(parameters);

			saveWeights(parameters);
			weightDecay(parameters);

			layers += meanGradient() * parameters.learningRate;

			// on vérifie s'ils sont meilleurs que les anciens, sinon on revient en arrière
			modifyLearningRate(parameters);
			io.newBatch();
			parameters.iteration++;
		}

		display("learning finished! \n");
        display("Iterations: " + to_string(int(parameters.iteration)) + "; Temps en secondes :  " + to_string(timeElapsed(parameters.startTime)));
		displayInfo(parameters);
	}
}


void MLP::modifyWeights(learningParameters &parameters, const integer &exampleIndex)
{
	modifyDelta(parameters, io.getInput(exampleIndex), io.getOutput(exampleIndex), 0);
	for (integer j = layers.last(); j > 0; --j)
	{
		layers[j] +=
			parameters.learningRate
			* parameters.algorithmSpecific.gradientDescent.delta[j]
			* addBias(run(exampleIndex, j - 1)).transpose();
	}

	layers[0] +=
		parameters.learningRate
		* parameters.algorithmSpecific.gradientDescent.delta[0]
		* addBias(io.getInput(exampleIndex)).transpose();
}


layerType MLP::gradient(const integer &exampleIndex)
{
	layerType gradient(layers.getStructure(), NOT_INIT);
	deltaType delta(layers);

	modifyDelta(delta, io.getInput(exampleIndex), io.getOutput(exampleIndex), 0);
	for (integer j = layers.last(); j > 0; --j)
	{
		gradient[j] = delta[j] * addBias(run(exampleIndex, j - 1)).transpose();
	}

	gradient[0] += delta[0] * addBias(io.getInput(exampleIndex)).transpose();
	return gradient;
}


layerType MLP::meanGradient()
{
	layerType meanGradient(layers.getStructure(), SET_ZERO);
	vector<integer> indexes = io.batch();

	for (integer j = 0; j < indexes.size(); ++j)
	{
		meanGradient += gradient(indexes[j]);
	}

	return meanGradient *= (float)1 / io.getBatchSize();
}


EigenVector MLP::modifyDelta(deltaType &delta, const EigenVector &yj, const EigenVector &yo, const integer &layer)
// yo = desiredOutput
{
	if (layer == layers.last())
	{
		delta[layer] =
			activation(layers[layer], yj, func.derivativeActivation).asDiagonal()
			* (yo - activation(layers[layer], yj, func.activation));
	}
	else
	{
		delta[layer] =
			activation(layers[layer], yj, func.derivativeActivation).asDiagonal()
			* layers[layer + 1].block(0, 0, layers[layer + 1].rows(), layers[layer + 1].cols() - 1).transpose()
			* modifyDelta(delta, activation(layers[layer], yj, func.activation), yo, layer + 1);
	}

	return delta[layer];
}


EigenVector MLP::modifyDelta(learningParameters &parameters, const EigenVector &yj, const EigenVector &yo, const integer &layer)
// yo = desiredOutput
{
	if (layer == layers.last())
	{
		parameters.algorithmSpecific.gradientDescent.delta[layer] =
			activation(layers[layer], yj, func.derivativeActivation).asDiagonal()
			* (yo - activation(layers[layer], yj, func.activation));
	}
	else
	{
		parameters.algorithmSpecific.gradientDescent.delta[layer] =
			activation(layers[layer], yj, func.derivativeActivation).asDiagonal()
			* layers[layer + 1].block(0, 0, layers[layer + 1].rows(), layers[layer + 1].cols() - 1).transpose()
			* modifyDelta(parameters, activation(layers[layer], yj, func.activation), yo, layer + 1);
	}

	return parameters.algorithmSpecific.gradientDescent.delta[layer];
}


void MLP::modifyLearningRate(learningParameters &parameters)
{
	realnumber newmqe = MQE(parameters);
	if (parameters.adaptativeLearningRate)
	{
		if (newmqe > (1 + 0.03 / io.examples()) * parameters.mqe)
		{
			restoreWeights(parameters);
			parameters.learningRate *= 0.97;
		}
		else if (newmqe < (1 + 0.02 / io.examples()) * parameters.mqe)
		{
			parameters.mqe = newmqe;
			parameters.learningRate *= 1.01;
		}
		else
		{
			parameters.mqe = newmqe;
		}
	}
	else
	{
		parameters.mqe = newmqe;
	}
}


EigenMatrix MLP::run() const
{
	EigenMatrix output = io.getInput();
	for (integer j = 0; j <= layers.last(); ++j)
	{
		output = activation(layers[j], output, func.activation);
	}

	return output;
}


EigenMatrix MLP::run(const integer &exampleIndex, const integer &layer) const
// calcule la sortie associée à la matrice "input" jusqu'à couche numéro "layer"
{
	EigenMatrix output = io.getInput(exampleIndex);
	for (integer j = 0; j <= layer; ++j)
	{
		output = activation(layers[j], output, func.activation);
	}

	return output;
}


EigenMatrix MLP::run(const mode &lvt) const
{
	EigenMatrix output = io.getInput(lvt);
	for (integer j = 0; j <= layers.last(); ++j)
	{
		output = activation(layers[j], output, func.activation);
	}

	return output;
}


void MLP::weightDecay(const learningParameters &parameters)
{
	integer rows, cols, j;
	for (j = 0; j < layers.last(); ++j)
	{
		rows = layers[j].rows();
		cols = layers[j].cols();
		layers[j].block(0, 0, rows, cols - 1) =
			(1 - parameters.lambda[0])
			* layers[j].block(0, 0, rows, cols - 1);
		layers[j].block(0, cols - 1, rows, 1) =
			(1 - parameters.lambda[2])
			* layers[j].block(0, cols - 1, rows, 1);
	}

	j = layers.last();
	rows = layers[j].rows();
	cols = layers[j].cols();
	layers[j].block(0, 0, rows, cols - 1) =
		(1 - parameters.lambda[1])
		* layers[j].block(0, 0, rows, cols - 1);
	layers[j].block(0, cols - 1, rows, 1) =
		(1 - parameters.lambda[2])
		* layers[j].block(0, cols - 1, rows, 1);
}


realnumber MLP::weightCost(const learningParameters &parameters) const
{
	realnumber sum = 0;
	if (parameters.lambda != decayArray())
	{
		integer j = layers.last();
		sum += parameters.lambda[1] * norm2(layers[j].block(0, 0, layers[j].rows(), layers[j].cols() - 1));
		sum += parameters.lambda[2] * norm2(layers[j].block(0, layers[j].cols() - 1, layers[j].rows(), 1));
		for (--j; j >= 0; --j)
		{
			sum += parameters.lambda[0] * norm2(layers[j].block(0, 0, layers[j].rows(), layers[j].cols() - 1));
			sum += parameters.lambda[2] * norm2(layers[j].block(0, layers[j].cols() - 1, layers[j].rows(), 1));
		}
	}
	return sum;
}


realnumber MLP::MQE(const learningParameters &parameters, const mode &lvt) const
// renvoie l'erreur quadratique moyenne
{
	return (norm(run(lvt) - io.getOutput(lvt)) + weightCost(parameters)) / 2;
}


layerType MLP::getWeights() const
{
    return layers;
}


void MLP::saveWeights(learningParameters &parameters) const
{
	parameters.algorithmSpecific.gradientDescent.backup = layers;
}


void MLP::restoreWeights(const learningParameters &parameters)
{
	layers = parameters.algorithmSpecific.gradientDescent.backup;
}


void MLP::restoreWeights(const layerType &layers_backup)
{
	layers = layers_backup;
}


void MLP::setActivationFunction(integer i)
{
	string str;
	switch (i)
	{
	default:
		func.activation = sigmoid;
		func.derivativeActivation = sigmoidDerivative;
		str = "sigmoid";
		break;
	case 1:
		func.activation = tanH;
		func.derivativeActivation = tanHDerivative;
		str = "tanh";
		break;
	}
	display("activation function: " + str);
}


void MLP::displayInfo(const learningParameters &parameters) const
// affiche les informations sur le MLP
{
	realnumber maxCoeff = 0, mean = 0;

	for (integer j = 0; j <= layers.last(); ++j)
	{
		maxCoeff = max(max(abs(layers[j].maxCoeff()), abs(layers[j].minCoeff())), maxCoeff);
		mean += layers[j].array().abs().mean();
	}

	string str;

    str +=   "MQE = "                     +   to_string(parameters.mqe) + "\n";
    str +=   "examples = "                +   to_string(io.examples()) + "\n";
    str +=   "cost of weights = "         +   to_string(weightCost(parameters) / 2) + "\n";
    str +=   "max weight = "              +   to_string(maxCoeff) + "\n";
	str +=   "mean of abs weights = "     +   to_string(mean / io.examples()) + "\n";
	display(str);
}


bool MLP::displayMQE(learningParameters &parameters) const
// affiche le "learningRate" et la "mqe" toutes les secondes
{
    if (timeElapsed(parameters.startTime)> parameters.nextDisplayTime)
	{
		parameters.nextDisplayTime += parameters.refreshTime;
		display("learning rate : " + to_string(parameters.learningRate) + "; MQE : " + to_string(MQE(parameters, LEARNING)) + "; MQEV : " + to_string(parameters.mqe));
		return 1;
	}
	else
	{
		return 0;
	}
}


/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*                                                                                                                           *
*                                                   RECUIT SIMULE                                                           *
*                                                                                                                           *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

                if (deltaMQE > 0 && rand()/RAND_MAX > exp(-deltaMQE/learningRate))
                {
                    restoreWeights();
                    learningRate *= alpha;
                }
                else
                    mqe += deltaMQE;

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */



/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*                                                                                                                           *
*                                                  BACK PROP WITH MOMEMTUM                                                  *
*                                                                                                                           *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

                Delta[layers.last()] = DeltaLastLayer(io.output, run(io.input));
                previousDeltaWeight[layers.last()] = DeltaWeight(learningRate, Delta[layers.last()], run(io.input, layers.last()-1, 1))
+ momentum * previousDeltaWeight[layers.last()];

                layers[layers.last()] += previousDeltaWeight[layers.last()];
                for (integer j = layers.last()-1; j >= 0; --j)
                {
                    Delta[j] = DeltaHiddenLayer( layers[j+1],
                                                 Delta[j+1],
                                                 run(io.input,j)  );

                    previousDeltaWeight[j] = DeltaWeight( learningRate,
                                                          Delta[j],
                                                          run(io.input, j-1, 1)  );

                    layers[j]+= previousDeltaWeight[j];
                }

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */



/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*                                                                                                                           *
*                                                  BACK PROP WITH MOMEMTUM                                                  *
*                                                                                                                           *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

                realnumber MLP::findLambda()
                {
                    realnumber sum = 0;
                    for (integer j = layers.last(); j >= 0; --j)
                        sum += norm2(layers[j]);
                    return MAX_ERROR/(2*sum);
                }

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
