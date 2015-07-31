#include "multilayerperceptron.h"

MLP::MLP( void(*dispFunc)(string const &) ) :
	layers(),
	func(),
	io(),
	displayFunction(dispFunc)
{
	srand(0);//time(NULL) );
}


void MLP::clone(const MLP & other)
{
	setStructure(other.getStructure(), NOT_INIT, RESET);
	for (integer i = 0; i < other.layers.size(); ++i)
	{
		layers[i] = other.layers[i];
	}
}


void MLP::clear()
{
	if ( isSet() )
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


void MLP::set(const arrayOfLayers &futurLayers)
{
	layers = futurLayers;
}


MLP::arrayOfLayers MLP::get() const
{
	return layers;
}


void MLP::setStructure(const vector<integer> &str, const initialise &init, const resetOrNot &overrideIfAlreadySet)
{
	if (isSet() && overrideIfAlreadySet == RESET)
	{
		clear();
	}

	if (!isSet() || overrideIfAlreadySet == RESET)
	{
		layers.set(str);
	}

	if (init == INIT)
	{
		// then initialise random
		integer I = str[0], O = str[layers.last() + 1];
		const float factor = sqrtf( (float) 6 / (I + O) );
		for (integer j = 0; j <= layers.last(); ++j)
		{
			layers[j].setRandom(str[j + 1], str[j] + 1);
			layers[j] *= factor;
		}
	}
}


vector<integer> MLP::getStructure() const
{
	return layers.get();
}


void MLP::setLearningData(learningData &data)
{
	io = data;
}


MLP::learningData MLP::getLearningData() const
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

	if ( isSet() )
	{
		integer index;
		parameters.iteration = 0;
		parameters.nextDisplayTime = 0;
		parameters.mqe = MQE(parameters);
		parameters.startingTime = clock();
		arrayOfLayers layers_backup = layers, delta;
		delta.resize( layers.size() );
		for (int j = 0; j <= layers.last(); ++j)
		{
			delta[j].resize(layers[j].rows(), 1);
		}


		displayInfo(parameters);
		display("learning starting...");

		// pour la suite: "index" est le numéro de l'exemple que l'on est en train de traiter
		// et "j" est le numéro de la couche
		while (parameters.mqe > parameters.maxError && (clock() - parameters.startingTime) * CLOCKS_PER_SEC_INV < parameters.maxTime)
		{
			// affiche "mqe" et "learningRate" si le dernier affichage date de plus d'une seconde
			displayMQE(parameters);

			// présente un exemple au hasard pour l'apprendre

			index = rand() % io.examples(); // ATTENTION! A améliorer

			saveWeights(layers_backup);
			weightDecay(parameters);
			modifyWeights(parameters, index, delta);

			// on vérifie s'ils sont meilleurs que les anciens, sinon on revient en arrière
			modifyLearningRate(parameters, layers_backup);
			parameters.iteration++;
		}

		display("learning finished! \n");
        display("Iterations: " + to_string( int(parameters.iteration) ) + "; Temps en secondes :  " + to_string( (clock() - parameters.startingTime) * CLOCKS_PER_SEC_INV ) + "");
		displayInfo(parameters);
	}
}


realnumber MLP::weightCost(const learningParameters &parameters) const
{
	realnumber sum = 0;
	integer j = layers.last();
	sum += parameters.lambda[1] * norm2( layers[j].block(0, 0, layers[j].rows(), layers[j].cols() - 1) );
	sum += parameters.lambda[2] * norm2( layers[j].block(0, layers[j].cols() - 1, layers[j].rows(), 1) );
	for (--j; j >= 0; --j)
	{
		sum += parameters.lambda[0] * norm2( layers[j].block(0, 0, layers[j].rows(), layers[j].cols() - 1) );
		sum += parameters.lambda[2] * norm2( layers[j].block(0, layers[j].cols() - 1, layers[j].rows(), 1) );
	}
	return sum;
}


void MLP::modifyWeights(const learningParameters &parameters, const integer &exampleIndex, arrayOfLayers &delta)
{
	modifyDelta(io.getInput(exampleIndex), io.getOutput(exampleIndex), 0, delta);
	for (integer j = layers.last(); j > 0; --j)
	{
		layers[j] +=
		    parameters.learningRate
		    * delta[j]
		    * addBias( run(exampleIndex, j - 1) ).transpose();
	}
	layers[0] +=
	    parameters.learningRate
	    * delta[0]
	    * addBias( io.getInput(exampleIndex) ).transpose();
}


EigenVector MLP::modifyDelta(EigenVector const &yj, EigenVector const &yo, integer const & layer, arrayOfLayers &delta)
// yo = desiredOutput
{
	if ( layer == layers.last() )
	{
		delta[layer] =
		    activation(layers[layer], yj, func.derivativeActivation).asDiagonal()
		    * ( yo - activation(layers[layer], yj, func.activation) );
	}
	else
	{
		delta[layer] =
		    activation(layers[layer], yj, func.derivativeActivation).asDiagonal()
		    * layers[layer + 1].block(0, 0, layers[layer + 1].rows(), layers[layer + 1].cols() - 1).transpose()
		    * modifyDelta(activation(layers[layer], yj, func.activation), yo, layer + 1, delta);
	}

	return delta[layer];
}


void MLP::modifyLearningRate(learningParameters &parameters, const arrayOfLayers &layers_backup)
{
	realnumber newmqe = MQE(parameters);
	if (parameters.adaptativeLearningRate)
	{
		if (newmqe > ( 1 + 0.03 / io.examples() ) * parameters.mqe)
		{
			restoreWeights(layers_backup);
			parameters.learningRate = min( max( parameters.learningRate - realnumber(0.3), realnumber(0.01) ), parameters.learningRate * realnumber(0.7) );
		}
		else
		{
			parameters.mqe = newmqe;
			parameters.learningRate += 0.001;
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


realnumber MLP::MQE(const learningParameters &parameters) const
// renvoie l'erreur quadratique moyenne
{
	if ( parameters.lambda != tab3() )
	{
		return ( norm( run() - io.getInput() ) + weightCost(parameters) ) / 2;
	}
	else
	{
		// en fait c'est Tr(tE*E)
		return norm( run() - io.getOutput() ) / 2;
	}
	cout << run() << endl;
	cout << io.getOutput() << endl;
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


void MLP::saveWeights(arrayOfLayers &layers_backup) const
{
	for (integer j = 0; j <= layers.last(); ++j)
	{
		layers_backup[j] = layers[j];
	}
}


void MLP::restoreWeights(const arrayOfLayers &layers_backup)
{
	for (integer j = 0; j <= layers.last(); ++j)
	{
		layers[j] = layers_backup[j];
	}
}


void MLP::displayInfo(const learningParameters &parameters) const
// affiche les informations sur le MLP
{
	realnumber maxCoeff = 0, mean = 0;

	for (integer j = 0; j <= layers.last(); ++j)
	{
		maxCoeff = max(max( abs( layers[j].maxCoeff() ), abs( layers[j].minCoeff() ) ), maxCoeff);
		mean += layers[j].array().abs().mean();
	}
	string str;

    str +=   "MQE = "                     +   to_string(parameters.mqe)               + "\n";
    str +=   "cost of weights = "         +   to_string(weightCost(parameters) / 2)      + "\n";
    str +=   "max weight = "              +   to_string(maxCoeff)            + "\n";
    str +=   "mean of abs weights = "     +   to_string( mean / io.examples() ) + "\n";
	display(str);
}


bool MLP::displayMQE(learningParameters &parameters) const
// affiche le "learningRate" et la "mqe" toutes les secondes
{
	if ( (clock() - parameters.startingTime) * CLOCKS_PER_SEC_INV > parameters.nextDisplayTime )
	{
		parameters.nextDisplayTime += parameters.refreshTime;
        display( "learning rate : " + to_string(parameters.learningRate) + "; MQE : " + to_string(parameters.mqe) );
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
