#include "multilayerperceptron.h"

MLP::MLP(int HL, integer PL):
m_perLayer(PL),
m_last(HL+1),
m_layers(NULL),
m_input(),
m_output(),
m_mean(),
m_sigma(0),
m_oldLayers(NULL),
m_Delta(NULL),
m_activationFunction(tanH),
m_derivativeActivationFunction(tanHDerivative)
{ 
    srand (time(NULL));
}

void MLP::clone(const MLP & other)
{
    m_perLayer = other.m_perLayer;
    m_last = other.m_last;
    m_input = other.m_input;
    m_output = other.m_output;
    m_mean = other.m_mean;
    m_sigma = other.m_sigma;
    m_activationFunction = other.m_activationFunction;
    m_derivativeActivationFunction = other.m_derivativeActivationFunction;
    reset(NOTINIT);
    if (other.m_layers != NULL)
    {
        for (int i = 0; i <= m_last; ++i)
            m_layers[i] = other.m_layers[i];
    }
}

void MLP::clear()
{
    if(m_layers != NULL)
    {
        delete[] m_layers;
        delete[] m_oldLayers;
        delete[] m_Delta;
        m_layers = NULL;
        m_oldLayers = NULL;
        m_Delta = NULL;
    }
}

MLP& MLP::operator=(const MLP & other)
{
    clone(other);
    return *this;
}

MLP::MLP (const MLP & other): MLP()
{
    clone(other);
}

MLP::~MLP()
{
    clear();
}

bool MLP::setArchitecture(initialise init, integer I, integer O)
{
    if (m_layers == NULL)
    {
        if (m_last > 1 && m_perLayer > 0)
        {
            if (I == 0 && O == 0)
            {
                I = m_input.rows();
                O = m_output.rows();
            }

            if (I > 0 && O > 0)
            {
                if (m_input.cols() != m_output.cols())
                {
                    display("Error! not the same number of examples");
                    m_input.resize(I, min(m_input.cols(),m_output.cols()));
                    m_output.resize(O, min(m_input.cols(),m_output.cols()));
                }

                m_layers = new EigenMatrix[m_last+1];
                m_Delta = new EigenMatrix[m_last+1];
                m_oldLayers = new EigenMatrix[m_last+1];

                // initialise randomly
                if (init)
                {
                    m_layers[0]      = EigenMatrix::Random(m_perLayer,I+1);
                    for(int j = 1; j < m_last ; ++j)
                        m_layers[j] = EigenMatrix::Random(m_perLayer,m_perLayer+1);
                    m_layers[m_last] = EigenMatrix::Random(O,m_perLayer+1);

                    // rescale
                    for(int j = 0; j <= m_last; ++j)
                        m_layers[j] *= sqrt(6/(I+O));


                    // create a backup
                    for(int j = 0; j < m_last; ++j)
                        m_oldLayers[j] = m_layers[j];
                }
                else
                {
                    m_layers[0].resize(m_perLayer,I+1);
                    for(int j = 1; j < m_last ; ++j)
                        m_layers[j].resize(m_perLayer,m_perLayer+1);
                    m_layers[m_last].resize(O,m_perLayer+1);

                    m_oldLayers[0].resize(m_perLayer,I+1);
                    for(int j = 1; j < m_last ; ++j)
                        m_oldLayers[j].resize(m_perLayer,m_perLayer+1);
                    m_oldLayers[m_last].resize(O,m_perLayer+1);
                }

                m_Delta[0].resize(m_perLayer,1);
                for(int j = 1; j < m_last ; ++j)
                    m_Delta[j].resize(m_perLayer,1);
                m_Delta[m_last].resize(O,1);
            }
            else
                display("Error! There isn't any input or output");
        }
        else
            clear();
    }
    else if (m_input.rows()+1 != m_layers[0].cols() || m_output.rows() != m_layers[m_last].rows())
            reset();
    return (m_layers != NULL);
}

void MLP::reset(initialise init, integer HL, integer PL)
{
    if (HL > 0)
        m_last = HL+1;
    if (PL > 0)
        m_perLayer = PL;

    clear();
    setArchitecture(init);
}

bool MLP::learn(realnumber ME, realnumber MT, realnumber LR, bool ALR, realnumber lambda, realnumber lambda1, realnumber lambda2)
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

    if (setArchitecture(INIT))
    {
        integer index, compteur = 0;
        realnumber  nextDisplayTime = 0,
                newMQE = MQE(lambda, lambda1, lambda2),
                oldMQE = newMQE;
        clock_t start = clock();


//        integer validationPercent = 75, validationNumber = validationPercent/100*m_input.cols(), i;
//        bool test;
//        vector<integer> validationSet(validationNumber);
//        for (integer j = 0; j < validationNumber; ++j)
//        {
//            test = 0;
//            i = rand()%m_input.cols();
//            for (integer k = 0; k < j ; ++k)
//                if (validationSet[k] == i)
//                    test = 1;
//            if (!test)
//            validationSet[j] = i;
//        }



        displayInfo(lambda, lambda1, lambda2);
        display("learning starting...");

        // pour la suite: "index" est le numéro de l'exemple que l'on est en train de traiter
        // et "j" est le numéro de la couche
        while(newMQE > ME && (clock() - start) / (realnumber)CLOCKS_PER_SEC < MT)
        {
            // affiche "mqe" et "m_learningRate" si le dernier affichage date de plus d'une seconde
            displayMQE(start, nextDisplayTime, newMQE, LR);

            // présente un exemple au hasard pour l'apprendre

            index = rand()% m_input.cols(); // ATTENTION! A améliorer
//            index = validationSet[rand()%validationSet.size()];

            saveWeights();
            weightDecay(lambda, lambda1, lambda2);
            modifyWeights(index, LR);

            // on vérifie s'ils sont meilleurs que les anciens, sinon on revient en arrière
            newMQE = MQE(lambda, lambda1, lambda2);
            modifyLearningRate(LR, ALR, oldMQE, newMQE);
            compteur++;
        }


        display("learning finished! \n");
        display("Iterations: " + toStr(int(compteur)) + "; Temps en secondes :  " + toStr ((clock() - start) / (realnumber)CLOCKS_PER_SEC) + "");
        displayInfo(lambda, lambda1, lambda2);
        return (newMQE <= ME);
    }
    else
        return 0;
}

realnumber MLP::weightCost(const realnumber &lambda, const realnumber &lambda1, const realnumber & lambda2)
{
    realnumber sum = 0;
    for (int j = m_last; j >= 0; --j)
    {
        sum += ((j==m_last)? lambda1: lambda) * norm2(m_layers[j].block(0,0, m_layers[j].rows(), m_layers[j].cols()-1));
        sum += lambda2 * norm2(m_layers[j].block(0,m_layers[j].cols()-1, m_layers[j].rows(),1));
    }
    return sum;
}

void MLP::modifyWeights(const integer &exampleIndex, const realnumber &learningRate)
{
    modifyDelta(m_input.col(exampleIndex), m_output.col(exampleIndex), 0);
    for (int j = m_last; j >= 0; --j)
        m_layers[j] += learningRate * m_Delta[j] * addBias(run(j-1, exampleIndex)).transpose();
}


EigenVector MLP::modifyDelta(EigenVector const &input, EigenVector const &output, integer const & layer)
{
    EigenVector yj = input;

    if (layer == m_last)
        m_Delta[layer] =
                activation(m_layers[layer], yj, m_derivativeActivationFunction).asDiagonal()
                * (output - activation(m_layers[layer], yj, m_activationFunction));
    else
        m_Delta[layer] =
                activation(m_layers[layer], yj, m_derivativeActivationFunction).asDiagonal()
                * m_layers[layer+1].block(0,0,m_layers[layer+1].rows(), m_layers[layer+1].cols()-1).transpose()
                * modifyDelta(activation(m_layers[layer], yj, m_activationFunction), output,layer+1);
    return m_Delta[layer];
}

void MLP::modifyLearningRate(realnumber &learningRate, bool adaptativeLearningRate, realnumber &oldMQE, realnumber &newMQE)
{
    if (adaptativeLearningRate)
    {
        if (newMQE > (1+0.03/m_input.cols()) * oldMQE)
        {
            restoreWeights();
            learningRate = min(max(learningRate - 0.3, realnumber(0.01)), learningRate*0.7);
        }
        else
        {
            oldMQE = newMQE;
            learningRate += 0.001;
        }
    }
    else
        oldMQE = newMQE;
}

EigenMatrix MLP::run(const integer &layer, const integer &exampleIndex)
// calcule la sortie associée à la matrice "m_input" jusqu'à couche numéro "layer"
{
    EigenMatrix output;
    if (exampleIndex == -1)
        output = m_input;
    else
        output = m_input.col(exampleIndex);

    for(int j = 0; j <= layer; ++j)
        output = activation(m_layers[j], output, m_activationFunction);
    return output;
}

STLVector MLP::run(const STLVector &input)
{
    if (setArchitecture(INIT))
    {
        EigenMatrix saveInput = m_input;
        m_input = (STLToEigenVector(input) - m_mean) / m_sigma;
        EigenVector output = run(m_last);
        m_input = saveInput;
        return EigenToSTLVector(output);
    }
    else
        return STLVector();
}

realnumber MLP::MQE(const realnumber &lambda, const realnumber &lambda1, const realnumber & lambda2)
// renvoie l'erreur quadratique moyenne
{
    if (lambda != 0)
        return (norm( run(m_last) - m_output ) + weightCost(lambda, lambda1, lambda2))/2;
    else
        // en fait c'est Tr(tE*E)
        return norm(run(m_last)-m_output)/2;
}

void MLP::setLearningExamples(const setOfExamples &set)
{
    m_input = setOfExamplesToEigenInputMatrix(set);
    m_output = setOfExamplesToEigenOutputMatrix(set);
}

void MLP::setInput(const EigenMatrix &input, bool skipNormalisation, bool recalc)
{
    m_input = input;

    if(!skipNormalisation)
    {
        if(recalc)
        {
            EigenVector mean(m_input.rows());
            for(int i = 0; i < m_input.rows(); ++i)
                mean(i) = m_input.row(i).sum()/m_input.cols();
            m_mean = mean * (EigenVector::Ones(m_input.cols())).transpose();
            m_sigma = sqrt(norm(m_input-m_mean));
        }
        m_input = (m_input - m_mean) / m_sigma;
    }
}

void MLP::setOutput(const EigenMatrix &output)
{
    m_output = output;
}

EigenMatrix MLP::getInput()
{
    return m_input;
}

EigenMatrix MLP::getOutput()
{
    return m_output;
}

void MLP::setActivationFunction(int i)
{
    string str;
    switch (i)
    {
    default:
        m_activationFunction = sigmoid;
        m_derivativeActivationFunction = sigmoidDerivative;
        str = "sigmoid";
        break;
    case 1:
        m_activationFunction = tanH;
        m_derivativeActivationFunction = tanHDerivative;
        str = "tanh";
        break;
    }
    display("activation function: " + str);
}

void MLP::weightDecay(const realnumber &lambda, const realnumber &lambda1, const realnumber &lambda2)
{
    integer rows, cols;
    for (int j = 0; j <= m_last ; ++j)
    {
        rows = m_layers[j].rows();
        cols = m_layers[j].cols();
        m_layers[j].block(0, 0, rows, cols-1) =
                ( 1 - ((j==m_last)? lambda1: lambda) )
                * m_layers[j].block(0, 0, rows, cols-1);
        m_layers[j].block(0, cols-1, rows, 1) =
                ( 1 - lambda2 )
                * m_layers[j].block(0, cols-1, rows, 1);
    }
}

void MLP::saveWeights()
{
    for (int j = 0; j <= m_last; ++j)
        m_oldLayers[j] = m_layers[j];
}

void MLP::restoreWeights()
{
    for (int j = 0; j <= m_last; ++j)
        m_layers[j] = m_oldLayers[j];
}

void MLP::displayInfo(const realnumber &lambda, const realnumber &lambda1, const realnumber & lambda2)
// affiche les informations sur le MLP
{
    realnumber maxCoeff = 0, mean = 0;

    for (int j = 0; j <= m_last; ++j)
    {
        maxCoeff = max(max(abs(m_layers[j].maxCoeff()), abs(m_layers[j].minCoeff())), maxCoeff);
        mean += m_layers[j].array().abs().mean();
    }
    string str;

    str +=   "MQE = "                     +   toStr(MQE())               + "\n";
    str +=   "cost of weights = "         +   toStr(weightCost(lambda, lambda1, lambda2)/2)      + "\n";
    str +=   "max weight = "              +   toStr(maxCoeff)            + "\n";
    str +=   "mean of abs weights = "     +   toStr(mean/m_input.cols()) + "\n";
    display(str);
}

void MLP::display(const string & str)
{
    cout << str << endl;
}

bool MLP::displayMQE(clock_t const &start, realnumber &nextDisplayTime, realnumber const &MQE, realnumber const & learningRate, realnumber const &refreshTime)
// affiche le "m_learningRate" et la "mqe" toutes les secondes
{
    if ((clock() - start) / (realnumber)CLOCKS_PER_SEC > nextDisplayTime)
    {
        nextDisplayTime += refreshTime;
        display("learning rate : " + toStr(learningRate) + "; MQE : " + toStr(MQE));
        return 1;
    }
    else
        return 0;
}


/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                                                           *
 *                                                   RECUIT SIMULE                                                           *
 *                                                                                                                           *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

                if (deltaMQE > 0 && rand()/RAND_MAX > exp(-deltaMQE/m_learningRate))
                {
                    restoreWeights();
                    m_learningRate *= alpha;
                }
                else
                    mqe += deltaMQE;

 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */



/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                                                           *
 *                                                  BACK PROP WITH MOMEMTUM                                                  *
 *                                                                                                                           *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

                Delta[m_last] = DeltaLastLayer(m_output, run(m_input));
                previousDeltaWeight[m_last] = DeltaWeight(m_learningRate, Delta[m_last], run(m_input, m_last-1, 1))
                                                + momentum * previousDeltaWeight[m_last];

                m_layers[m_last] += previousDeltaWeight[m_last];
                for (int j = m_last-1; j >= 0; --j)
                {
                    Delta[j] = DeltaHiddenLayer( m_layers[j+1],
                                                 Delta[j+1],
                                                 run(m_input,j)  );

                    previousDeltaWeight[j] = DeltaWeight( m_learningRate,
                                                          Delta[j],
                                                          run(m_input, j-1, 1)  );

                    m_layers[j]+= previousDeltaWeight[j];
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
                    for (int j = m_last; j >= 0; --j)
                        sum += norm2(m_layers[j]);
                    return MAX_ERROR/(2*sum);
                }

 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
