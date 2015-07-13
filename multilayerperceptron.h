#ifndef MULTILAYERPERCEPTRON_H
#define MULTILAYERPERCEPTRON_H

/******************************************* DEBUG *******************************************/
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

/***************************************** EXCEPTION *****************************************/
#include <stdexcept>

/******************************************** MLP ********************************************/
#include <time.h>
#include <algorithm>
#include "multilayerperceptron_global.h"



class MLP
{
public:
    enum initialise {NOTINIT, INIT};
    struct learningData
    {
        int iteration;
        clock_t startingTime;
        clock_t nextDisplayTime;
        realnumber currentMQE;
        realnumber maxError;
        realnumber maxTime;
        realnumber learningRate;
        bool adaptativeLearningRate;
        realnumber lambda;
        realnumber lambda1;
        realnumber lambda2;
        EigenMatrix * m_oldLayers;
        EigenMatrix * m_Delta;
    };


    MLP(int HL = 0, integer PL = 0);
    MLP &operator = (const MLP &);
    MLP (const MLP &other);

    bool setArchitecture(initialise init = INIT, integer I = 0, integer O = 0);
    void reset(initialise init = INIT, integer HL = 0, integer PL = 0);
    virtual ~MLP();

    void setInput(const EigenMatrix &input, bool skipNormalisation = 0, bool recalc = 1);
    void setOutput(const EigenMatrix &output);
    EigenMatrix getInput();
    EigenMatrix getOutput();
    void setLearningExamples(const setOfExamples &set);
    void setActivationFunction(int i); // 0 sig, 1 tanh

    STLVector run(const STLVector &input);
    realnumber MQE(const realnumber &lambda0 = 0, const realnumber &lambda1 = 0, const realnumber & lambda2 = 0);
    bool learn(realnumber ME = MAX_ERROR, realnumber MT = MAX_TIME, realnumber LR = LEARNING_RATE, bool ALR = ADAPTATIVELR, realnumber lambda = LAMBDA, realnumber lambda1 = LAMBDA1, realnumber lambda2 = LAMBDA2);
    virtual void displayInfo(const realnumber &lambda, const realnumber &lambda1, const realnumber & lambda2);

    
protected:

    void clone(const MLP &);
    void clear();
    EigenMatrix run(const integer &layer, const integer &exampleIndex = -1);

    virtual void display(const string & str);
    virtual bool displayMQE(clock_t const &start, realnumber &nextDisplayTime, const realnumber & mqe, const realnumber &learningRate, const realnumber &refreshTime = 1);

    // apprentissage
    void weightDecay(const realnumber &lambda, const realnumber &lambda1, const realnumber &lambda2);
    realnumber weightCost(const realnumber &lambda, const realnumber &lambda1, const realnumber &lambda2);
    EigenVector modifyDelta(const EigenVector &input, const EigenVector &output, const integer &exampleIndex);
    void modifyWeights(const integer &exampleIndex, const realnumber &learningRate);
    void modifyLearningRate(realnumber &learningRate, bool adaptativeLearningRate, realnumber &oldMQE, realnumber &newMQE);
    void saveWeights();
    void restoreWeights();

    // structure du MLP
    integer m_perLayer;
    integer m_last;
    EigenMatrix * m_layers;

    // données entres/sorties
    EigenMatrix m_input,
                m_output;
    EigenMatrix m_mean;
    realnumber m_sigma;

    EigenMatrix * m_oldLayers,
                * m_Delta;

    realnumber (*m_activationFunction)(realnumber const&);
    realnumber (*m_derivativeActivationFunction)(realnumber const&);
};

#endif // MULTILAYERPERCEPTRON_H
