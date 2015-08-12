#ifndef GLOBAL
#define GLOBAL

#include <iostream>
#include <iomanip>
#include <getopt.h>
#include <signal.h>
#include "multilayerperceptron.h"
#include "io.h"

// type definitions
enum chooseMode {LEARNING_MODE, TEST_MODE, HELP_MODE, FAIL};
typedef void sigfunc (int);
sigfunc *signal (int, sigfunc*);


// global variables
chooseMode appMode = LEARNING_MODE;
bool adaptativeLearningRate = false;
string MLPStructure, inputMLPFile, outputMLPFile, inputDataFile = "./MNIST_Images.dat", outputDataFile = "./MNIST_Labels.dat";
int numberOfExamples = 2000, batchSize = -1, batchPercent = -1;
learningParameters parameters;
EigenMatrix images, labels;
MLP mlp;

#endif	// GLOBAL

