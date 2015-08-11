#ifndef GLOBAL
#define GLOBAL

#include <iostream>
#include <getopt.h>
#include "multilayerperceptron.h"
#include "io.h"

enum chooseMode {LEARNING_MODE, TEST_MODE, HELP_MODE, FAIL};

// global variables
chooseMode appMode = LEARNING_MODE;
bool adaptativeLearningRate = false;
string MLPStructure, inputFile, outputFile, imageFile = "./trainImages", labelFile = "./trainLabels";
int nbExamples = 2000, batchSize = -1, batchPercent = -1;
learningParameters parameters;

#endif	// GLOBAL

