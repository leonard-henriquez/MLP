#include <iostream>
#include "multilayerperceptron.h"
#include "io.h"
#include <getopt.h>

using namespace std;


bool adaptativeLearningRate = false, helpMode = false;
mode mlpMode = LEARNING;
string MLPStructure, inputFile, outputFile, imageFile = "./trainImages", labelFile = "./trainLabels";
int nbExamples = 2000, batchSize = -1, batchPercent = -1;
float maxTime = 30, learningRate = 0.0005;


vector<integer> stringToVector (string str);

void getArguments (int argc, char* argv[]);

bool help ();


int main(int argc, char* argv[])
{
	cout << "\033[1;1H\x1b[2J";

	// get arguments from console to create mlp
	getArguments(argc, argv);

	if (helpMode)
		return help();


	MLP mlp;

	// create learning ata from arguments
	learningData data(readMNISTPics(imageFile, nbExamples), readMNISTLabels(labelFile, nbExamples));
	if (batchPercent != -1)
		batchSize = floor((float) batchPercent * nbExamples / 100);
	if (batchSize != -1)
		data.setBatchSize(batchSize);
	mlp.setLearningData(data);

	// create learning parameters
	learningParameters parameters;
	parameters.adaptativeLearningRate = adaptativeLearningRate;
	parameters.learningRate = learningRate;
	parameters.maxTime = maxTime;
	parameters.refreshTime = parameters.maxTime / 100;

	if (!MLPStructure.empty())
		mlp.setStructure(stringToVector(to_string(data.inputs()) + "," + MLPStructure + "," + to_string(data.outputs())));
	else if (!inputFile.empty())
		readMLP(inputFile, mlp);
	else
	{
		cout << "No structure specified; Can't setup MLP; use --help" << endl;
		return 0;
	}

	if (mlpMode == LEARNING)
	{
		mlp.gradientDescent(parameters);
		if (!outputFile.empty())
			writeMLP(outputFile, mlp);
	}
	else
	{
		// test mode
	}

	return 0;
}


void getArguments(int argc, char* argv[])
{
	static struct option long_options[] =
	{
		{"input", required_argument, NULL, 'i'},
		{"output", required_argument, NULL, 'o'},
		{"image", required_argument, NULL, 'd'},
		{"label", required_argument, NULL, 'l'},
		{"example", required_argument, NULL, 'e'},
		{"struct", required_argument, NULL, 's'},
		{"alr", no_argument, NULL, 'a'},
		{"lr", required_argument, NULL, 'r'},
		{"time", required_argument, NULL, 't'},
		{"batch", required_argument, NULL, 'b'},
		{"batchp", required_argument, NULL, 'p'},
		{"test", no_argument, NULL, 'm'},
		{"help", no_argument, NULL, 'h'},
		{NULL, 0, NULL, 0}
	};

	char c;
	while ((c = getopt_long(argc, argv, "at:r:i:o:e:d:l:s:h:", long_options, NULL)) != -1)
	{
		switch (c)
		{
		case 'i':
			inputFile = optarg;
			break;
		case 'o':
			outputFile = optarg;
			break;
		case 'd':
			imageFile = optarg;
			break;
		case 'l':
			labelFile = optarg;
			break;
		case 'e':
			nbExamples = atoi(optarg);
			break;
		case 's':
			MLPStructure = optarg;
			break;
		case 'a':
			adaptativeLearningRate = true;
			break;
		case 'r':
			learningRate = atof(optarg);
			break;
		case 't':
			maxTime = atof(optarg);
			break;
		case 'b':
			batchSize = atoi(optarg);
			break;
		case 'c':
			batchPercent = atoi(optarg);
			break;
		case 'm':
			mlpMode = TEST;
			break;
		case 'h':
			helpMode = true;
			break;
		}
	}
}


vector<integer> stringToVector(string str)
{
	vector<integer> vect;
	integer i = str.find(",");
	while (i != string::npos)
	{
		vect.push_back(stoi(str.substr(0, i)));
		str = str.substr(i + 1);
		i = str.find(",");
	}
	vect.push_back(stoi(str));
	return vect;
}


bool help()
{
	cout << "--input or -i ./dir/mlpfile ; this option loads the MLP in mlpfile" << endl;
	cout << "--output or -o./dir/mlpfile ; the mlp will be written in mlpfile after learning" << endl;
	cout << "--image or -d ./dir/trainImages ; if the file 'trainImages' is not in the same folder as this application" << endl;
	cout << "--label or -l ./dir/trainLabels ; if the file 'trainLabels' is not in the same folder as this application" << endl;
	cout << "--example or -e 2000 ; specifies how many images will be loaded from the MNIST database" << endl;
	cout << "--struct or -s '300,150,50' ; this will build a MLP with three hidden layers of 300, 150 and 50 neurons" << endl;
	cout << "--alr or -a (no argument) ; if enabled, learning rate will vary" << endl;
	cout << "--lr or -r 0.0005 ; to set up learning rate at 0.0005" << endl;
	cout << "--time or -t 30 ; to stop learning after 30 seconds" << endl;
	cout << "--batch or -b 200 (integer); to choose the size of batches" << endl;
	cout << "--test or -m ; to use the test mode" << endl;
	return 0;
}


