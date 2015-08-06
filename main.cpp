#include <iostream>
#include "multilayerperceptron.h"
#include "io.h"
#include <getopt.h>


using namespace std;

bool alr = 0;
realnumber mT = 10, lR = 0.001;
string structure, input, output, dataImage = "/home/leonard/MNIST/trainImages", dataLabel = "/home/leonard/MNIST/trainLabels";
integer nbExamples = 2000, percentOfValidationExamples = 10, percentOfTestExamples = 0;

void arguments(int argc, char* argv[])
{
	static struct option long_options[] =
	{
		{"alr", no_argument, NULL, 'a'},
		{"lr", required_argument, NULL, 'r'},
		{"time", required_argument, NULL, 't'},
		{"input", required_argument, NULL, 'i'},
		{"output", required_argument, NULL, 'o'},
		{"image", required_argument, NULL, 'd'},
		{"label", required_argument, NULL, 'l'},
		{"example", required_argument, NULL, 'e'},
		{"struct", required_argument, NULL, 's'},
		{"validation", required_argument, NULL, 'v'},
		{"test", required_argument, NULL, 'y'},
		{NULL, 0, NULL, 0}
	};

	integer c;
	while ( ( c = getopt_long(argc, argv, "at:r:i:o:e:d:l:s:", long_options, NULL) ) != -1 )
	{
		switch (c)
		{
		case 'a':
			alr = 1;
			break;
		case 'r':
			lR = atof(optarg);
			break;
		case 't':
			mT = atof(optarg);
			break;
		case 'i':
			input = optarg;
			break;
		case 'o':
			output = optarg;
			break;
		case 'd':
			dataImage = optarg;
			break;
		case 'l':
			dataLabel = optarg;
			break;
		case 'e':
			nbExamples = atoi(optarg);
			break;
		case 's':
			structure = optarg;
			break;
		case 'v':
			percentOfValidationExamples = atoi(optarg);
			break;
		case 'y':
			percentOfTestExamples = atoi(optarg);
			break;
		}
	}
}


int main(int argc, char* argv[])
{
	cout << "\033[1;1H\x1b[2J";

	arguments(argc, argv);

	MLP mlp;

	learningData data( readMNISTPics(dataImage, nbExamples), readMNISTLabels(dataLabel, nbExamples) );
	data.setProportion(percentOfValidationExamples, percentOfTestExamples);

	mlp.setLearningData(data);
	learningParameters parameters;
	parameters.adaptativeLearningRate = alr;
	parameters.learningRate = lR;
	parameters.maxTime = mT;
	parameters.refreshTime = mT / 100;

	if ( !structure.empty() )
	{
		vector<integer> str;
		str.push_back( data.inputs() );
		integer i = structure.find(",");
		while (i != string::npos)
		{
			str.push_back( stoi( structure.substr(0, i) ) );
			structure = structure.substr(i + 1);
			i = structure.find(",");
		}
		str.push_back( stoi(structure) );
		str.push_back( data.outputs() );

		mlp.setStructure(str);
	}
	else if ( !input.empty() )
		readMLP(input, mlp);
	else
		mlp.setStructure({784, 2000, 1500, 1000, 500, 10});

	mlp.gradientDescent(parameters);

	if ( !output.empty() )
		writeMLP(output, mlp);

	return 0;
}


