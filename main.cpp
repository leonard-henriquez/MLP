#include <iostream>
#include "multilayerperceptron.h"
#include "io.h"


using namespace std;

bool adr = 0;
realnumber mT = 10, lR = 0.001;
string input, output = "./out.mlp";
integer nbExamples = 2000;

void arguments(int argc, char* argv[])
{
	integer opt;
	while ( ( opt = getopt(argc, argv, "at:l:i:o:e:") ) != -1 )
	{
		switch (opt)
		{
		case 'a':
			adr = 1;
			break;
		case 't':
			mT = atof(optarg);
			break;
		case 'l':
			lR = atof(optarg);
			break;
		case 'i':
			input = optarg;
			break;
		case 'o':
			output = optarg;
			break;
		case 'e':
			nbExamples = atoi(optarg);
			break;
		}
	}
}


int main(int argc, char* argv[])
{
    cout << "\033[1;1H\x1b[2J";

    arguments(argc, argv);

	MLP mlp;

	learningData data( readMNISTPics(nbExamples), readMNISTLabels(nbExamples) );

	mlp.setLearningData(data);
	learningParameters parameters;
	parameters.adaptativeLearningRate = adr;
	parameters.learningRate = lR;
	parameters.maxTime = mT;

	if ( !input.empty() )
		readMLP(input, mlp);
	else
		mlp.setStructure({784, 2000, 1500, 1000, 500, 10});

    cout << endl;

	mlp.gradientDescent(parameters);

	writeMLP(output, mlp);

	return 0;
}


