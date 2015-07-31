#include <iostream>
#include "multilayerperceptron.h"
#include "io.h"

using namespace std;

bool adr = 0;
realnumber mT = 10, lR = 0.001;
string input, output = "./out.mlp";


void arguments(int argc, char* argv[])
{
	integer opt;
	while ( ( opt = getopt(argc, argv, "at:l:i:o:") ) != -1 )
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
		}
	}
}


int main(int argc, char* argv[])
{
    Eigen::setNbThreads(8);


	arguments(argc, argv);
	cout << "ADR? " << adr << "\nLearning Rate? " << lR << "\nMax Time? " << mT << "\n" << endl;

	MLP mlp;
	MLP::learningData data( readMNISTPics(), readMNISTLabels() );
	mlp.setLearningData(data);
	MLP::learningParameters parameters;
	parameters.adaptativeLearningRate = adr;
	parameters.learningRate = lR;
	parameters.maxTime = mT;

	if ( !input.empty() )
		readMLP(input, mlp);
	else
		mlp.setStructure({784, 2000, 1500, 1000, 500, 10});

	mlp.gradientDescent(parameters);

	writeMLP(output, mlp);

	return 0;
}


