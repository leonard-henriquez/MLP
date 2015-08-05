#include <iostream>
#include "multilayerperceptron.h"
#include "io.h"


using namespace std;

bool adr = 0;
realnumber mT = 10, lR = 0.001;
string structure, input, output = "./out.mlp", dataImage = "/home/leonard/MNIST/trainImages", dataLabel = "/home/leonard/MNIST/trainLabels";
integer nbExamples = 2000;

void arguments(int argc, char* argv[])
{
	integer opt;
    while ( ( opt = getopt(argc, argv, "at:r:i:o:e:d:l:s:") ) != -1 )
	{
		switch (opt)
		{
		case 'a':
			adr = 1;
			break;
		case 't':
			mT = atof(optarg);
			break;
        case 'r':
			lR = atof(optarg);
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
		}
	}
}


int main(int argc, char* argv[])
{
    cout << "\033[1;1H\x1b[2J";

    arguments(argc, argv);

	MLP mlp;

    learningData data( readMNISTPics(dataImage, nbExamples), readMNISTLabels(dataLabel, nbExamples) );

	mlp.setLearningData(data);
	learningParameters parameters;
	parameters.adaptativeLearningRate = adr;
	parameters.learningRate = lR;
	parameters.maxTime = mT;

    if ( structure.empty() )
    {
        vector<integer> str;
        integer i = structure.find(",");
        while( i != string::npos)
        {
            str.push_back(stoi(structure.substr(0, i)));
            structure = structure.substr(i+1);
            i = structure.find(",");
        }
        mlp.setStructure(str);
        for(integer i = 0; i < str.size(); ++i)
            cout << str[i] << endl;
    }
    else if ( input.empty() )
		mlp.setStructure({784, 2000, 1500, 1000, 500, 10});
    else
        readMLP(input, mlp);

    cout << endl;

    arrayOfLayers backup;
    mlp.saveWeights(backup);
    learningParameters back(parameters);
	mlp.gradientDescent(parameters);
    mlp.restoreWeights(backup);
    cout << "\n\nRestart (iter)\n" << endl;
    mlp.gradientDescent(back);

//	writeMLP(output, mlp);

	return 0;
}


