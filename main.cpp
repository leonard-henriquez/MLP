#include <iostream>
#include "multilayerperceptron.h"
#include "io.h"
#include <omp.h>

using namespace std;

bool adr = 0;
realnumber mT = 10, lR = 0.001;
string input, output = "./out.mlp";
integer nbExamples = 2000;

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
        case 'e':
            nbExamples = atoi(optarg);
            break;
		}
	}
}


int main(int argc, char* argv[])
{
	int nthreads = omp_get_num_threads();             // computation of the total number of threads
	cout << endl << nthreads << " thread(s) available for computation" << endl;
    cout << Eigen::nbThreads() << " thread(s) used by Eigen" << endl;

	arguments(argc, argv);
    cout << "\nADR? " << adr << "\nLearning Rate? " << lR << "\nMax Time? " << mT << "\n" << endl;

	MLP mlp;


//	EigenMatrix in(2, 4), out(1, 4);
//    in << -1, -1,  1,  1,
//          -1,  1, -1,  1;
//	out <<  0,  1,  1,  0;
//	MLP::learningData data(in, out);


    MLP::learningData data( readMNISTPics(nbExamples), readMNISTLabels(nbExamples) );


	mlp.setLearningData(data);
	MLP::learningParameters parameters;
	parameters.adaptativeLearningRate = adr;
	parameters.learningRate = lR;
	parameters.maxTime = mT;

	if ( !input.empty() )
		readMLP(input, mlp);
	else
		mlp.setStructure({784, 2000, 1500, 1000, 500, 10});
//	mlp.setStructure({2, 2, 1});


	mlp.gradientDescent(parameters);

	writeMLP(output, mlp);

	return 0;
}


