#include <fstream>
#include <iostream>
#include "multilayerperceptron.h"

using namespace std;


bool adr = 0;
realnumber mT = 10, lR = 0.001;
string input, output = "./out.mlp";



int reverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return ( (int)ch1 << 24 ) + ( (int)ch2 << 16 ) + ( (int)ch3 << 8 ) + ch4;
}


EigenMatrix readMNISTPics()
{
	EigenMatrix dataSet;

	ifstream file("/home/leonard/MNIST/trainImages", ios::binary);
	if ( file.is_open() )
	{
		int magicNumber = 0, numberOfImages = 0, nbRows = 0, nbCols = 0;
		file.read( (char*)&magicNumber, sizeof(magicNumber) );
		magicNumber = reverseInt(magicNumber);
		file.read( (char*)&numberOfImages, sizeof(numberOfImages) );
		numberOfImages = reverseInt(numberOfImages);
		file.read( (char*)&nbRows, sizeof(nbRows) );
		nbRows = reverseInt(nbRows);
		file.read( (char*)&nbCols, sizeof(nbCols) );
		nbCols = reverseInt(nbCols);

		numberOfImages = NBEXAMPLES;

		dataSet.resize(nbRows * nbCols, numberOfImages);

		cout << "loading images:" << endl;
		int j = 1;
		for (int i = 0; i < numberOfImages; ++i)
		{
			for (int r = 0; r < nbRows; ++r)
			{
				for (int c = 0; c < nbCols; ++c)
				{
					unsigned char temp = 0;
					file.read( (char*)&temp, sizeof(temp) );
					dataSet(nbRows * r + c, i) = temp;
				}
			}
			if (j * numberOfImages / 100 <= i)
			{
				cout << "*";
				j++;
			}
		}
		cout << "100%" << endl;
	}
	file.close();
	return dataSet;
}


EigenMatrix readMNISTLabels()
{
	EigenMatrix dataSet;

	ifstream file("/home/leonard/MNIST/trainLabels", ios::binary);
	if ( file.is_open() )
	{
		int magicNumber = 0, numberOfImages = 0;
		file.read( (char*)&magicNumber, sizeof(magicNumber) );
		magicNumber = reverseInt(magicNumber);
		file.read( (char*)&numberOfImages, sizeof(numberOfImages) );
		numberOfImages = reverseInt(numberOfImages);

		numberOfImages = NBEXAMPLES;

		dataSet = -EigenMatrix::Ones(10, numberOfImages);

		cout << "loading labels:" << endl;
		int j = 1;
		for (int i = 0; i < numberOfImages; ++i)
		{
			unsigned char temp = 0;
			file.read( (char*)&temp, sizeof(temp) );
			dataSet(temp, i) = 1;
			if (j * numberOfImages / 100 <= i)
			{
				cout << "*";
				j++;
			}
		}
		cout << "100%" << endl;
	}
	file.close();
	return dataSet;
}


void readMLP(const string &input, MLP &mlp)
{
	std::ifstream file(input);
	if ( file.is_open() )
	{
		integer size;
		file >> size;
		vector<integer> structure(size);
		for (integer i = 0; i < size; ++i)
			file >> structure[i];
		MLP::arrayOfLayers layers(structure);
		for (integer k = 0; k <= layers.last(); ++k)
			for (integer i = 0; i < layers[k].rows(); ++i)
				for (integer j = 0; j < layers[k].cols(); ++j)
					file >> layers[k](i, j);
		mlp.set(layers);
		file.close();
	}
}


void writeMLP(const string &output, const MLP &mlp)
{
	std::ofstream file(output);
	if ( !output.empty() && file.is_open() )
	{
		MLP::arrayOfLayers layers = mlp.get();
		file << layers.size() + 1;
		for (integer i = 0; i < layers.size() + 1; ++i)
			file << layers.get().at(i);
		for (integer k = 0; k <= layers.last(); ++k)
			for (integer i = 0; i < layers[k].rows(); ++i)
				for (integer j = 0; j < layers[k].cols(); ++j)
					file << layers[k].coeff(i, j);
		file.close();
	}
}


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


