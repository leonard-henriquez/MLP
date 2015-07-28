#include <fstream>
#include <iostream>
#include "multilayerperceptron.h"

using namespace std;

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

		numberOfImages = 2000;

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
		numberOfImages = 2000;

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


int main(int argc, char* argv[])
{
	MLP mlp;
	MLP::learningData data( readMNISTPics(), readMNISTLabels() );
	mlp.setLearningData(data);
	MLP::learningParameters paramaters;
	paramaters.adaptativeLearningRate = true;
	paramaters.learningRate = 0.05;
	paramaters.lambda = {0, 0, 0};
	paramaters.maxTime = 100000;
	mlp.gradientDescent(paramaters);
	cout << mlp.MQE(paramaters) << endl;
	return 0;
}


