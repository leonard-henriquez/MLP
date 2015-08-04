#ifndef IO
#define IO

#include <fstream>
#include <iostream>

const integer fs = sizeof(float), is = sizeof(integer);

int reverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return ( (int)ch1 << 24 ) + ( (int)ch2 << 16 ) + ( (int)ch3 << 8 ) + ch4;
}


EigenMatrix readMNISTPics(const string &fileloc, const int &nbExamples)
{
	EigenMatrix dataSet;

    ifstream file(fileloc, ios::binary);
	if ( file.is_open() )
	{
		cout << "loading images..." << endl;

		int magicNumber = 0, numberOfImages = 0, nbRows = 0, nbCols = 0;
		file.read( (char*)&magicNumber, sizeof(magicNumber) );
		magicNumber = reverseInt(magicNumber);
		file.read( (char*)&numberOfImages, sizeof(numberOfImages) );
		numberOfImages = reverseInt(numberOfImages);
		file.read( (char*)&nbRows, sizeof(nbRows) );
		nbRows = reverseInt(nbRows);
		file.read( (char*)&nbCols, sizeof(nbCols) );
		nbCols = reverseInt(nbCols);

        numberOfImages = min(nbExamples, 60000);
        float percent = 1 / (float) numberOfImages * 100;

		dataSet.resize(nbRows * nbCols, numberOfImages);

		int j = 1;
        for (int i = 0; i != numberOfImages; ++i)
		{
            for (int r = 0; r != nbRows; ++r)
			{
                for (int c = 0; c != nbCols; ++c)
				{
					unsigned char temp = 0;
					file.read( (char*)&temp, sizeof(temp) );
					dataSet(nbRows * r + c, i) = temp;
				}
			}
            cout << "\r" << floor(i * percent) << "%";
		}
		cout << "\r" << "100%" << endl;
	}
	file.close();
	return dataSet;
}


EigenMatrix readMNISTLabels(const string &fileloc, const int &nbExamples)
{
	EigenMatrix dataSet;

    ifstream file(fileloc, ios::binary);
	if ( file.is_open() )
	{
		cout << "loading labels..." << endl;

		int magicNumber = 0, numberOfImages = 0;
		file.read( (char*)&magicNumber, sizeof(magicNumber) );
		magicNumber = reverseInt(magicNumber);
		file.read( (char*)&numberOfImages, sizeof(numberOfImages) );
		numberOfImages = reverseInt(numberOfImages);

        numberOfImages = min(nbExamples, 60000);
        float percent = 1 / (float) numberOfImages * 100;

		dataSet = -EigenMatrix::Ones(10, numberOfImages);

		int j = 1;
        for (int i = 0; i != numberOfImages; ++i)
		{
			unsigned char temp = 0;
			file.read( (char*)&temp, sizeof(temp) );
			dataSet(temp, i) = 1;
            cout << "\r" << floor(i * percent) << "%";
		}
		cout << "\r" << "100%" << endl;
	}
	file.close();
	return dataSet;
}


void readMLP(const string &input, MLP &mlp)
{
	ifstream file(input, ios::binary);
	if ( file.is_open() )
	{
		cout << "loading mlp..." << endl;
		integer size;
		file.read( (char*) &size, is );
		vector<integer> structure(size);
		for (integer i = 0; i != size; ++i)
			file.read( (char*) &structure[i], is );

		arrayOfLayers layers(structure);
		integer rows, cols;

		/* --sert juste pour le %--- */
		vector<integer> sum(size);
		sum[0] = 0;
		for (integer i = 1; i != size; ++i)
			sum[i] = sum[i - 1] + (structure[i - 1] + 1) * structure[i];
        float percent = 1 / (float) sum[size - 1] * 100;
		/* ------------------------- */

		for (integer i = 0; i != size - 1; ++i)
		{
			EigenMatrix &mat = layers[i];
			rows = mat.rows();
			cols = mat.cols();
			for (integer r = 0; r != rows; ++r)
			{
				for (integer c = 0; c != cols; ++c)
				{
					float value = 0;
					file.read( (char*) &value, fs );
					mat(r, c) = value;
				}
                cout << "\r" << floor( (sum[i] + r * rows) * percent ) << "%";
			}
		}
		cout << "\r" << "100%" << endl;

        mlp.restoreWeights(layers);
		file.close();
	}
}


void writeMLP(const string &output, const MLP &mlp)
{
	ofstream file(output, ios::binary);

	if ( file.is_open() )
	{
		cout << "saving mlp..." << endl;
		const vector<integer> structure = mlp.getStructure();
		const integer size = structure.size();
		file.write( (char*) &size, is );
		for (integer i = 0; i != size; ++i)
			file.write( (char*) &structure[i], is );

		/* --sert juste pour le %--- */
		vector<integer> sum(size);
		sum[0] = 0;
		for (integer i = 1; i != size; ++i)
			sum[i] = sum[i - 1] + (structure[i - 1] + 1) * structure[i];
        float percent = 1 / (float) sum[size - 1] * 100;
		/* ------------------------- */

		const arrayOfLayers layers = mlp.get();
		integer rows, cols;
		for (integer i = 0; i <= layers.last(); ++i)
		{
			const EigenMatrix &mat = layers[i];
			rows = mat.rows();
			cols = mat.cols();
			int j = 1;
			for (integer r = 0; r != rows; ++r)
			{
				for (integer c = 0; c != cols; ++c)
				{
					file.write( (char*) &mat(r, c), fs );
				}
                cout << "\r" << floor( (sum[i] + r * rows) * percent ) << "%";
			}
		}
		cout << "\r" << "100%" << endl;

		file.close();
	}
}


#endif // IO
