#ifndef IO
#define IO

#include <fstream>
#include <iostream>

const integer fs = sizeof(float), iss = sizeof(int), is = sizeof(integer);


EigenMatrix readData(const string &fileloc, const int &numberOfExamplesMax)
{
	EigenMatrix mat;

	ifstream file(fileloc, ios::binary);
	if (file.is_open())
	{
		cout << "loading data..." << endl;

		int n = 0, p = 0;
        file.read((char*)&p, iss);
        file.read((char*)&n, iss);

		p = min(numberOfExamplesMax, p);
		float percent = 1 / (float) p * 100;
		mat.resize(n, p);

		for (int j = 0; j != p; ++j)
		{
			for (int i = 0; i != n; ++i)
			{
                float temp = 0;
                file.read((char*)&temp, fs);
				mat(i, j) = temp;
			}

			cout << "\r" << floor(j * percent) << "%";
		}

		cout << "\r" << "100%" << endl;
		file.close();


	}
	else
	{
		cout << "ERROR: cannot open data file" << endl;
	}
	return mat;
}


void writeData(const string &fileloc, const EigenMatrix &mat)
{

	ofstream file(fileloc, ios::binary | ios::trunc);
	if (file.is_open())
	{
		int n = mat.rows(), p = mat.cols();
		cout << "saving data..." << endl;
        file.write((char*) &p, iss);
        file.write((char*) &n, iss);

		float percent = 1 / (float) p * 100;

		for (integer j = 0; j != p; ++j)
		{
			for (int i = 0; i != n; ++i)
			{
                float temp = mat(i, j);
                file.write((char*)&temp, fs);
			}

			cout << "\r" << floor(j * percent) << "%";
		}

		cout << "\r" << "100%" << endl;

		file.close();
	}
	else
	{
		cout << "ERROR: cannot open data file" << endl;
	}
}


void readMLP(const string &input, MLP &mlp)
{
	ifstream file(input, ios::binary);
	if (file.is_open())
	{
		cout << "loading mlp..." << endl;
		integer size;
		file.read((char*) &size, is);
		vector<integer> structure(size);
		for (integer i = 0; i != size; ++i)
			file.read((char*) &structure[i], is);

		layerType layers(structure);
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
					file.read((char*) &value, fs);
					mat(r, c) = value;
				}

				cout << "\r" << floor((sum[i] + r * rows) * percent) << "%";
			}
		}

		cout << "\r" << "100%" << endl;

		mlp.restoreWeights(layers);
		file.close();
	}
}


void writeMLP(const string &output, const MLP &mlp)
{
	ofstream file(output, ios::binary | ios::trunc);

	if (file.is_open())
	{
		cout << "saving mlp..." << endl;
		const vector<integer> structure = mlp.getStructure();
		const integer size = structure.size();
		file.write((char*) &size, is);
		for (integer i = 0; i != size; ++i)
			file.write((char*) &structure[i], is);

		/* --sert juste pour le %--- */
		vector<integer> sum(size);
		sum[0] = 0;
		for (integer i = 1; i != size; ++i)
			sum[i] = sum[i - 1] + (structure[i - 1] + 1) * structure[i];

		float percent = 1 / (float) sum[size - 1] * 100;
		/* ------------------------- */

		const layerType layers = mlp.get();
		integer rows, cols;
		for (integer i = 0; i <= layers.last(); ++i)
		{
			const EigenMatrix &mat = layers[i];
			rows = mat.rows();
			cols = mat.cols();
			for (integer r = 0; r != rows; ++r)
			{
				for (integer c = 0; c != cols; ++c)
				{
					float value = mat(r, c);
					file.write((char*) &value, fs);
				}

				cout << "\r" << floor((sum[i] + r * rows) * percent) << "%";
			}
		}

		cout << "\r" << "100%" << endl;

		file.close();
	}
}


#endif	// IO
