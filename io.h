#ifndef IO
#define IO

#include <fstream>
#include <iostream>
#include <string>

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
//	std::ifstream file(input);
//	if ( file.is_open() )
//	{
//		integer size;
//		file >> size;
//		vector<integer> structure(size);
//		for (integer i = 0; i < size; ++i)
//			file >> structure[i];
//		MLP::arrayOfLayers layers(structure);
//		for (integer k = 0; k <= layers.last(); ++k)
//			for (integer i = 0; i < layers[k].rows(); ++i)
//				for (integer j = 0; j < layers[k].cols(); ++j)
//					file >> layers[k](i, j);
//		mlp.set(layers);
//		file.close();
//	}
    std::ifstream file(input, ios::in);
    if ( file.is_open() )
    {
        vector<integer> structure;
        string str, svalue;

        getline(file, str, '#');
        str = str.substr(0, str.find(";\n", 0));
        integer i = 0, j = str.find(",", 0);
        svalue = stoi(str.substr(i,j-i));
        do
        {
            structure.push_back(stoi(svalue));
            i = j+1;
            j = str.find(",", i);
            svalue = str.substr(i, j-i);
        } while (j != string::npos);


        MLP::arrayOfLayers layers(structure);
        integer rows, cols;
        istringstream matrix, line;
        string smatrix, sline, scoeff;

        for (integer i = 0; i <= layers.last(); ++i)
        {
            rows = layers[i].rows();
            cols = layers[i].cols();

            smatrix.clear();
            getline(file, smatrix, '#');
            matrix.str(smatrix);

            for (integer r = 0; r < rows; ++r)
            {
                getline(matrix, sline, ';\n');
                line.str(sline);
                for (integer c = 0; c < cols; ++c)
                {
                    getline(line, scoeff, ",");
                    layers[i](r, c) = stoi(scoeff);
                }
            }
        }
        mlp.set(layers);
        file.close();
    }
}


void writeMLP(const string &output, const MLP &mlp)
{
//	std::ofstream file(output);
//	if ( !output.empty() && file.is_open() )
//	{
//		MLP::arrayOfLayers layers = mlp.get();
//		file << layers.size() + 1;
//		for (integer i = 0; i < layers.size() + 1; ++i)
//			file << layers.get().at(i);
//		for (integer k = 0; k <= layers.last(); ++k)
//			for (integer i = 0; i < layers[k].rows(); ++i)
//				for (integer j = 0; j < layers[k].cols(); ++j)
//					file << layers[k].coeff(i, j);
//		file.close();
//	}

    std::ofstream file(output, ios::out | ios::trunc);
    if ( file.is_open() )
    {
        const vector<integer> structure = mlp.getStructure();
        string str = to_string(structure[0]);
        for (integer i = 1; i < structure.size(); ++i)
            str += "," + to_string(structure[i]);
        str += ";";

        file.write(str.c_str(), sizeof(char)*str.size());
        str.clear();

        const MLP::arrayOfLayers layers = mlp.get();

        integer rows, cols;
        for (integer i = 0; i <= layers.last(); ++i)
        {
            rows = layers[i].rows();
            cols = layers[i].cols();
            str += "\n#";
            for (integer r = 0; r < rows; ++r)
            {
                str += 	to_string(layers[i](r, 0));
                for (integer c = 1; c < cols; ++c)
                {
                    str += "," + to_string(layers[i](r, c));
                }
                str += ";\n";

                file.write(str.c_str(), sizeof(char)*str.size());
                str.clear();
            }
        }
        file.close();
    }

}


#endif // IO
