#include "global.h"

// prototypes
vector<integer> stringToVector (string str);
void getArguments (int argc, char* argv[]);
void signalHandler(int signo);
bool setup (MLP &mlp);
bool helpMode ();
bool testMode (const MLP &mlp);
void displayImage (const EigenVector &image);
void displayResults (const EigenVector &outputVector);


int main(int argc, char* argv[])
{
	cout << "\033[1;1H\x1b[2J";
    signal(SIGINT, signalHandler);

	// get arguments from console
	getArguments(argc, argv);
	MLP mlp;

	switch (appMode)
	{
	case LEARNING_MODE:
        images = readData(inputDataFile, numberOfExamples);
        labels = readData(outputDataFile, numberOfExamples);
		setup(mlp);
		mlp.gradientDescent(parameters);
        if (!outputMLPFile.empty())
            writeMLP(outputMLPFile, mlp);
		break;

	case TEST_MODE:
        images = readData(inputDataFile, numberOfExamples);
        labels = readData(outputDataFile, numberOfExamples);
		setup(mlp);
		testMode(mlp);
		break;

	default:
		helpMode();
		break;
	}

	return 0;
}


void getArguments(int argc, char* argv[])
{
	parameters.maxTime = 30;
	parameters.learningRate = 0.0005;

	static struct option long_options[] =
	{
		{"input", required_argument, NULL, 'i'},
		{"output", required_argument, NULL, 'o'},
		{"image", required_argument, NULL, 'd'},
		{"label", required_argument, NULL, 'l'},
		{"ex", required_argument, NULL, 'e'},
		{"example", required_argument, NULL, 'e'},
		{"examples", required_argument, NULL, 'e'},
		{"structure", required_argument, NULL, 's'},
		{"struct", required_argument, NULL, 's'},
		{"alr", no_argument, NULL, 'a'},
		{"lr", required_argument, NULL, 'r'},
		{"time", required_argument, NULL, 't'},
		{"batch", required_argument, NULL, 'b'},
		{"batchp", required_argument, NULL, 'p'},
		{"test", no_argument, NULL, 'm'},
		{"help", no_argument, NULL, 'h'},
		{NULL, 0, NULL,   0}
	};

	char c;
	while ((c = getopt_long(argc, argv, "at:r:i:o:e:d:l:s:h:b:p:m:", long_options, NULL)) != -1)
	{
		switch (c)
		{
		case 'i':
            inputMLPFile = optarg;
			break;
		case 'o':
            outputMLPFile = optarg;
			break;
		case 'd':
            inputDataFile = optarg;
			break;
		case 'l':
            outputDataFile = optarg;
			break;
		case 'e':
			numberOfExamples = atoi(optarg);
			break;
		case 's':
			MLPStructure = optarg;
			break;
		case 'a':
			parameters.adaptativeLearningRate = true;
			break;
		case 'r':
			parameters.learningRate = atof(optarg);
			break;
		case 't':
			parameters.maxTime = atof(optarg);
			break;
		case 'b':
			batchSize = atoi(optarg);
			break;
		case 'c':
			batchPercent = atoi(optarg);
			break;
		case 'm':
			appMode = TEST_MODE;
			break;
		case 'h':
			appMode = HELP_MODE;
			break;
		default:
			cout << "one of the option specified doen not exist\n\nHelp:" << endl;
			appMode = FAIL;
			break;
		}
	}

	if (batchPercent != -1)
		batchSize = floor((float) batchPercent * numberOfExamples / 100);

}

void signalHandler(int signo)
{
    if (signo == SIGINT)
        cout << "\r" << flush << "\n" << "\033[38;5;46mSignal received!\033[m" << endl;
    if (signo == 2)
    {
        cout << "Would you like to stop? [\033[38;5;46mN\033[m/\033[38;5;196my\033[m] " << flush;
        char value = getchar(), yes = 'y', no = 'n';
        if (value == yes)
        {
            if (appMode == LEARNING_MODE)
            {
                cout << "Would you like to save? [\033[38;5;46mY\033[m/\033[38;5;196mn\033[m] " << flush;
                value = getchar();
                cout << endl;
                if (value != no)
                {
                    parameters.maxTime = 0;
                    if (outputMLPFile.empty())
                    {
                        cout << "No output file where specified; MLP will be save in ./mlp" << endl;
                        outputMLPFile = "mlp";
                    }
                }
            }
            else
                exit(0);
        }
    }

}

bool setup(MLP &mlp)
{
	// create learning data from arguments
	learningData data(images, labels);
	if (batchSize != -1)
		data.setBatchSize(batchSize);
	mlp.setLearningData(data);

	if (!MLPStructure.empty())
	{
		mlp.setStructure(stringToVector(to_string(data.inputs()) + "," + MLPStructure + "," + to_string(data.outputs())));
		return 1;
	}
    else if (!inputMLPFile.empty())
	{
        readMLP(inputMLPFile, mlp);
		return 1;
	}
	else
	{
		cout << "No structure specified; Can't setup MLP; use --help" << endl;
		return 0;
	}
}


vector<integer> stringToVector(string str)
{
	vector<integer> vect;
	integer i = str.find(",");
	while (i != string::npos)
	{
		vect.push_back(stoi(str.substr(0, i)));
		str = str.substr(i + 1);
		i = str.find(",");
	}
	vect.push_back(stoi(str));
	return vect;
}


bool helpMode()
{
	cout << "--input or -i ./dir/mlpfile ; this option loads the MLP in mlpfile" << endl;
	cout << "--output or -o./dir/mlpfile ; the mlp will be written in mlpfile after learning" << endl;
	cout << "--image or -d ./dir/trainImages ; if the file 'trainImages' is not in the same folder as this application" << endl;
	cout << "--label or -l ./dir/trainLabels ; if the file 'trainLabels' is not in the same folder as this application" << endl;
	cout << "--example or -e 2000 ; specifies how many images will be loaded from the MNIST database" << endl;
	cout << "--struct or -s '300,150,50' ; this will build a MLP with three hidden layers of 300, 150 and 50 neurons" << endl;
	cout << "--alr or -a (no argument) ; if enabled, learning rate will vary" << endl;
	cout << "--lr or -r 0.0005 ; to set up learning rate at 0.0005" << endl;
	cout << "--time or -t 30 ; to stop learning after 30 seconds" << endl;
	cout << "--batch or -b 200 (integer); to choose the size of batches" << endl;
	cout << "--test or -m ; to use the test mode" << endl;
	return 0;
}


bool testMode(const MLP &mlp)
{
	EigenMatrix realOutputRaw = mlp.run(), desiredOutputRaw = mlp.getLearningData().getOutput();
	EigenVector realOutput(realOutputRaw.cols()), desiredOutput(desiredOutputRaw.cols());
	int countGoodOnes = 0, rejectedOnes = 0, dMax, rMax;
	float dValue, rValue;

	for (int j = 0; j < numberOfExamples; ++j)
	{
		dValue = -10;
		rValue = -10;
		for (int i = 0; i < 10; ++i)
		{
			if (desiredOutputRaw(i, j) > dValue)
			{
				dMax = i;
				dValue = desiredOutputRaw(i, j);
			}
			if (realOutputRaw(i, j) > rValue)
			{
				rMax = i;
				rValue = realOutputRaw(i, j);
			}
		}
		realOutput[j] = (unsigned char) rMax;
		desiredOutput[j] = (unsigned char) dMax;

		if (rValue < 0)
			rejectedOnes++;
		else if (rMax == dMax)
			countGoodOnes++;
	}


	cout << "The MLP got " << countGoodOnes << " right predictions out of " << numberOfExamples - rejectedOnes << endl;
	cout << "The MLP rejected " << rejectedOnes << " examples out of " << numberOfExamples << endl;
    cout << "\nWhich example would you like to display? (Ctrl+C to stop)" << endl;
    int value = 0;
    while (1)
	{
		cin >> value;
		cout << "\n\n" << endl;
		if (cin.good() && value >= 0)
		{
			value = min((int)images.cols(), value);
            displayImage(images.col(value));
            displayResults(desiredOutputRaw.col(value));
			displayResults(realOutputRaw.col(value));

			cout << "MLP predicted that this image is a " << realOutput[value] << " (it is actually a " << desiredOutput[value] << ")" << endl;
            cout << "Which example would you like to display? (Ctrl+C to stop)" << endl;
		}
        else
			cin.clear();
	}

	return 0;
}


void displayImage(const EigenVector &image)
{
	int oldvalue = 232, value = 232;
	cout << "\033[48;5;232m" << std::endl;
	for (int i = 112; i < 672; ++i)
	{
		oldvalue = value;
		value = 232 + floor((float)image(i) / 255 * 24);
		if (oldvalue != value)
			cout << "\033[48;5;" << value << "m ";
		else
			cout << " ";
		if ((i + 1) % 28 == 0)
			cout << "\033[m\n\033[48;5;232m" << std::flush;
	}
	cout << "\033[m" << std::endl;
}


void displayResults(const EigenVector &outputVector)
{
    for (int i = 0; i < 10; ++i)
		cout << "   " << i << "   |";
	cout << endl;
	for (int i = 0; i < 10; ++i)
	{
        float output = (outputVector[i]+1)/2;
		string str = to_string(output);
		str.resize(5);
		cout << " "  << str << " |";
	}
	cout << endl;
}


/*
 * {
 * EigenMatrix mat;
 *
 * }
 *
 * k = nbRows * i + j
 *
 * vector<integer> findSquare(integer i0, integer j0, integer sizeSquare, integer sizeImage)
 * {
 * vector<integer> value;
 * if (i0+sizeSquare <= sizeImage && j0+sizeSquare <= sizeImage)
 * {
 * for (integer i = 0; i < sizeSquare; ++i)
 * for (integer j = 0; j < sizeSquare; ++j)
 * value.push_back(sizeImage * i + j);
 * }
 * return value;
 * }
 */
