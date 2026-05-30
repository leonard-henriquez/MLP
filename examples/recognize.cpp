// Recognise handwritten digits with a network trained by mnist_train.
//
//     recognize <network.mlp> <image.pgm>...
//
// Images can be of any size, white on black or black on white: they are
// cropped, scaled and centred like the MNIST images before classification.

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

#include "mlp/Image.h"
#include "mlp/Network.h"

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cerr << "usage: " << argv[0] << " <network.mlp> <image.pgm>...\n";
        return EXIT_FAILURE;
    }

    try
    {
        mlp::Network net = mlp::Network::load(argv[1]);
        if (net.inputCount() != 28 * 28 || net.outputCount() != 10)
            throw std::runtime_error("the network was not trained on 28x28 digits");

        int failures = 0;
        for (int a = 2; a < argc; a++)
        {
            try
            {
                mlp::Image image = mlp::Image::load(argv[a]).toMnistFormat();
                std::vector<double> inputs;
                image.inputs(inputs);
                std::vector<double> outputs = net.predict(inputs);

                // Digits sorted by decreasing output.
                std::vector<std::pair<double, int> > ranking;
                for (int d = 0; d < 10; d++)
                    ranking.push_back(std::make_pair(-outputs[d], d));
                std::sort(ranking.begin(), ranking.end());

                std::cout << argv[a] << "\n\n";
                image.print(std::cout);
                std::cout << "\nRecognised digit: " << ranking[0].second << "  (scores:";
                for (int r = 0; r < 3; r++)
                    std::cout << ' ' << ranking[r].second << '=' << std::fixed << std::setprecision(2)
                              << -ranking[r].first;
                std::cout << ")\n\n";
            }
            catch (const std::exception& e)
            {
                std::cerr << "error: " << argv[a] << ": " << e.what() << '\n';
                failures++;
            }
        }
        return failures ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    catch (const std::exception& e)
    {
        std::cerr << "error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
