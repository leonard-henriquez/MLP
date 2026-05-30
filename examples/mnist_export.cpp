// Write MNIST test images as PGM files, to try the recognize program.
//
//     mnist_export [count] [output-dir] [mnist-dir]
//
// Files are named <index>-<digit>.pgm, e.g. data/mnist/samples/0000-7.pgm.

#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "mlp/Image.h"
#include "mlp/Mnist.h"

int main(int argc, char* argv[])
{
    try
    {
        long count = argc > 1 ? std::atol(argv[1]) : 10;
        std::string output = argc > 2 ? argv[2] : "data/mnist/samples";
        std::string data = argc > 3 ? argv[3] : "data/mnist";

        mlp::Mnist test = mlp::Mnist::loadDirectory(data, true);
        for (long i = 0; i < count && static_cast<std::size_t>(i) < test.size(); i++)
        {
            std::ostringstream path;
            path << output << '/' << std::setfill('0') << std::setw(4) << i << '-' << test.label(i)
                 << ".pgm";
            mlp::Image::fromMnist(test, i).save(path.str());
            std::cout << path.str() << '\n';
        }
        return EXIT_SUCCESS;
    }
    catch (const std::exception& e)
    {
        std::cerr << "error: " << e.what() << " (does the output directory exist?)\n";
        return EXIT_FAILURE;
    }
}
