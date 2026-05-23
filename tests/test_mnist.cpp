#include <fstream>
#include <string>
#include <vector>

#include "check.h"
#include "mlp/Mnist.h"

using mlp::Mnist;

namespace
{

void writeInt(std::ofstream& out, unsigned long value)
{
    char bytes[4] = { static_cast<char>((value >> 24) & 0xFF), static_cast<char>((value >> 16) & 0xFF),
                      static_cast<char>((value >> 8) & 0xFF), static_cast<char>(value & 0xFF) };
    out.write(bytes, 4);
}

// Write a small IDX file in the build directory and return its path.
std::string writeIdx(const std::string& name, const std::vector<unsigned long>& dimensions,
                     const std::vector<unsigned char>& data)
{
    std::string path = "bin/" + name;
    std::ofstream out(path.c_str(), std::ios::binary);
    writeInt(out, 0x0800UL | dimensions.size());
    for (std::size_t d = 0; d < dimensions.size(); d++)
        writeInt(out, dimensions[d]);
    out.write(reinterpret_cast<const char*>(&data[0]), static_cast<std::streamsize>(data.size()));
    return path;
}

} // namespace

void testMnist()
{
    // Three 2x3 images.
    std::vector<unsigned long> imageDimensions(3);
    imageDimensions[0] = 3;
    imageDimensions[1] = 2;
    imageDimensions[2] = 3;
    std::vector<unsigned char> pixels(18, 0);
    pixels[0] = 255;
    pixels[7] = 51;
    std::string images = writeIdx("test-images", imageDimensions, pixels);

    std::vector<unsigned long> labelDimensions(1, 3);
    std::vector<unsigned char> labels(3);
    labels[0] = 7;
    labels[1] = 0;
    labels[2] = 9;
    std::string labelFile = writeIdx("test-labels", labelDimensions, labels);

    Mnist mnist = Mnist::load(images, labelFile);
    CHECK(mnist.size() == 3);
    CHECK(mnist.rows() == 2);
    CHECK(mnist.columns() == 3);
    CHECK(mnist.label(0) == 7);
    CHECK(mnist.label(2) == 9);
    CHECK(mnist.pixels(1)[1] == 51);

    std::vector<double> inputs, targets;
    mnist.inputs(0, inputs);
    CHECK(inputs.size() == 6);
    CHECK(inputs[0] == 1.0);
    CHECK(inputs[1] == 0.0);
    mnist.inputs(1, inputs);
    CHECK_CLOSE(inputs[1], 0.2, 1e-12);

    mnist.targets(0, targets);
    CHECK(targets.size() == 10);
    CHECK(targets[7] == 1.0);
    CHECK(targets[0] == 0.0);

    CHECK_THROWS(mnist.pixels(3));
    // Labels and images swapped, mismatched counts, missing files.
    CHECK_THROWS(Mnist::load(labelFile, images));
    std::vector<unsigned long> twoLabels(1, 2);
    CHECK_THROWS(Mnist::load(images, writeIdx("test-labels-2", twoLabels, labels)));
    CHECK_THROWS(Mnist::loadDirectory("does/not/exist", true));
}
