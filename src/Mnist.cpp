#include "mlp/Mnist.h"

#include <fstream>
#include <stdexcept>

namespace mlp
{

namespace
{

// IDX files store their integers in big endian.
unsigned long readInt(std::istream& in, const std::string& path)
{
    unsigned char bytes[4];
    if (!in.read(reinterpret_cast<char*>(bytes), 4))
        throw std::runtime_error("truncated IDX header in '" + path + "'");
    return (static_cast<unsigned long>(bytes[0]) << 24) | (static_cast<unsigned long>(bytes[1]) << 16)
           | (static_cast<unsigned long>(bytes[2]) << 8) | static_cast<unsigned long>(bytes[3]);
}

// Read an IDX file of unsigned bytes with the given number of dimensions.
std::vector<unsigned char> readIdx(const std::string& path, unsigned dimensionCount,
                                   std::vector<unsigned long>& dimensions)
{
    std::ifstream file(path.c_str(), std::ios::binary);
    if (!file)
        throw std::runtime_error("cannot open '" + path + "'");

    // Magic number: 0, 0, data type (0x08 = unsigned byte), number of dimensions.
    if (readInt(file, path) != (0x0800UL | dimensionCount))
        throw std::runtime_error("'" + path + "' is not an IDX file of the expected type"
                                 " (is it still gzipped?)");

    unsigned long total = 1;
    dimensions.resize(dimensionCount);
    for (unsigned d = 0; d < dimensionCount; d++)
    {
        dimensions[d] = readInt(file, path);
        total *= dimensions[d];
    }

    std::vector<unsigned char> data(total);
    if (total > 0 && !file.read(reinterpret_cast<char*>(&data[0]), static_cast<std::streamsize>(total)))
        throw std::runtime_error("truncated data in '" + path + "'");
    return data;
}

} // namespace

Mnist::Mnist() : m_rows(0), m_columns(0)
{
}

Mnist Mnist::load(const std::string& imagesPath, const std::string& labelsPath)
{
    Mnist mnist;
    std::vector<unsigned long> imageDimensions, labelDimensions;
    mnist.m_pixels = readIdx(imagesPath, 3, imageDimensions);
    mnist.m_labels = readIdx(labelsPath, 1, labelDimensions);
    mnist.m_rows = imageDimensions[1];
    mnist.m_columns = imageDimensions[2];

    if (imageDimensions[0] != labelDimensions[0])
        throw std::runtime_error("'" + imagesPath + "' and '" + labelsPath
                                 + "' do not have the same number of entries");
    for (std::size_t i = 0; i < mnist.m_labels.size(); i++)
    {
        if (mnist.m_labels[i] > 9)
            throw std::runtime_error("invalid digit label in '" + labelsPath + "'");
    }
    return mnist;
}

Mnist Mnist::loadDirectory(const std::string& directory, bool test)
{
    std::string prefix = directory + "/" + (test ? "t10k" : "train");
    return load(prefix + "-images-idx3-ubyte", prefix + "-labels-idx1-ubyte");
}

const unsigned char* Mnist::pixels(std::size_t i) const
{
    if (i >= size())
        throw std::out_of_range("MNIST image index out of range");
    return &m_pixels[i * pixelCount()];
}

void Mnist::inputs(std::size_t i, std::vector<double>& inputs) const
{
    const unsigned char* p = pixels(i);
    inputs.resize(pixelCount());
    for (std::size_t j = 0; j < inputs.size(); j++)
        inputs[j] = p[j] / 255.0;
}

void Mnist::targets(std::size_t i, std::vector<double>& targets) const
{
    targets.assign(10, 0.0);
    targets[label(i)] = 1.0;
}

} // namespace mlp
