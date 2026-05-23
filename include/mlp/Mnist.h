#ifndef MLP_MNIST_H
#define MLP_MNIST_H

#include <cstddef>
#include <string>
#include <vector>

namespace mlp
{

// Images and labels of the MNIST handwritten digit database
// (http://yann.lecun.com/exdb/mnist/), stored in the IDX format.
//
// The 60000 training and 10000 test images are kept as bytes (about 50 MB)
// and only converted to network inputs when needed.
class Mnist
{
public:
    // Throw std::runtime_error if a file cannot be read or is not a valid
    // IDX file, or if the images and labels do not match.
    static Mnist load(const std::string& imagesPath, const std::string& labelsPath);

    // Load "train-images-idx3-ubyte" and "train-labels-idx1-ubyte" (or the
    // "t10k-" files when test is true) from the given directory.
    static Mnist loadDirectory(const std::string& directory, bool test);

    std::size_t size() const { return m_labels.size(); }
    std::size_t rows() const { return m_rows; }
    std::size_t columns() const { return m_columns; }
    std::size_t pixelCount() const { return m_rows * m_columns; }

    unsigned label(std::size_t i) const { return m_labels.at(i); }
    // Gray levels of image i, row by row, 0 is the background and 255 the ink.
    const unsigned char* pixels(std::size_t i) const;

    // Pixels scaled to [0, 1], one input per pixel.
    void inputs(std::size_t i, std::vector<double>& inputs) const;
    // 10 outputs: 1 for the label of image i, 0 for the other digits.
    void targets(std::size_t i, std::vector<double>& targets) const;

private:
    Mnist();

    std::size_t m_rows;
    std::size_t m_columns;
    std::vector<unsigned char> m_pixels;
    std::vector<unsigned char> m_labels;
};

} // namespace mlp

#endif // MLP_MNIST_H
