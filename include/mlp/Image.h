#ifndef MLP_IMAGE_H
#define MLP_IMAGE_H

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

namespace mlp
{

class Mnist;

// Grayscale image with 8 bit pixels, read from and written to the PGM
// format (also reads colour PPM files, converted to gray).
class Image
{
public:
    Image();
    Image(std::size_t width, std::size_t height, unsigned char value = 0);

    // Throw std::runtime_error if the file is not a valid PGM/PPM image.
    static Image load(const std::string& path);
    void save(const std::string& path) const;

    static Image fromMnist(const Mnist& mnist, std::size_t index);

    std::size_t width() const { return m_width; }
    std::size_t height() const { return m_height; }
    unsigned char pixel(std::size_t x, std::size_t y) const;
    void setPixel(std::size_t x, std::size_t y, unsigned char value);

    // Prepare any picture of a single digit the way MNIST images were made:
    // white ink on a black background (light backgrounds are inverted), the
    // drawing cropped and scaled to fit in a 20x20 box, then placed in a
    // 28x28 image so that its centre of mass is in the middle.
    Image toMnistFormat() const;

    // Pixels scaled to [0, 1], row by row.
    void inputs(std::vector<double>& inputs) const;

    // Draw the image with characters, darker characters for brighter pixels.
    void print(std::ostream& out) const;

private:
    std::size_t m_width;
    std::size_t m_height;
    std::vector<unsigned char> m_pixels;
};

} // namespace mlp

#endif // MLP_IMAGE_H
