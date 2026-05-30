#include "mlp/Image.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include "mlp/Mnist.h"

namespace mlp
{

namespace
{

// Read the next number of a PNM header, skipping blanks and comments.
unsigned long readHeaderNumber(std::istream& in, const std::string& path)
{
    char c;
    while (in.get(c))
    {
        if (c == '#')
        {
            std::string comment;
            std::getline(in, comment);
        }
        else if (!std::isspace(static_cast<unsigned char>(c)))
        {
            in.putback(c);
            break;
        }
    }
    unsigned long value;
    if (!(in >> value))
        throw std::runtime_error("invalid image header in '" + path + "'");
    return value;
}

// Average of the pixels of `source` covered by the rectangle [x0, x1) x [y0, y1),
// each pixel weighted by the area it shares with the rectangle.
double areaAverage(const Image& source, double x0, double y0, double x1, double y1)
{
    double sum = 0.0, area = 0.0;
    for (std::size_t y = static_cast<std::size_t>(y0); y < source.height() && y < y1; y++)
    {
        double h = std::min<double>(y + 1, y1) - std::max<double>(y, y0);
        for (std::size_t x = static_cast<std::size_t>(x0); x < source.width() && x < x1; x++)
        {
            double w = std::min<double>(x + 1, x1) - std::max<double>(x, x0);
            sum += w * h * source.pixel(x, y);
            area += w * h;
        }
    }
    return area > 0 ? sum / area : 0.0;
}

} // namespace

Image::Image() : m_width(0), m_height(0)
{
}

Image::Image(std::size_t width, std::size_t height, unsigned char value)
    : m_width(width), m_height(height), m_pixels(width * height, value)
{
}

Image Image::load(const std::string& path)
{
    std::ifstream file(path.c_str(), std::ios::binary);
    if (!file)
        throw std::runtime_error("cannot open image '" + path + "'");

    std::string magic;
    file >> magic;
    bool ascii = magic == "P2" || magic == "P3";
    bool colour = magic == "P3" || magic == "P6";
    if (magic != "P2" && magic != "P5" && !colour)
        throw std::runtime_error("'" + path + "' is not a PGM or PPM image");

    std::size_t width = readHeaderNumber(file, path);
    std::size_t height = readHeaderNumber(file, path);
    unsigned long maxValue = readHeaderNumber(file, path);
    if (width == 0 || height == 0 || maxValue == 0 || maxValue > 255)
        throw std::runtime_error("unsupported image format in '" + path + "'");
    file.get(); // single blank before binary data

    Image image(width, height);
    std::size_t channels = colour ? 3 : 1;
    std::vector<unsigned char> raw(width * height * channels);
    if (ascii)
    {
        for (std::size_t i = 0; i < raw.size(); i++)
        {
            unsigned long value;
            if (!(file >> value) || value > maxValue)
                throw std::runtime_error("invalid pixel data in '" + path + "'");
            raw[i] = static_cast<unsigned char>(value);
        }
    }
    else if (!file.read(reinterpret_cast<char*>(&raw[0]), static_cast<std::streamsize>(raw.size())))
    {
        throw std::runtime_error("truncated pixel data in '" + path + "'");
    }

    for (std::size_t i = 0; i < image.m_pixels.size(); i++)
    {
        double value = colour ? 0.299 * raw[3 * i] + 0.587 * raw[3 * i + 1] + 0.114 * raw[3 * i + 2]
                              : raw[i];
        image.m_pixels[i] = static_cast<unsigned char>(value * 255.0 / maxValue + 0.5);
    }
    return image;
}

void Image::save(const std::string& path) const
{
    std::ofstream file(path.c_str(), std::ios::binary);
    if (!file)
        throw std::runtime_error("cannot write image '" + path + "'");
    file << "P5\n" << m_width << ' ' << m_height << "\n255\n";
    if (!m_pixels.empty())
        file.write(reinterpret_cast<const char*>(&m_pixels[0]), static_cast<std::streamsize>(m_pixels.size()));
    if (!file)
        throw std::runtime_error("error while writing image '" + path + "'");
}

Image Image::fromMnist(const Mnist& mnist, std::size_t index)
{
    Image image(mnist.columns(), mnist.rows());
    const unsigned char* pixels = mnist.pixels(index);
    std::copy(pixels, pixels + mnist.pixelCount(), image.m_pixels.begin());
    return image;
}

unsigned char Image::pixel(std::size_t x, std::size_t y) const
{
    if (x >= m_width || y >= m_height)
        throw std::out_of_range("pixel out of the image");
    return m_pixels[y * m_width + x];
}

void Image::setPixel(std::size_t x, std::size_t y, unsigned char value)
{
    if (x >= m_width || y >= m_height)
        throw std::out_of_range("pixel out of the image");
    m_pixels[y * m_width + x] = value;
}

Image Image::toMnistFormat() const
{
    const std::size_t SIZE = 28, BOX = 20;
    Image result(SIZE, SIZE);
    if (m_pixels.empty())
        return result;

    // Ink must be bright: invert pictures drawn in dark on a light background.
    unsigned long total = 0;
    for (std::size_t i = 0; i < m_pixels.size(); i++)
        total += m_pixels[i];
    bool invert = total / m_pixels.size() > 127;
    Image ink(m_width, m_height);
    unsigned char lightest = 0;
    for (std::size_t i = 0; i < m_pixels.size(); i++)
    {
        ink.m_pixels[i] = invert ? 255 - m_pixels[i] : m_pixels[i];
        lightest = std::max(lightest, ink.m_pixels[i]);
    }

    // Bounding box of the drawing, ignoring the faint background noise.
    unsigned char threshold = static_cast<unsigned char>(lightest / 4);
    std::size_t left = m_width, right = 0, top = m_height, bottom = 0;
    for (std::size_t y = 0; y < m_height; y++)
    {
        for (std::size_t x = 0; x < m_width; x++)
        {
            if (ink.pixel(x, y) > threshold)
            {
                left = std::min(left, x);
                right = std::max(right, x);
                top = std::min(top, y);
                bottom = std::max(bottom, y);
            }
        }
    }
    if (left > right)
        return result; // blank picture

    // Scale the box so that its longest side is 20 pixels.
    std::size_t boxWidth = right - left + 1, boxHeight = bottom - top + 1;
    double scale = static_cast<double>(BOX) / std::max(boxWidth, boxHeight);
    std::size_t width = std::max<std::size_t>(1, static_cast<std::size_t>(boxWidth * scale + 0.5));
    std::size_t height = std::max<std::size_t>(1, static_cast<std::size_t>(boxHeight * scale + 0.5));

    Image scaled(width, height);
    double sum = 0.0, sumX = 0.0, sumY = 0.0;
    for (std::size_t y = 0; y < height; y++)
    {
        for (std::size_t x = 0; x < width; x++)
        {
            double value = areaAverage(ink, left + x / scale, top + y / scale,
                                       left + (x + 1) / scale, top + (y + 1) / scale);
            // Remove the background and stretch the ink to the full range.
            value = (value - threshold) * 255.0 / (lightest - threshold);
            value = std::max(0.0, std::min(255.0, value));
            scaled.setPixel(x, y, static_cast<unsigned char>(value + 0.5));
            sum += value;
            sumX += value * x;
            sumY += value * y;
        }
    }

    // Move the centre of mass to the centre of the 28x28 image.
    double centreX = sum > 0 ? sumX / sum : width / 2.0;
    double centreY = sum > 0 ? sumY / sum : height / 2.0;
    long offsetX = static_cast<long>(std::floor(SIZE / 2.0 - centreX + 0.5));
    long offsetY = static_cast<long>(std::floor(SIZE / 2.0 - centreY + 0.5));
    offsetX = std::max(0L, std::min(offsetX, static_cast<long>(SIZE - width)));
    offsetY = std::max(0L, std::min(offsetY, static_cast<long>(SIZE - height)));

    for (std::size_t y = 0; y < height; y++)
        for (std::size_t x = 0; x < width; x++)
            result.setPixel(x + offsetX, y + offsetY, scaled.pixel(x, y));
    return result;
}

void Image::inputs(std::vector<double>& inputs) const
{
    inputs.resize(m_pixels.size());
    for (std::size_t i = 0; i < m_pixels.size(); i++)
        inputs[i] = m_pixels[i] / 255.0;
}

void Image::print(std::ostream& out) const
{
    const char shades[] = " .:-=+*#%@";
    for (std::size_t y = 0; y < m_height; y++)
    {
        for (std::size_t x = 0; x < m_width; x++)
        {
            char c = shades[pixel(x, y) * 9 / 255];
            out << c << c; // characters are about twice as high as wide
        }
        out << '\n';
    }
}

} // namespace mlp
