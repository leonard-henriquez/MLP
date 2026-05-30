#include <fstream>
#include <vector>

#include "check.h"
#include "mlp/Image.h"

using mlp::Image;

void testImage()
{
    // ASCII PGM with a comment, values scaled from a maximum of 15.
    {
        std::ofstream file("bin/test-ascii.pgm");
        file << "P2\n# comment\n3 2\n15\n0 15 5\n15 0 0\n";
    }
    Image ascii = Image::load("bin/test-ascii.pgm");
    CHECK(ascii.width() == 3);
    CHECK(ascii.height() == 2);
    CHECK(ascii.pixel(1, 0) == 255);
    CHECK(ascii.pixel(2, 0) == 85);
    CHECK(ascii.pixel(0, 1) == 255);
    CHECK_THROWS(ascii.pixel(3, 0));

    // Binary round trip.
    ascii.save("bin/test-binary.pgm");
    Image binary = Image::load("bin/test-binary.pgm");
    CHECK(binary.width() == 3);
    CHECK(binary.pixel(2, 0) == 85);

    CHECK_THROWS(Image::load("does/not/exist.pgm"));
    {
        std::ofstream file("bin/test-invalid.pgm");
        file << "P5\n3 2\n255\nab";
    }
    CHECK_THROWS(Image::load("bin/test-invalid.pgm"));

    // A dark 10x40 bar on a white 100x100 picture, in the top left corner.
    Image picture(100, 100, 255);
    for (std::size_t y = 5; y < 45; y++)
        for (std::size_t x = 5; x < 15; x++)
            picture.setPixel(x, y, 0);

    Image digit = picture.toMnistFormat();
    CHECK(digit.width() == 28);
    CHECK(digit.height() == 28);
    // Inverted, scaled to 5x20 and its centre of mass (2, 9.5) moved to
    // (14, 14): columns 12 to 16, rows 5 to 24.
    CHECK(digit.pixel(14, 14) == 255);
    CHECK(digit.pixel(12, 5) == 255);
    CHECK(digit.pixel(16, 24) == 255);
    CHECK(digit.pixel(11, 14) == 0);
    CHECK(digit.pixel(17, 14) == 0);
    CHECK(digit.pixel(14, 4) == 0);
    CHECK(digit.pixel(14, 25) == 0);

    std::vector<double> inputs;
    digit.inputs(inputs);
    CHECK(inputs.size() == 28 * 28);
    CHECK(inputs[14 * 28 + 14] == 1.0);

    // A blank picture gives a blank digit.
    Image blank = Image(50, 50, 255).toMnistFormat();
    CHECK(blank.width() == 28);
    CHECK(blank.pixel(14, 14) == 0);
}
