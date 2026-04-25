#ifndef MLP_RANDOM_H
#define MLP_RANDOM_H

#include <cstddef>

namespace mlp
{

// Park-Miller "minimal standard" generator. Used instead of std::rand() so
// that a given seed produces the same network on every platform.
class Random
{
public:
    explicit Random(unsigned long seed = 1);

    void seed(unsigned long seed);

    // Uniform value in [min, max).
    double uniform(double min, double max);
    // Uniform index in [0, n).
    std::size_t index(std::size_t n);

private:
    long next();

    long m_state;
};

} // namespace mlp

#endif // MLP_RANDOM_H
