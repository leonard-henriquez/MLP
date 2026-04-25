#include "mlp/Random.h"

namespace mlp
{

namespace
{

const long MODULUS = 2147483647; // 2^31 - 1
const long MULTIPLIER = 48271;
// Schrage's method: compute (MULTIPLIER * state) % MODULUS without overflow.
const long QUOTIENT = MODULUS / MULTIPLIER;
const long REMAINDER = MODULUS % MULTIPLIER;

} // namespace

Random::Random(unsigned long seed) : m_state(1)
{
    this->seed(seed);
}

void Random::seed(unsigned long seed)
{
    m_state = static_cast<long>(seed % (MODULUS - 1)) + 1;
}

long Random::next()
{
    long hi = m_state / QUOTIENT;
    long lo = m_state % QUOTIENT;
    m_state = MULTIPLIER * lo - REMAINDER * hi;
    if (m_state <= 0)
        m_state += MODULUS;
    return m_state;
}

double Random::uniform(double min, double max)
{
    double unit = static_cast<double>(next() - 1) / (MODULUS - 1);
    return min + (max - min) * unit;
}

std::size_t Random::index(std::size_t n)
{
    std::size_t i = static_cast<std::size_t>(uniform(0.0, static_cast<double>(n)));
    return i < n ? i : n - 1;
}

} // namespace mlp
