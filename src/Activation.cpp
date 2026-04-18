#include "mlp/Activation.h"

#include <cmath>

namespace mlp
{

Activation::Activation(ActivationType type, double steepness)
    : m_type(type), m_steepness(steepness)
{
}

double Activation::value(double x) const
{
    switch (m_type)
    {
    case STEP:
        return x > 0 ? 1.0 : 0.0;
    case SIGMOID:
        return 1.0 / (1.0 + std::exp(-m_steepness * x));
    }
    return 0.0;
}

double Activation::derivative(double x) const
{
    switch (m_type)
    {
    case STEP:
        return 1.0;
    case SIGMOID:
    {
        double s = value(x);
        return m_steepness * s * (1.0 - s);
    }
    }
    return 0.0;
}

} // namespace mlp
