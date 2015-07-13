#ifndef TYPES_H
#define TYPES_H

#include <Eigen/Dense>
#include <vector>
#include <list>

// Typedef
using namespace Eigen;
using namespace std;
typedef signed char integer;
typedef double realnumber;
typedef vector<double> STLVector;
typedef Matrix<double, Dynamic, 1> EigenVector;
typedef Matrix<double, Dynamic, Dynamic> EigenMatrix;

inline STLVector EigenToSTLVector(EigenVector const & vector)
{
    STLVector newVector(vector.size());
    for(int i = 0; i < vector.size(); ++i)
        newVector[i] = vector[i];
    return newVector;
}

inline EigenVector STLToEigenVector(STLVector const & vector)
{
    EigenVector newVector(vector.size());
    for(integer i = 0; i < (integer)vector.size() ; ++i)
        newVector[i] = vector[i];
    return newVector;
}

struct example
{
public:
    example(STLVector inputVector = STLVector(), STLVector desiredOutputVector = STLVector());

    STLVector input;
    STLVector output;
};

struct setOfExamples: vector<example>
{
    void add(STLVector inputVector = STLVector(), STLVector desiredOutputVector = STLVector());
};

EigenMatrix setOfExamplesToEigenInputMatrix(setOfExamples const & set);
EigenMatrix setOfExamplesToEigenOutputMatrix(setOfExamples const & set);

template <typename T>
string toStr(T pNumber)
{
 ostringstream oOStrStream;
 oOStrStream << pNumber;
 return oOStrStream.str();
};


#endif // TYPES_H
