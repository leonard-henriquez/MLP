#ifndef TYPES_H
#define TYPES_H

//#define NDEBUG
//#define EIGEN_USE_MKL_ALL

#include <Eigen/Dense>
#include <vector>

// Typedef
using namespace Eigen;
using namespace std;
typedef long long int integer;
typedef float realnumber;
typedef vector<realnumber> STLVector;
typedef Matrix<realnumber, Dynamic, 1> EigenVector;
typedef Matrix<realnumber, Dynamic, Dynamic> EigenMatrix;

template <typename T>
string toStr(T pNumber)
{
	ostringstream oOStrStream;
	oOStrStream << pNumber;
	return oOStrStream.str();
};


#endif // TYPES_H
