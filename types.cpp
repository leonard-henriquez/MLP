#include"types.h"

example::example(STLVector inputVector, STLVector desiredOutputVector):
    input(inputVector),
    output(desiredOutputVector)
{
}

void setOfExamples::add(STLVector inputVector, STLVector desiredOutputVector)
{
    push_back(example(inputVector, desiredOutputVector));
}

EigenMatrix setOfExamplesToEigenInputMatrix(setOfExamples const & set)
{
    EigenMatrix newInputSet(set[0].input.size(), set.size());
    for (integer i = 0; i < (integer)set.size(); ++i)
        newInputSet.col(i) = STLToEigenVector(set[i].input);
    return newInputSet;
}

EigenMatrix setOfExamplesToEigenOutputMatrix(setOfExamples const & set)
{
    EigenMatrix  newOutputSet(set[0].output.size(), set.size());
    for (integer i = 0; i < (integer)set.size(); ++i)
        newOutputSet.col(i) = STLToEigenVector(set[i].output);
    return newOutputSet;
}
