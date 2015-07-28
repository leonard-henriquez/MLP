TEMPLATE = app
CONFIG += console
CONFIG += c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    multilayerperceptron.cpp

INCLUDEPATH += $$PWD/Eigen

include(deployment.pri)
qtcAddDeployment()

HEADERS += \
    mlpmath.h \
    types.h \
    multilayerperceptron.h \
    multilayerperceptron_global.h

