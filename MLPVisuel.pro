#-------------------------------------------------
#
# Project created by QtCreator 2015-06-08T19:12:22
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets printsupport

TARGET = MLPVisuel
TEMPLATE = app

CONFIG   += c++11


SOURCES += main.cpp\
    mainwindow.cpp \
    multilayerperceptron.cpp \
    types.cpp \
    qmlp.cpp \
    qcustomplot.cpp \
    exampledialog.cpp \
    mlpdialog.cpp \
    loadmnistdialog.cpp

HEADERS  += mainwindow.h \
    multilayerperceptron.h \
    multilayerperceptron_global.h \
    types.h \
    qmlp.h \
    qcustomplot.h \
    exampledialog.h \
    mlpdialog.h \
    mlpmath.h \
    loadmnistdialog.h

FORMS    += mainwindow.ui \
    mlpdialog.ui \
    exampledialog.ui \
    loadmnistdialog.ui

INCLUDEPATH += $$PWD/Eigen
