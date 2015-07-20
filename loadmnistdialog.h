#ifndef LOADMNISTDIALOG_H
#define LOADMNISTDIALOG_H

#include <QMainWindow>
#include <QFileDialog>
#include <fstream>
#include "qmlp.h"

class MNISTThread;

namespace Ui {
class loadMNISTDialog;
}

class loadMNISTDialog : public QMainWindow
{
    Q_OBJECT

public:
    explicit loadMNISTDialog(QWidget *parent, QMLP *mlp);
    ~loadMNISTDialog();

private:
    Ui::loadMNISTDialog *ui;
    MNISTThread *thread;
    QMLP *_mlp;
};


/********************************************/
class MNISTThread : public QThread
{
    Q_OBJECT

public:
    MNISTThread(QWidget* parent, QMLP* mlp): QThread(parent), _mlp(mlp) {}
    static int reverseInt (int i);
    EigenMatrix readMNISTPics();
    EigenMatrix readMNISTLabels();
    void start(string filename1, string filename2);
    void run() Q_DECL_OVERRIDE;

signals:
    void state(int);
    void finished();

private:
    QMLP *_mlp;
    string _filename1,
           _filename2;
};

#endif // LOADMNISTDIALOG_H
