#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "multilayerperceptron.h"
#include "exampledialog.h"
#include "mlpdialog.h"
#include "qmlp.h"
#include "workerthread.h"
#include "qcustomplot.h"
#include <QMainWindow>

class WorkerThread;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void compute();
    void updateDisplay(const qreal & second);
    void computeFinished();
    void resetMLP(qint8 HL = -1, qint8 PL = -1);
    void on_Examples_clicked();
    void on_MLP_clicked();

private:
    Ui::MainWindow *ui;
    exampleDialog *exampleWindow;
    MLPDialog *mlpWindow;
    QMLP *mlp;
    WorkerThread *thread;
};





/********************************************/
class WorkerThread : public QThread
{
    Q_OBJECT

public:
    WorkerThread(QObject* parent, QMLP* mlp): QThread(parent), _mlp(mlp),_str() {}
    void start (const QString &str) { _str=str; QThread::start();}
    void run() Q_DECL_OVERRIDE { _mlp->learn(_str); exit();}

private:
    QMLP *_mlp;
    QString _str;
};

#endif // MAINWINDOW_H
