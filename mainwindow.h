#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "exampledialog.h"
#include "mlpdialog.h"
#include "qmlp.h"
#include "qcustomplot.h"
#include "loadmnistdialog.h"

class MLPThread;




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
    void resetMLP(qint64 HL = -1, qint64 PL = -1);
    void on_Examples_clicked();
    void on_MLP_clicked();

private:
    Ui::MainWindow *ui;
    loadMNISTDialog *mnistWindow;
    exampleDialog *exampleWindow;
    MLPDialog *mlpWindow;
    QMLP *mlp;
    MLPThread *thread;
};





/********************************************/
class MLPThread : public QThread
{
    Q_OBJECT

public:
    MLPThread(QObject* parent, QMLP* mlp): QThread(parent), _mlp(mlp),_str() {}
    void start (const QString &str) { _str=str; QThread::start();}
    void run() Q_DECL_OVERRIDE { _mlp->learn(_str); emit finished();}

signals:
    void finished();

private:
    QMLP *_mlp;
    QString _str;
};


#endif // MAINWINDOW_H
