#ifndef WORKERTHREAD_H
#define WORKERTHREAD_H

#include "qmlp.h"
#include "qcustomplot.h"
#include <QThread>

//class WorkerThread : public QThread
//{
//    Q_OBJECT

//public:
//    WorkerThread(QObject* parent, QMLP* mlp): QThread(parent), _mlp(mlp) {}
//    void run() Q_DECL_OVERRIDE { _mlp->learn(); exit();}

//public slots:
//    void updateGraph(const double &value) { _graph->graph(0)->addData(value, _mlp->MQE()); _graph->replot(); }

//private:
//    QCustomPlot *_graph;
//    QMLP *_mlp;
//};

#endif // WORKERTHREAD_H
