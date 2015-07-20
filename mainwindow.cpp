#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    QMLP::init();

    ui->setupUi(this);
    mlp = new QMLP(ui->HL->value(), ui->PL->value());
    thread = new MLPThread(this, mlp);
    mlpWindow = new MLPDialog(this, mlp);
    exampleWindow = new exampleDialog(this, mlp);
    mnistWindow = new loadMNISTDialog(this, mlp);

    ui->graph->addGraph();

    connect(ui->Compute, &QPushButton::clicked, this, &MainWindow::compute);
    connect(mlpWindow, &MLPDialog::reset, [=]{resetMLP();});
    connect(mlp, &QMLP::newSecond, this, &MainWindow::updateDisplay);
    connect(mlp, &QMLP::disp, ui->Console, &QTextEdit::append);
    connect(thread, &MLPThread::finished, this, &MainWindow::computeFinished);
    connect(ui->HL, static_cast<void (QSpinBox::*)(qint32)>(&QSpinBox::valueChanged), [=](qint32 value){resetMLP(value, -1);});
    connect(ui->PL, static_cast<void (QSpinBox::*)(qint32)>(&QSpinBox::valueChanged), [=](qint32 value){resetMLP(-1, value);});   
}

MainWindow::~MainWindow()
{
    delete ui;
    delete mlp;
}

void MainWindow::resetMLP(qint64 HL, qint64 PL)
{
    if (!thread->isRunning())
    {
        mlp->reset(MLP::INIT, HL, PL);
        ui->graph->graph(0)->clearData();
        ui->graph->replot();
        ui->Console->clear();
        ui->LearningProgress->setValue(0);
    }
}

void MainWindow::compute()
{
    if (thread->isRunning())
    {
        thread->terminate();
        ui->Compute->setText("Compute");
        mlpWindow->enable(1);
    }
    else
    {
        ui->graph->graph(0)->clearData();
        if(mlp->isSet())
        {
            ui->graph->xAxis->setRange(0, ui->MaxTime->value());
            ui->graph->yAxis->setRange(0, mlp->MQE()*1.5);
            ui->Console->append("*****************************");
            mlp->setActivationFunction((ui->ActivationFunction->currentText() == "Tanh")? 1 : 0);
            ui->LearningProgress->setMaximum(ui->MaxTime->value()*10);

            QString str = "maxError:"+ toQStr(ui->MaxError->value()) + ";"
                    +   "maxTime:" + toQStr(ui->MaxTime->value()) + ";"
                    +   "learningRate:"+ toQStr(ui->LearningRate->value()) + ";"
                    +   "lambda:"  + toQStr((ui->WeightDecay->isChecked())? ui->L0->value() : 0) + ";"
                    +   "lambda1:" + toQStr((ui->WeightDecay->isChecked())? ui->L1->value() : 0) + ";"
                    +   "lambda2:" + toQStr((ui->WeightDecay->isChecked())? ui->L2->value() : 0) + ";"
                    +   "adaptativeLearningRate:"+ toQStr(ui->AdaptativeLearningRate->checkState()) + ";";

            thread->start(str);
            mlpWindow->enable(0);
            ui->Compute->setText("Stop");
        }
    }
}

void MainWindow::updateDisplay(const qreal &second)
{
    ui->LearningProgress->setValue(second*10);
    ui->graph->graph(0)->addData(second, mlp->MQE());
    ui->graph->replot();
}

void MainWindow::computeFinished()
{
    ui->LearningProgress->setValue(ui->MaxTime->value()*10);
    mlpWindow->enable(1);
    ui->Compute->setText("Compute");
}

void MainWindow::on_Examples_clicked()
{
    if(exampleWindow->isVisible())
        exampleWindow->hide();
    else
        exampleWindow->show();
}

void MainWindow::on_MLP_clicked()
{
    if(mlpWindow->isVisible())
        mlpWindow->hide();
    else
        mlpWindow->show();
}
