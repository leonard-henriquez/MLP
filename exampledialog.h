#ifndef DIALOG_H
#define DIALOG_H

#include <QMainWindow>
#include <QtCore>
#include <QStandardItemModel>
#include <Eigen/Dense>
#include <QFileDialog>
#include <QMessageBox>
#include "qmlp.h"

namespace Ui {
class exampleDialog;
}

class exampleDialog : public QMainWindow
{
    Q_OBJECT

public:
    explicit exampleDialog(QWidget *parent, QMLP *mlp);
    ~exampleDialog();

public slots:
    void refreshView();
    void loadExamples();
    void saveExamples();
    void saveToMLP();
    void loadFromMLP();
    void addOrRemoveExample(qint32 add);
    void addOrRemoveEntry(qint32 add);

private slots:
    void on_pushButton_clicked();

private:
    Ui::exampleDialog *ui;
    QMLP *_mlp;
    QSignalMapper *signalMapper, *signalMapper2;
    QStandardItemModel *model, *validationModel;
};

#endif // DIALOG_H
