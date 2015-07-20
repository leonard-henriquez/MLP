#ifndef DIALOG_H
#define DIALOG_H

#include <QMainWindow>
#include <QtCore>
#include <QStandardItemModel>
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
    void refreshView(int index);

private:
    Ui::exampleDialog *ui;
    QMLP *_mlp;
    QStandardItemModel *model;
};

#endif // DIALOG_H
