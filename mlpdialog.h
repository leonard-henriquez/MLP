#ifndef MLPDIALOG_H
#define MLPDIALOG_H

#include <QMainWindow>
#include "qmlp.h"
#include "qcustomplot.h"

namespace Ui {
class MLPDialog;
}

class MLPDialog : public QMainWindow
{
    Q_OBJECT

public:
    explicit MLPDialog(QWidget *parent, QMLP *mlp);
    ~MLPDialog();
    void enable(bool state);
public slots:
    void load();
    void save();

signals:
    void MLPLoaded();
    void reset();

private:
    Ui::MLPDialog *ui;
    QMLP *_mlp;
};

#endif // MLPDIALOG_H
