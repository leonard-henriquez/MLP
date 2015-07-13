#include "mlpdialog.h"
#include "ui_mlpdialog.h"

MLPDialog::MLPDialog(QWidget *parent, QMLP *mlp) :
    QMainWindow(parent),
    ui(new Ui::MLPDialog),
    _mlp(mlp)
{
    ui->setupUi(this);
    connect(ui->Load, &QPushButton::clicked, this, &MLPDialog::load);
    connect(ui->Save, &QPushButton::clicked, this, &MLPDialog::save);
    connect(ui->Reset, &QPushButton::clicked, this, &MLPDialog::reset);
}

MLPDialog::~MLPDialog()
{
    delete ui;
}

void MLPDialog::enable(bool state)
{
    ui->Load->setEnabled(state);
    ui->Save->setEnabled(state);
    ui->Reset->setEnabled(state);
}

void MLPDialog::load()
{
    QFile f(QFileDialog::getOpenFileName(this, "Open...", "", "mlp (*.mlp)"));
    *_mlp = QMLP::load(&f);
    QMessageBox msgBox;
    msgBox.setText("MLP and examples loaded.");
    msgBox.exec();
    emit MLPLoaded();
}

void MLPDialog::save()
{
    QFile f(QFileDialog::getSaveFileName(this, "Save as...", "", "mlp (*.mlp)"));
    QMLP::save(&f, *_mlp);
}
