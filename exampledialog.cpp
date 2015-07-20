#include "exampledialog.h"
#include "ui_exampledialog.h"

exampleDialog::exampleDialog(QWidget *parent, QMLP *mlp) :
    QMainWindow(parent),
    ui(new Ui::exampleDialog),
    _mlp(mlp)
{
    ui->setupUi(this);
    connect(ui->input, SIGNAL(valueChanged(int)), this, SLOT(refreshView(int)));
}

exampleDialog::~exampleDialog()
{
    delete ui;
}

void exampleDialog::refreshView(int index)
{
    qint64 dMax = 0,
           rMax = 0;
    qreal dValue = -10,
          rValue = -10;
    EigenVector input = _mlp->getInput().col(index),
                desiredOutput = _mlp->getOutput().col(index),
                realOutput = _mlp->run(index);
    QImage image(28, 28, QImage::Format_Grayscale8);


    for (qint64 i = 0; i < 28*28 ; ++i)
    {
        qint8 value = 255-input(i);
        quint32 dot = qRgb(value, value, value);
        image.setPixel(i%28, floor(i/28), dot);
    }

    for (qint64 i = 0; i < 10; ++i)
    {
        if (desiredOutput(i) > dValue)
        {
            dMax = i;
            dValue = desiredOutput(i);
        }
        if (realOutput(i) > rValue)
        {
            rMax = i;
            rValue = realOutput(i);
        }
    }


    QPixmap pic = QPixmap::fromImage(image);
    ui->img->setPixmap(pic);
    ui->img->setScaledContents(1);
    ui->lcdNumber->display(qreal(rMax));
}
