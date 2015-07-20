#include "loadmnistdialog.h"
#include "ui_loadmnistdialog.h"

loadMNISTDialog::loadMNISTDialog(QWidget *parent, QMLP *mlp) :
    QMainWindow(parent),
    ui(new Ui::loadMNISTDialog),
    _mlp(mlp)
{
    ui->setupUi(this);
    show();
    thread = new MNISTThread(this, _mlp);
    connect(thread, &MNISTThread::finished, this, &QMainWindow::hide);
    connect(thread, &MNISTThread::state, ui->progressBar, &QProgressBar::setValue);
    string filename1 = QFileDialog::getOpenFileName(this, "Open trainImages", QDir::homePath(), "trainImages").toStdString();
    string filename2 = QFileDialog::getOpenFileName(this, "Open trainLabels", QDir::homePath(), "trainLabels").toStdString();
    thread->start(filename1, filename2);
}

loadMNISTDialog::~loadMNISTDialog()
{
    delete ui;
}

void MNISTThread::start(string filename1, string filename2)
{
    _filename1 = filename1;
    _filename2 = filename2;
    QThread::start();
}

void MNISTThread::run()
{
    _mlp->setInput(readMNISTPics());
    _mlp->setOutput(readMNISTLabels());
    _mlp->setArchitecture(MLP::INIT);
    emit finished();
}

int MNISTThread::reverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

EigenMatrix MNISTThread::readMNISTPics()
{
    EigenMatrix dataSet;

    ifstream file (_filename1,ios::binary);
    if (file.is_open())
    {
        int magicNumber=0, numberOfImages=0, nbRows=0, nbCols=0;
        file.read((char*)&magicNumber,sizeof(magicNumber));
        magicNumber= reverseInt(magicNumber);
        file.read((char*)&numberOfImages,sizeof(numberOfImages));
        numberOfImages= reverseInt(numberOfImages);
        file.read((char*)&nbRows,sizeof(nbRows));
        nbRows= reverseInt(nbRows);
        file.read((char*)&nbCols,sizeof(nbCols));
        nbCols= reverseInt(nbCols);

        numberOfImages = 2000;

        dataSet.resize(nbRows * nbCols, numberOfImages);

        int j=1;
        for(int i=0;i<numberOfImages;++i)
        {
            for(int r=0;r<nbRows;++r)
            {
                for(int c=0;c<nbCols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    dataSet(nbRows * r + c, i) = temp;
                }
            }
            if (j*numberOfImages/100 <= i)
            {
                emit state(j);
                j++;
            }
        }
    }
    file.close();
    return dataSet;
}

EigenMatrix MNISTThread::readMNISTLabels()
{
    EigenMatrix dataSet;

    ifstream file (_filename2,ios::binary);
    if (file.is_open())
    {
        int magicNumber=0, numberOfImages=0;
        file.read((char*)&magicNumber,sizeof(magicNumber));
        magicNumber= reverseInt(magicNumber);
        file.read((char*)&numberOfImages,sizeof(numberOfImages));
        numberOfImages= reverseInt(numberOfImages);
        numberOfImages = 2000;

        dataSet = -EigenMatrix::Ones(10, numberOfImages);

        int j=1;
        for(int i=0;i<numberOfImages;++i)
        {
            unsigned char temp=0;
            file.read((char*)&temp,sizeof(temp));
            dataSet(temp, i) = 1;
            if (j*numberOfImages/100 <= i)
            {
                emit state(j);
                j++;
            }
        }
    }
    file.close();
    return dataSet;
}
