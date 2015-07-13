#include "exampledialog.h"
#include "ui_exampledialog.h"

exampleDialog::exampleDialog(QWidget *parent, QMLP *mlp) :
    QMainWindow(parent),
    ui(new Ui::exampleDialog),
    _mlp(mlp)
{
    ui->setupUi(this);

    model = new QStandardItemModel(this);
    validationModel = new QStandardItemModel(this);
    ui->PickExample->setModel(model);
    ui->PickEntry->setModel(model);
    ui->PickExample->setSelectionMode(QListView::SingleSelection);

    signalMapper = new QSignalMapper(this);
    signalMapper->setMapping(ui->AddExample, 1);
    signalMapper->setMapping(ui->RemoveExample, 0);
    connect(ui->AddExample, SIGNAL(clicked()), signalMapper, SLOT(map()));
    connect(ui->RemoveExample, SIGNAL(clicked()), signalMapper, SLOT(map()));
    connect(signalMapper, SIGNAL(mapped(qint32)), this, SLOT(addOrRemoveExample(qint32)));

    signalMapper2 = new QSignalMapper(this);
    signalMapper2->setMapping(ui->AddEntry, 1);
    signalMapper2->setMapping(ui->RemoveEntry, 0);
    connect(ui->AddEntry, SIGNAL(clicked()), signalMapper2, SLOT(map()));
    connect(ui->RemoveEntry, SIGNAL(clicked()), signalMapper2, SLOT(map()));
    connect(signalMapper2, SIGNAL(mapped(qint32)), this, SLOT(addOrRemoveEntry(qint32)));

    connect(ui->PickExample->selectionModel(), SIGNAL(currentChanged(const QModelIndex&, const QModelIndex&)), this, SLOT(refreshView()));
    connect(ui->InputOrOutput, SIGNAL(currentTextChanged(const QString&)), this, SLOT(refreshView()));

    connect(ui->LoadExamples, SIGNAL(clicked()), this, SLOT(loadExamples()));
    connect(ui->SaveExamples, SIGNAL(clicked()), this, SLOT(saveExamples()));

}

exampleDialog::~exampleDialog()
{
    delete ui;
}

void exampleDialog::loadExamples()
{
    QFile file(QFileDialog::getOpenFileName(this, "Open", QDir::homePath(), "data (*.dat)"));
    QString exampleName, total, input, output;
    QStringList rawExamples;
    QStandardItem *ex, *inputParent, *outputParent, *inputChild, *outputChild;

    model->clear();

    if(!file.open(QIODevice::ReadOnly))
        QMessageBox::critical(this, "Error", "Failed to open the data file.");
    else
    {
        total = file.readAll();
        file.close();


        QRegExp exp1("input size:*;");
        exp1.setPatternSyntax(QRegExp::Wildcard);
        QRegExp exp2("output size:*;");
        exp2.setPatternSyntax(QRegExp::Wildcard);
        if (total.contains(exp1) & total.contains(exp2))
        {
            qint8 inputSize =  total.split("input size:")[1].split(";")[0].toInt();
            qint8 outputSize =  total.split("output size:")[1].split(";")[0].toInt();
            rawExamples = QString(total).simplified().remove(" ").split(";", QString::SkipEmptyParts);


            foreach (QString str, rawExamples)
            {
                if (str.contains(":") & str.split(":", QString::SkipEmptyParts)[1].contains("->"))
                {
                    exampleName = str.split(":", QString::SkipEmptyParts)[0];
                    input = str.split(":", QString::SkipEmptyParts)[1].split("->", QString::SkipEmptyParts)[0];
                    output = str.split(":", QString::SkipEmptyParts)[1].split("->", QString::SkipEmptyParts)[1];

                    ex = new QStandardItem(exampleName);

                    inputParent = new QStandardItem("input");
                    for (qint8 i = 0; i < inputSize; ++i)
                    {
                        inputChild = new QStandardItem();
                        if (i <= input.count(","))
                            inputChild->setData(input.section(",", i,i).toDouble(), Qt::EditRole);
                        else
                            inputChild->setData(qreal(0), Qt::EditRole);
                        inputParent->appendRow(inputChild);
                    }
                    ex->appendRow(inputParent);
                    outputParent = new QStandardItem("output");
                    for (qint8 i = 0; i < outputSize; ++i)
                    {
                       outputChild = new QStandardItem();
                       if (i <= input.count(","))
                            outputChild->setData(output.section(",", i,i).toDouble(), Qt::EditRole);
                       else
                            outputChild->setData(qreal(0), Qt::EditRole);
                       outputParent->appendRow(outputChild);
                    }
                    ex->appendRow(outputParent);
                    model->appendRow(ex);
                }
            }
        }
    }
    ui->PickExample->setCurrentIndex(model->index(0,0));
    refreshView();
}

void exampleDialog::saveExamples()
{
    QStandardItem* top = model->invisibleRootItem();
    QFile MLPRawData(QFileDialog::getSaveFileName(this, "Save as...", QDir::homePath(), "data (*.dat)"));
    QByteArray outputFile;
    if(!MLPRawData.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text))
        QMessageBox::critical(this, "Error", "Failed to open the data file.");
    else
    {
        if (top->hasChildren())
        {
            outputFile.append("input size:"+QString::number(top->child(0)->child(0)->rowCount())+"\n;");
            outputFile.append("output size:"+QString::number(top->child(0)->child(1)->rowCount())+"\n;");
        }
        for(qint8 i = 0; i < top->rowCount(); ++i)
        {
            outputFile.append(top->child(i)->data(Qt::DisplayRole).toString()+"\t:");
            for(qint8 j = 0; j < top->child(i)->child(0)->rowCount(); ++j)
                outputFile.append(top->child(i)->child(0)->child(j)->data(Qt::EditRole).toString()+"\t,");
            outputFile.append("->");
            for(qint8 j = 0; j < top->child(i)->child(1)->rowCount(); ++j)
                outputFile.append(top->child(i)->child(1)->child(j)->data(Qt::EditRole).toString()+"\t,");
            outputFile.append(";\n");
        }
        MLPRawData.write(outputFile);
        MLPRawData.close();
    }
}

void exampleDialog::addOrRemoveExample(qint32 add)
{
    if (add)
    {
        QStandardItem *newItem = new QStandardItem("new");
        newItem->appendRow(new QStandardItem("input"));
        newItem->appendRow(new QStandardItem("output"));
        model->appendRow(newItem);

        if(model->rowCount() > 1)
        {
            QStandardItem *newItemChild;
            for (qint8 i = 0; i < model->item(0)->child(0)->rowCount(); ++i)
            {
                newItemChild = new QStandardItem();
                newItemChild->setData(qreal(0), Qt::EditRole);
                newItem->child(0)->appendRow(newItemChild);
            }
            for (qint8 i = 0; i < model->item(0)->child(1)->rowCount(); ++i)
            {
                newItemChild = new QStandardItem();
                newItemChild->setData(qreal(0), Qt::EditRole);
                newItem->child(1)->appendRow(newItemChild);
            }
        }
    }
    else
    {
        model->removeRow(ui->PickExample->selectionModel()->currentIndex().row());
    }
    refreshView();
}

void exampleDialog::addOrRemoveEntry(qint32 add)
{
    if (add)
    {
        for (qint8 i = 0; i < model->rowCount(); ++i)
        {
            QStandardItem *newItem = new QStandardItem();
            newItem->setData(qreal(0), Qt::EditRole);
            model->item(i,0)->child((ui->InputOrOutput->currentText() == "input") ? 0 : 1 , 0)->appendRow(newItem);
        }
    }
    else
    {
        for (qint8 i = 0; i < model->rowCount(); ++i)
            model->item(i,0)->child((ui->InputOrOutput->currentText() == "input") ? 0 : 1 , 0)->removeRow(model->item(i,0)->child((ui->InputOrOutput->currentText() == "input") ? 0 : 1 , 0)->rowCount()-1);
    }
    refreshView();
}

void exampleDialog::refreshView()
{
    QModelIndex selectedItem = ui->PickExample->selectionModel()->currentIndex();
    QModelIndex entryIndex;
    if (selectedItem == QModelIndex())
        entryIndex = model->index(0,0).child((ui->InputOrOutput->currentText() == "input") ? 0 : 1 , 0);
    else
        entryIndex = selectedItem.child((ui->InputOrOutput->currentText() == "input") ? 0 : 1 , 0);
    ui->PickEntry->setRootIndex(entryIndex);

    QString exampleName;
    QStringList examplesNameFormated;
    for (qint8 i = 0; i < model->invisibleRootItem()->rowCount(); ++i)
    {
        exampleName = model->invisibleRootItem()->child(i)->text();
        while(examplesNameFormated.contains(exampleName))
        {
            if(exampleName.contains(QRegExp("_[0-9]+$")))
            {
                qint8 i = exampleName.section("_", -1).toInt()+1;
                exampleName.truncate(exampleName.lastIndexOf("_"));
                exampleName+= "_" + QString::number(i);
            }
            else
                exampleName.append("_1");
            model->invisibleRootItem()->child(i)->setText(exampleName);
        }
        examplesNameFormated.append(exampleName);
    }
}

void exampleDialog::saveToMLP()
{
    qint8 nbExamples = model->rowCount();
    if (nbExamples > 0)
    {
        qint8 nbInputs = model->item(0)->child(0)->rowCount(), nbOutputs = model->item(0)->child(1)->rowCount();
        EigenMatrix input(nbInputs, nbExamples), output(nbOutputs, nbExamples);
        for (qint8 i = 0; i < nbExamples; ++i)
        {
            for (qint8 j = 0; j < nbInputs; ++j)
                input(j,i) = model->item(i)->child(0)->child(j)->data(Qt::EditRole).toDouble();
            for (qint8 j = 0; j < nbOutputs; ++j)
                output(j,i) = model->item(i)->child(1)->child(j)->data(Qt::EditRole).toDouble();
        }
        _mlp->setInput(input);
        _mlp->setOutput(output);
    }
}

void exampleDialog::loadFromMLP()
{
    model->clear();
    EigenMatrix input = _mlp->getInput(), output = _mlp->getOutput();
    qint8 nbExamples = input.cols();
    if (nbExamples > 0)
    {
        for (qint8 i = 0; i < nbExamples; ++i)
        {
            QStandardItem *newItem = new QStandardItem("ex_"+QString::number(i));
            newItem->appendRow(new QStandardItem("input"));
            newItem->appendRow(new QStandardItem("output"));
            model->appendRow(newItem);

            QStandardItem *newItemChild;
            for (qint8 j = 0; j < input.rows(); ++j)
            {
                newItemChild = new QStandardItem();
                newItemChild->setData(input(j,i), Qt::EditRole);
                newItem->child(0)->appendRow(newItemChild);
            }
            for (qint8 j = 0; j < output.rows(); ++j)
            {
                newItemChild = new QStandardItem();
                newItemChild->setData(output(j,i), Qt::EditRole);
                newItem->child(1)->appendRow(newItemChild);
            }
        }
    }
    refreshView();
}

void exampleDialog::on_pushButton_clicked()
{

}
