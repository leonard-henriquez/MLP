#include "qmlp.h"

bool QMLP::displayMQE(clock_t const &start, qreal &nextDisplayTime, const qreal & mqe, const qreal &learningRate, const qreal & refreshTime)
{
    Q_UNUSED(refreshTime)
    if (MLP::displayMQE(start, nextDisplayTime, mqe, learningRate, 0.1))
    {
        emit newSecond((clock() - start) / (qreal)CLOCKS_PER_SEC);
        return 1;
    }
    else
        return 0;
}

void QMLP::init ()
{
    qRegisterMetaTypeStreamOperators<QMLP>("QMLP");
    qMetaTypeId<QMLP>();
}


bool QMLP::learn(QString str)
{
QMap<QString, qint8> argname;
    argname["maxError"] = 0;
    argname["maxTime"] = 1;
    argname["learningRate"] = 2;
    argname["adaptativeLearningRate"] = 3;
    argname["lambda"] = 4;
    argname["lambda1"] = 5;
    argname["lambda2"] = 6;

    str = str.simplified();
    while (str.contains(QRegExp("\\;\\S")))
        str.insert(str.indexOf(QRegExp("\\;\\S"))+1, "\n");
    while (str.contains(QRegExp("\\s\\:")))
        str.remove(str.indexOf(QRegExp("\\s\\:")), 1);
    while (str.contains(QRegExp("\\:\\s")))
        str.remove(str.indexOf(QRegExp("\\:\\s"))+1, 1);

    QVector<qreal> args(argname.size(), 0);

    foreach(QString key,argname.keys())
        if (str.contains(key+":"))
            args[argname.value(key)] = str.split(key+":")[1].split(";")[0].toDouble();
    bool result = MLP::learn(args[0], args[1], args[2], bool(args[3]), args[4], args[5], args[6]);
    emit learningFinished();
    return result;
}

QMLP QMLP::load(QIODevice *f)
{
    if (f->open(QIODevice::ReadOnly))
    {
        QDataStream in(f);
        QMLP mlp;
        in >> mlp;
        f->close();
        return mlp;
    }
    else
    {
        QMessageBox msgBox;
        msgBox.setText("File cannot be opened.");
        msgBox.exec();
        return QMLP();
    }
}


void QMLP::save(QIODevice *f, const QMLP &mlp)
{
    if(f->open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        QDataStream out(f);
        out << mlp;
        f->close();
    }
    else
    {
        QMessageBox msgBox;
        msgBox.setText("File cannot be opened.");
        msgBox.exec();
    }

}

QDataStream & operator << (QDataStream & out, const QMLP & mlp)
{
    out     << (qint8) mlp.m_perLayer
            << (qint8) mlp.m_last
            << mlp.m_input
            << mlp.m_output
            << mlp.m_mean
            << (realnumber) mlp.m_sigma;
    for (qint8 i = 0; i <= mlp.m_last; ++i)
        out << mlp.m_layers[i];
    return out;
}

QDataStream & operator >> (QDataStream & in, QMLP & mlp)
{
   in      >> mlp.m_perLayer
           >> mlp.m_last
           >> mlp.m_input
           >> mlp.m_output
           >> mlp.m_mean
           >> mlp.m_sigma;
   mlp.reset(MLP::NOTINIT);
   for (qint8 i = 0; i <= mlp.m_last; ++i)
       in  >> mlp.m_layers[i];
    return in;
}


QDataStream & operator << (QDataStream & out, const EigenMatrix & mat)
{
    QString str;
    for(qint8 i = 0; i < mat.rows(); ++i)
    {
        for(qint8 j = 0; j < mat.cols(); ++j)
            str.append(QString::number(mat.coeff(i,j))+",");
        str.truncate(str.size()-1);
        str.append(";");
    }
    out << str;
    return out;
}

QDataStream & operator >> (QDataStream & in, EigenMatrix & mat)
{
    QString str;
    QStringList strCut, rowCut;
    in >> str;
    qint8 jm = 0, im = str.split(";", QString::SkipEmptyParts).size();
    foreach (QString row, str.split(";", QString::SkipEmptyParts))
        jm = max(qint8(row.count(",")+1), jm);

    mat.resize(im,jm);

    strCut = str.split(";", QString::SkipEmptyParts);
    for (qint8 i = 0 ; i < im ; ++i)
    {
        rowCut = strCut.value(i).split(",", QString::SkipEmptyParts);
        for (qint8 j = 0 ; j < jm ; ++j)
            mat(i,j) = rowCut.value(j).toDouble();
    }
    return in;
}
