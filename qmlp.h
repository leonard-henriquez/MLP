#ifndef QMLP_H
#define QMLP_H

#include "multilayerperceptron.h"
#include <QMessageBox>
#include <QtCore>

template <typename T> QString toQStr(T pNumber) {return QString::number(pNumber);}

class QMLP : public QObject, public MLP
{
    Q_OBJECT

public:
    QMLP(qint64 HL = 0, qint64 PL = 0): MLP(HL,PL) {}
    QMLP(const QMLP &other): QObject(), MLP(static_cast<MLP>(other)) {}
    virtual ~QMLP () {}
    QMLP& operator = (const QMLP &other) { clone(other); return *this;}
    static void init();

    static QMLP load(QIODevice *f);
    static void save(QIODevice *f, const QMLP &mlp);
    bool learn(QString str);

    virtual void display(const string & str) { emit disp(QString::fromStdString(str)); }
    virtual bool displayMQE(const clock_t &start, qreal &nextDisplayTime, const qreal & mqe, const qreal &learningRate, const qreal & refreshTime = 0.1);

public slots:
    void resetSlot() {reset();}

signals:
    void disp(const QString &s);
    void newSecond(const qreal &sec = 0);
    void learningFinished();

private:
    friend QDataStream & operator << (QDataStream & out, const QMLP & mlp);
    friend QDataStream & operator >> (QDataStream & in, QMLP & mlp);
};

Q_DECLARE_METATYPE(QMLP)

QDataStream & operator << (QDataStream & out, const QMLP & mlp);
QDataStream & operator >> (QDataStream & in, QMLP & mlp);

QDataStream & operator << (QDataStream & out, const EigenMatrix & mat);
QDataStream & operator >> (QDataStream & in, EigenMatrix & mat);

#endif // QMLP_H
