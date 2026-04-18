#include "neurone.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <time.h>
#include <stdlib.h>
#define KA 100 // coefficient de convergence de la fonction sigmoide

using namespace std;

//   Class Neurone   //

neurone::neurone() : m_arrondi(1000)
{
    // Constructeur
}

/*
neurone::neurone(neurone & N_E) : m_arrondi(1000), m_biais(1)
{
    // Constructeur
    //srand(time(NULL));
    //m_poids.resize(1,float(rand()%1000)/1000);
    m_poids.resize(1,1);
    m_entrees.resize(1, & N_E);
    m_valeur = f(entrees_ponderees() - m_biais);
}
*/

neurone::neurone(std::vector <neurone*> & N_E) : m_arrondi(1000), m_biais(1), m_entrees(N_E)
{
    // Constructeur
    //srand(time(NULL));
    //m_poids.resize(1,float(rand()%1000)/1000);
    m_poids.resize(m_entrees.size(),1);
    m_valeur = f(entrees_ponderees() - m_biais);
}

//neurone::neurone(std::vector <neurone> N_E, std::vector <neurone> N_S) : m_arrondi(1000), m_biais(1), m_entrees(N_E), m_sorties(N_S)
//{
//    // Constructeur
//    //srand(time(NULL));
//    //m_poids.resize(1,float(rand()%1000)/1000);
//    m_poids.resize(1,1);
//    m_valeur = f(entrees_ponderees() - m_biais);
//}

neurone::~neurone()
{
    // Destructeur
}

const neurone neurone::operator=(neurone const &N)
{
    if (&N != this)
    {
        this->m_poids = N.m_poids;
        this->m_entrees = N.m_entrees;
        this->m_sorties = N.m_sorties;
        this->m_biais = N.m_biais;
        this->m_arrondi = N.m_arrondi;
        this->m_valeur = N.m_valeur;
    }
    return *this;
}

void neurone::modifier_entrees(std::vector <neurone*> N_E)
{
    m_entrees = N_E;
}

std::vector <neurone*> neurone::entrees()
{
    return m_entrees;
}

float neurone::poids(unsigned no)
{
    return m_poids[no];
}

void neurone::poids(unsigned no, float valeur)
{
    if (no < m_poids.size())
    {
        if (fabs(valeur*m_arrondi-floor(valeur*m_arrondi))<=1)
        {
            valeur = floor(valeur*m_arrondi)/m_arrondi;
        }
        if (fabs(valeur*m_arrondi-floor(valeur*m_arrondi)-1)<=1)
        {
            valeur = floor(valeur*m_arrondi+1)/m_arrondi;
        }

        m_poids[no] = valeur;
    }
}

float neurone::entrees_ponderees()
{
    float somme_ponderee = 0;
    for (unsigned i = 0; i < m_entrees.size(); i++)
    {
        try
        {
            somme_ponderee += (*m_entrees.at(i)).m_valeur * m_poids.at(i); // accès en dehors des limites !
        }

        catch (const char * Msg)
        {
            std::cout << Msg;
        }
    }
    return somme_ponderee;
}

float neurone::biais()
{
    return m_biais;
}

void neurone::biais(float biais)
{
    if (fabs(biais*m_arrondi-floor(biais*m_arrondi))<=1)
    {
        biais = floor(biais*m_arrondi)/m_arrondi;
    }
    if (fabs(biais*m_arrondi-floor(biais*m_arrondi)-1)<=1)
    {
        biais = floor(biais*m_arrondi+1)/m_arrondi;
    }
    m_biais = biais;
}

float neurone::f(float x)
{
//    if (x>0)
//    return 1;
//    else
//    return 0;
    return 1/(1+exp(-KA*x));
}

float neurone::f_(float x)
{
//    if (x>0)
//    return 1;
//    else
//    return 0;
    return (1/(1+exp(-KA*x)))*(1-1/(1+exp(-KA*x)));
}

float neurone::calcule()
{
    m_valeur = f(entrees_ponderees()-m_biais);
    return m_valeur;
}

float neurone::f_calcule()
{
    return f_(entrees_ponderees()-m_biais);
}

void neurone::affiche()
{
    for (unsigned i = 0; i < 1; i++)
    {
        (*m_entrees.at(i)).affiche();
        cout << "poids " << i << ": " << m_poids.at(i) << "\n";
    }
    cout << "seuil: " << m_biais << "\n";
    cout << "valeur: " << m_valeur << "\n\n";
}
