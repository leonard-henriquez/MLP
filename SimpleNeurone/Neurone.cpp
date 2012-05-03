#include "Neurone.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <time.h>
#include <stdlib.h>
#define KA 100 // coefficient de convergence de la fonction sigmoide

using namespace std;

/*   Class Neurone   */

neurone::neurone(std::vector <neurone> N_E, std::vector <neurone> N_S) : m_biais(1), m_arrondi(1000), m_entrees(N_E), m_sorties(N_S)
{
    // Constructeur
    //srand(time(NULL));
    //m_poids.resize(1,float(rand()%1000)/1000);
    m_poids.resize(1,1);
    m_valeur = f(entrees_ponderees() - m_biais);
}

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

virtual void modifier_entrees(std::vector <neurone> N_E)
{
    m_entrees=N_E;
}

virtual std::vector <neurone> entrees()
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
            somme_ponderee += *m_entrees.at(i) * m_poids.at(i); // accès en dehors des limites !
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
    m_sortie = f(entrees_ponderees()-m_biais);
    return m_sortie;
}

float neurone::derive_calcule()
{
    return f_(entrees_ponderees()-m_biais);
}

void neurone::affiche()
{
    for (unsigned i = 0; i < nombre_entrees(); i++)
    cout << "poids " << i << ": " << poids(i) << "\n";
    cout << "seuil: " << biais() << "\n";
}
/*
void neurone::affiche(vector <vector <float> > exemples)
{

    for (int i = 0; i < nombre_entrees(); i++)
    cout << "poids " << i << ": " << poids(i) << "\n";
    cout << "seuil: " << biais() << "\n";

    cout << exemples[0].size() << endl;

    for (int i = 0; i < exemples.size())
    {
        for (int j = 0; j < exemples[j].size())
        {

        }
    }

        entree(0,1);
        entree(1,1);
        cout << "e1: " << sortie() << "\n";
        entree(0,0);
        entree(1,1);
        cout << "e2: " << sortie() << "\n";
        entree(0,1);
        entree(1,0);
        cout << "e3: " << sortie() << "\n";
        entree(0,0);
        entree(1,0);
        cout << "e4: " << sortie() << "\n";
        cout << "poids 1: " << poids(0) << "\npoids 2: " << poids(1) << "\nseuil: " << biais() << "\n";
        cout << "\n";
}
*/

bool neurone::apprentissage(vector <vector <float> > exemples, float pas)
{
    float s, se, delta;
    unsigned apprentissage = 0 , compteur = 0, nbexemples = exemples.size();

    if (nbexemples == 0)
    {
        cout << "Apprentissage non realise\n";
        return 0;
    }

    while(apprentissage < nbexemples)
    {
        apprentissage = 0;

        for(unsigned i = 0; i < nbexemples ; i++)
        {
            for (unsigned j = 0; j < exemples[i].size() - 1; j++)
            {
                entree(j, exemples[i][j]);
            }
            se = exemples[i][exemples[i].size() - 1];
            // cout << "Sortie attendue: " << se << "\n";
            s = sortie();
            // cout << "Sortie: " << s << "\n";
            delta = pas*(se-s); // si fonction à pas utilisée
            //delta = pas*N.derive_sortie()*(se-s); // si fonction sigmoide utilisée

            if (fabs(se-s) >= float(1)/m_arrondi)
            {
                poids(0,poids(0)+delta*entree(0));
                poids(1,poids(1)+delta*entree(1));
                biais(biais()+delta*-1);
                // affiche();

                // wi(t+1) = wi(t) + µ.((d l - x).e i)
            }
            else
                apprentissage++;

            compteur++;
            if (compteur >= 1000000)
            {
                cout << "Apprentissage non reussi\n";
                return 0;
            }
        }
    }
    cout << "Apprentissage reussi\n";
    affiche();
    return 1;
}
