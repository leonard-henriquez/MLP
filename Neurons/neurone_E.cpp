#include "neurone.h"
#include "neurone_E.h"
#include <vector>
#include <iostream>

using namespace std;

neurone_E::neurone_E(float E) : neurone(), m_valeur(E)
{
    // Constructeur
    //srand(time(NULL));
    //m_poids.resize(1,float(rand()%1000)/1000);
    //m_poids.resize(1,1);
}

//neurone_E::neurone_E(float E, std::vector <neurone> N_S) : neurone(), m_valeur(E), m_sorties(N_S)
//{
//    // Constructeur
//    //srand(time(NULL));
//    //m_poids.resize(1,float(rand()%1000)/1000);
//    //m_poids.resize(1,1);
//}


neurone_E::~neurone_E()
{
    // Destructeur
}

void neurone_E::modifier_entree(float E)
{
    m_entree = E;
}

float neurone_E::entree()
{
    return m_entree;
}

void neurone_E::affiche()
{
    cout << "valeur: " << m_valeur << "\n\n";
}
