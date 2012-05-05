#include "neurone_S.h"

using namespace std;

neurone_S::neurone_S(neurone N_E) : neurone(N_E)
{
    // Constructeur
    //srand(time(NULL));
    //m_poids.resize(1,float(rand()%1000)/1000);
    //m_poids.resize(1,1);
}

neurone_S::neurone_S(std::vector <neurone*> & N_E) : neurone(N_E)
{
    // Constructeur
    //srand(time(NULL));
    //m_poids.resize(1,float(rand()%1000)/1000);
    //m_poids.resize(1,1);
}

neurone_S::~neurone_S()
{
    // Destructeur
}
