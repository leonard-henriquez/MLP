#include "neurone.h"
#include <vector>
#ifndef NEURONE_E_H
#define NEURONE_E_H

class neurone_E : public neurone
{
    public:
        neurone_E(float E);
        //neurone_E(float E, std::vector <neurone> N_S);
        ~neurone_E();

        void modifier_entree(float E);
        float entree();

        virtual void affiche(); // affiche l'état du neurone

    private:
        float m_valeur;
        float m_entree;
        std::vector<neurone> m_sorties;
};

#endif // NEURONE_H

