#include <vector>
#include "neurone.h"
#ifndef NEURONE_S_H
#define NEURONE_S_H

class neurone_S : public neurone
{
    public:
        neurone_S(neurone N_E);
        neurone_S(std::vector <neurone*> & N_E);
        ~neurone_S();

    private:
        std::vector<neurone*> m_entrees;
        std::vector<float> m_poids;
        float m_biais;
};

#endif // NEURONE_H

