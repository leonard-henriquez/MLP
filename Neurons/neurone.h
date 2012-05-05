#include <vector>
#ifndef NEURONE_H
#define NEURONE_H

class neurone
{
    public:
        neurone();
        //neurone(neurone & N_E);
        neurone(std::vector <neurone*> & N_E);
        //neurone(std::vector <neurone> N_E, std::vector <neurone> N_S);
        ~neurone();

        const neurone operator=(neurone const &N);

        std::vector <neurone*> entrees(); // get
        void modifier_entrees(std::vector <neurone*> N_E); //modifier
        float poids(unsigned no); // get
        void poids(unsigned no, float valeur); // modifier

        float entrees_ponderees(); // get

        float biais(); // get
        void biais(float biais); // modifier

        float f(float x); // fonction d'activation (à seuil ou sigmoide)
        float f_(float x); // dérivée de la fonction d'activation
        virtual float calcule(); // ce que renvoie le neurone compte tenu de ses entrées, de ses poids et de son biais
        float f_calcule(); // f_(sortie)

        virtual void affiche(); // affiche l'état du neurone : ses poids
        bool apprentissage(std::vector <std::vector <float> > exemples, float pas); // apprentissage grâce à une serié d'exemple et à un pas

    protected:
        int m_arrondi;
        float m_valeur;
        std::vector<float> m_poids;

    private:
        float m_biais;
        std::vector <neurone*> m_entrees;
        std::vector <neurone> m_sorties;

};

#endif // NEURONE_H
