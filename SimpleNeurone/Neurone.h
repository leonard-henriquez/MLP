#include <vector>
#ifndef NEURONE_H
#define NEURONE_H

class neurone
{
    public:
        neurone(std::vector <neurone> N_E, std::vector <neurone> N_S);
        ~neurone();

        const neurone operator=(neurone const &N);

        virtual void modifier_entrees(std::vector <neurone> N_E); //modifier
        virtual void modifier_entrees(std::vector <float> N_E);
        float poids(unsigned no); // get
        void poids(unsigned no, float valeur); // modifier

        float entrees_ponderees(); // get

        float biais(); // get
        void biais(float biais); // modifier

        float f(float x); // fonction d'activation (à seuil ou sigmoide)
        float f_(float x); // dérivée de la fonction d'activation
        virtual float calcule(); // ce que renvoie le neurone compte tenu de ses entrées, de ses poids et de son biais
        float f_calcule(); // f_(sortie)

        void affiche(); // affiche l'état du neurone : ses poids
        bool apprentissage(std::vector <std::vector <float> > exemples, float pas); // apprentissage grâce à une serié d'exemple et à un pas

    protected:
        std::vector <neurone> entrees(); // get

        float m_valeur;
        std::vector<float> m_poids;
        int m_arrondi;

    private:
        std::vector<neurone> m_entrees;
        std::vector<neurone> m_sorties;
        float m_biais;

};

/*
class neurone_E : public neurone
{
    public:
        virtual void modifier_entrees(std::vector <float> N_E);
        virtual float calcule(); // ce que renvoie le neurone compte tenu de ses entrées, de ses poids et de son biais

    private:
        std::vector <float> entrees();

        std::vector<float> m_entrees;
        std::vector<neurone> m_sorties;
}

class neurone_I : public neurone
{
    public:

    private:
        std::vector<neurone> m_entrees;
        std::vector<neurone> m_sorties;
        float m_biais;

}

class neurone_S : public neurone
{
    public:
        virtual float calcule(); // ce que renvoie le neurone compte tenu de ses entrées, de ses poids et de son biais

    private:
        std::vector<neurone> m_entrees;
        std::vector<float> m_poids;
        float m_biais;
}
*/
#endif // NEURONE_H
