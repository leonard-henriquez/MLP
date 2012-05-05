/*
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "Neurone.h"

using namespace std;

void affiche_neurone(neurone N);
int recuperer_donnees(vector<float> &e1, vector<float>  &e2, vector<float> &d);

const float pas = 0.1;

int main()
{
    // déclare les tableaux contennants les couples d'entrées et les sorties attendues
    vector<float> e1(0), e2(0), d(0);
    // déclare la variable qui contientra le nombre d'exemples traités
    int nbexemple;
    // récupère les donnees du ficher donnees.txt et les stocke dans les tableaux e1, e2 et d
    nbexemple = recuperer_donnees(e1, e2, d);

    // déclare le neurone numéro 1
    neurone neurone_1(1);

    // lui ajoute une seconde ligne
    neurone_1.ajouter_entrees(1);

    // Apprentissage
    int s, se, apprentissage = 0 ;

    while(apprentissage < nbexemple)
    {
        apprentissage = 0;
        for(unsigned i = 0; i < 4 ; i++)
        {
            neurone_1.entree(0, float(e1[i]));
            neurone_1.entree(1, float(e2[i]));
            s = neurone_1.sortie();
            cout << "Sortie: " << s << "\n";
            se = d[i];
            cout << "Sortie attendue: " << se << "\n";

            if (se!=s)
            {
                neurone_1.poids(0,neurone_1.poids(0)+pas*neurone_1.derive_sortie()*(se-s)*neurone_1.entree(0));
                neurone_1.poids(1,neurone_1.poids(1)+pas*neurone_1.derive_sortie()*(se-s)*neurone_1.entree(1));
                // wi(t+1) = wi(t) + µ.((d l - x).e i)
            }
            else
            {
                apprentissage++;
            }
        }
        affiche_neurone(neurone_1);
        cout << "\n";
    }
    affiche_neurone(neurone_1);
    return 0;
}

void affiche_neurone(neurone N)
{
        N.entree(0,1);
        N.entree(1,1);
        cout << "e1: " << N.sortie() << "\n";
        N.entree(0,0);
        N.entree(1,1);
        cout << "e2: " << N.sortie() << "\n";
        N.entree(0,1);
        N.entree(1,0);
        cout << "e3: " << N.sortie() << "\n";
        N.entree(0,0);
        N.entree(1,0);
        cout << "e4: " << N.sortie() << "\n";
        cout << "poids 1: " << N.poids(0) << "\npoids 2: " << N.poids(1) << "\n";
        cout << "\n";
}

int recuperer_donnees(vector<float> &e1, vector<float> &e2, vector<float> &d)
{
    e1.resize(0);
    e2.resize(0);
    d.resize(0);

    unsigned nb_lignes = 0;
    // ouvre le fichier de donnees en mode lecture

    ifstream donnees("C:/Users/Arkhan/Documents/Programmes/SimpleNeurone/bin/Debug/donnees.txt");
    if (!donnees)
    {
        std::cerr << "Erreur dans la lecture du fichier de donnees" << endl;
    }
    // lie le fichier
    // Il est nécéssaire de vérifier que donnees.txt ne se termine pas par un retour chariot //
    while (!donnees.eof())
    {
        e1.push_back(0);
        e2.push_back(0);
        d.push_back(0);
        donnees >> e1[e1.size()-1] >> e2[e2.size()-1] >> d[d.size()-1];
        nb_lignes++;
    }

    return nb_lignes;
}

-------------------------------------------------------------------







#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "Neurone.h"

using namespace std;

void affiche_neurone(neurone N);
vector <vector<float> > recuperer_donnees();
bool apprentissage(neurone &N, vector <vector <float> > exemples);

const float pas = 0.1;

int main()
{

    // déclare la variable qui contientra les exemples et les sortie attendues associées
    vector <vector<float> > exemples;
    // récupère les donnees du ficher donnees.txt et les stocke dans les tableaux e1, e2 et d
    exemples = recuperer_donnees();

    // déclare le neurone numéro 1_1, 1_2 et 2 puis l'ensemble
    neurone neurone_1_1, neurone_1_2, neurone_2;
    vector <neurone> systeme_neuronal(0);
//    systeme_neuronal.push_back(neurone_1_1);
//    systeme_neuronal.push_back(neurone_1_2);
    systeme_neuronal.push_back(neurone_2);

    // lui ajoute une seconde entrée
//    neurone_1_1.ajouter_entrees(1);
//    neurone_1_2.ajouter_entrees(1);
    neurone_2.ajouter_entrees(1);
//    neurone_2.entree(0,neurone_1_1.sortie());
//    neurone_2.entree(1,neurone_1_2.sortie());

//    apprentissage(systeme_neuronal, exemples);
    apprentissage(neurone_2, exemples);

    return 0;
}

bool apprentissage(neurone &N, vector <vector <float> > exemples)
{
    int s, se, delta;
    unsigned apprentissage = 0 , compteur = 0, nbexemples = exemples[exemples.size()-1].size();

    while(apprentissage < nbexemples)
    {
        apprentissage = 0;

        for(unsigned i = 0; i < nbexemples ; i++)
        {
            N.entree(0, float(exemples[0][i]));
            N.entree(1, float(exemples[1][i]));
            s = N.sortie();
            cout << "Sortie: " << s << "\n";
            se = exemples[2][i];
            cout << "Sortie attendue: " << se << "\n";
            delta = pas*N.derive_sortie()*(se-s);

            if (fabs(se-s) >= 0.1)
            {
                N.poids(0,N.poids(0)+delta*N.entree(0));
                N.poids(1,N.poids(1)+delta*N.entree(1));
                N.biais(N.biais()+delta*-1);


                // wi(t+1) = wi(t) + µ.((d l - x).e i)
            }
            else
            {
                apprentissage++;
            }

            compteur++;
            if (compteur >= 1000)
            {
                cout << "Apprentissage non reussi au bout de 10000 essais \n";
                affiche_neurone(N);
                return 0;
            }
        }
        // afficher l'état du neurone après chaque passage
        affiche_neurone(N);
        // cout << "\n";
    }
    affiche_neurone(N);
    return 1;
}

void affiche_neurone(neurone N)
{
        N.entree(0,1);
        N.entree(1,1);
        cout << "e1: " << N.sortie() << "\n";
        N.entree(0,0);
        N.entree(1,1);
        cout << "e2: " << N.sortie() << "\n";
        N.entree(0,1);
        N.entree(1,0);
        cout << "e3: " << N.sortie() << "\n";
        N.entree(0,0);
        N.entree(1,0);
        cout << "e4: " << N.sortie() << "\n";
        cout << "poids 1: " << N.poids(0) << "\npoids 2: " << N.poids(1) << "\nseuil: " << N.biais() << "\n";
        cout << "\n";
}

vector <vector<float> > recuperer_donnees()
{
    // déclare les tableaux contennants les couples d'entrées et les sorties attendues
    vector<float> e1(0), e2(0), d(0);

    // ouvre le fichier de donnees en mode lecture
    ifstream donnees("C:/Users/Arkhan/Documents/Programmes/SimpleNeurone/bin/Debug/donnees.txt");
    if (!donnees)
    {
        std::cerr << "Erreur dans la lecture du fichier de donnees" << endl;
    }
    // lie le fichier
    // Il est nécéssaire de vérifier que donnees.txt ne se termine pas par un retour chariot //
    while (!donnees.eof())
    {
        e1.push_back(0);
        e2.push_back(0);
        d.push_back(0);
        donnees >> e1[e1.size()-1] >> e2[e2.size()-1] >> d[d.size()-1];
    }

    vector <vector <float> > exemples(0);
    exemples.push_back(e1);
    exemples.push_back(e2);
    exemples.push_back(d);

    return exemples;
}










vector <vector<float> > recuperer_donnees(string chemin_fichier_donnees)
{
    //int i = 0;

    // ouvre le fichier de donnees en mode lecture
    ifstream fichier(chemin_fichier_donnees.c_str());
    if (!fichier)
    {
        std::cerr << "Erreur dans la lecture du fichier de donnees" << endl;
    }

    int nb_exemples=0;

    string entete;
    getline (fichier, entete);
    int position = entete.find("exemples: ");
    entete = entete.substr(position+10);
    //cout << entete << endl;
    entete.erase(entete.begin()+1, entete.end());
            //entete.erase(entete.begin()+int(entete.find(" ")), entete.end());
    //cout << entete << endl;
    istringstream nombre(entete);
    nombre >> nb_exemples;
    //cout << nb_exemples << endl;

    //cout << nb_exemples << endl;

    //getline (fichier, entete);
    //cout << entete;
//    while (!fichier.eof()) // Il est nécéssaire de vérifier que donnees.txt ne se termine pas par un retour chariot //
//    {
//        string exemple;
//        fichier >> exemple >> exemple >> exemple;
//        nb_exemples++;
//    }

    vector <vector <float> > exemples(nb_exemples);

    //ifstream donnees(chemin_fichier_donnees.c_str());
    //while (!fichier.eof())
    for (int i=0; i < nb_exemples;)
    {
        exemples[i].push_back(0);
        exemples[i].push_back(0);
        exemples[i].push_back(0);

        fichier >> exemples[i][0] >> exemples[i][1] >> exemples[i][2];
        i++;
    }
    return exemples;
}





_________
    // déclare le neurone
    neurone N_1_1(2), N_1_2(2), N_2(2);
    //N_2.entree(0, N_1_1.sortie());
    //N_2.entree(1, N_1_2.sortie());
    //vector <neurone*> &N;
    //N.resize(0);
    //N.push_back(*N_1_1);
    //N.push_back(*N_1_2);
    //neurone *N;
    //N = &N_1_1;
    //N_2.lier(&N_1_1);
    cout << "S1: " << N_2.sortie() << "\n";
    cout << "S: " << N_1_1.sortie()  << " "<< N_1_2.sortie() << "\n";
    N_1_1.affiche();
    N_1_2.affiche();
    N_1_1.entree(0, 1);
    N_1_1.entree(1, 1);
    N_1_2.entree(0, 1);
    N_1_2.entree(1, 1);
    cout << "S1: " << N_2.sortie() << "\n";
    cout << "S: " << N_1_1.sortie()  << " "<< N_1_2.sortie() << "\n";
    N_1_1.affiche();
    N_1_2.affiche();
    N_2.lier(N_1_1);
    cout << "S2: " << N_2.sortie() << " " << N_2.entree(0) << " " << N_2.entree(1) << "\n";
    N_1_1.entree(0,0);
    N_1_1.entree(1,0);
    cout << "N1: " << N_1_1.sortie() << endl;
    //N_2.lier(&N_1_1);
    //N_2.entree(0, N_1_1.sortie());
    //N_2.entree(1, &N_1_2.sortie());
    //N_2 = N_1_1;

    cout << "S3: " << N_2.sortie() << " " << N_2.entree(0) << " " << N_2.entree(1) << "\n";

    //vector <neurone> N;
    //N.resize(1);
    //N[0]=N_1_1;
    //N[1]=N_1_2;



*/



/*
#include "Neurone.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <time.h>
#include <stdlib.h>
#define KA 100 // coefficient de convergence de la fonction sigmoide

using namespace std;

//   Class Neurone   //

neurone::neurone() : m_biais(1), m_arrondi(1000), m_lie(0), m_liens(0)
{
    // Constructeur
    m_entrees.resize(1,0);
    srand(time(NULL));
    //m_poids.resize(1,float(rand()%1000)/1000);
    m_poids.resize(1,1);
    // m_entrees_ponderees.resize(1);
    sortie();
}

neurone::neurone(unsigned nb_entrees) : m_biais(1), m_arrondi(1000), m_lie(0), m_liens(0)
{
    // Constructeur
    m_entrees.resize(nb_entrees,0);

    srand(time(NULL));
    for (unsigned i = 0; i < nb_entrees; i++)
    {
        //m_poids.push_back(float(rand()%1000)/1000);
        m_poids.push_back(1);
    }

    // m_entrees_ponderees.resize(1);
    sortie();
}

neurone::neurone(unsigned nb_entrees, float biais) : m_biais(biais), m_arrondi(1000), m_lie(0), m_liens(0)
{
    // Constructeur
    m_entrees.resize(nb_entrees,0);

    srand(time(NULL));
    for (unsigned i = 0; i < nb_entrees; i++)
    {
        //m_poids.push_back(float(rand()%1000)/1000);
        m_poids.push_back(1);
    }

    // m_entrees_ponderees.resize(1);
    sortie();
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
        this->m_biais = N.m_biais;
        this->m_arrondi = N.m_arrondi;
        sortie();
    }
    return *this;
}

void neurone::ajouter_entrees(unsigned nb_entrees)
{
    m_entrees.resize(m_entrees.size()+nb_entrees);
    m_poids.resize(m_entrees.size(),1);
    sortie();
}

void neurone::retirer_entrees(unsigned nb_entrees)
{
    m_entrees.resize(m_entrees.size()-nb_entrees);
    m_poids.resize(m_entrees.size());
    sortie();
}

unsigned neurone::nombre_entrees()
{
    return m_entrees.size();
}

float * neurone::entree(unsigned no)
{
    return m_entrees[no];
}

void neurone::entree(unsigned no, float * valeur)
{
    if (no < m_entrees.size())
    {
        m_entrees[no] = valeur;
    }
    sortie();
}

void neurone::lier(neurone &N)
{
    //m_entrees.resize(N.size(),0);
    //m_poids.resize(N.size(),1);
    //for(unsigned i = 0; i < N.size(); i++)
    //{
    //    this->m_entrees[i] = N[i].m_sortie;
    //}
    //entre contient ptr
    //ptr contient un pointeur sur la case N.sortie
    m_entrees[0] = N.m_sortie;
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
    sortie();
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

float * neurone::sortie()
{
    *m_sortie = f(entrees_ponderees()-m_biais);
    return m_sortie;
}

float neurone::derive_sortie()
{
    return f_(entrees_ponderees()-m_biais);
}

void neurone::affiche()
{
    for (unsigned i = 0; i < nombre_entrees(); i++)
    cout << "poids " << i << ": " << poids(i) << "\n";
    cout << "seuil: " << biais() << "\n";
}

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
                *entree(j, &exemples[i][j]);
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


*/
