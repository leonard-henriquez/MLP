#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include "Neurone.h"

using namespace std;

void affiche_neurone(neurone N);
vector <vector<float> > recuperer_donnees(string);
void affiche_exemples(vector <vector <float> > exemples);
bool compatibilite_NE(neurone N, vector <vector <float> > exemples);

const float pas = 0.1;

int main(int argc, char *argv[])
{
    // déclare la variable qui contientra les exemples et les sortie attendues associées
    vector <vector<float> > exemples;

    // récupère les donnees du ficher donnees.txt et les stocke dans exemples
    string chemin;
    if (argc >= 2)
    chemin = argv[1];
    else
    chemin = "C:/donnees.txt";

    exemples = recuperer_donnees(chemin);

    affiche_exemples(exemples);

/*
    neurone N_1(2), N_2(1);

    float a;
    a = 1;
    N_1.entree(0, &a); // je donne aux deux entrées la valeur 1
    N_1.entree(1, &a);

    cout << "N1: " << N_1.sortie() << endl; // affiche 1

    N_2.lier(N_1);

    cout << "N2: " << N_2.sortie() << " " << N_2.entree(0) << endl; // affiche 1 1

    N_1.entree(0, 0); // je donne aux deux entrées la valeur 1
    N_1.entree(1, 0);

    cout << "N2: " << N_2.sortie() << " " << N_2.entree(0) << endl; // affiche 1 1

    if(compatibilite_NE(N_1_1, exemples) && compatibilite_NE (N_1_2, exemples))
    {
        N_1_1.apprentissage(exemples, pas);
    }
    else
    {
        cout << "Erreur: la sequence d'apprentissage n'est pas compatible avec le neurone";
    }
*/
    getchar();
    return 0;
}

vector <vector<float> > recuperer_donnees(string chemin_fichier_donnees)
{
    try
    {
        int nb_exemples = 0, nb_entrees = 0, i = 0;
        float exemple = 0;
        vector <vector <float> > exemples;
        //exemples.resize(0);
        string entete, ligne, mot;

        // ouvre le fichier de donnees en mode lecture
        ifstream fichier(chemin_fichier_donnees.c_str());
        if (!fichier)
            throw "Erreur: fichier inaccessible";


        getline (fichier, entete, ';');

        if (entete.find("exemples ") != string::npos)
            entete = entete.substr(int(entete.find("exemples "))+9);
        else
            throw "Erreur: pas d'entete dans le fichier\n";

        istringstream nombre1(entete);
        if (nombre1.operator >>( nb_exemples ).operator void*() == 0)
            throw "Erreur: l'entete n'est pas correct";

        getline(fichier, entete, ';');

        if (entete.find("entrees ") != string::npos)
            entete = entete.substr(int(entete.find("entrees "))+8);
        else
            throw "Erreur: pas d'entete dans le fichier\n";

        istringstream nombre2(entete);
        if (nombre2.operator >>( nb_entrees ).operator void*() == 0)
            throw "Erreur: l'entete n'est pas correct";

        //for (int i=0; i < nb_exemples;)
        while (getline(fichier, ligne, ';') && i < nb_exemples)
        {
            istringstream mots(ligne);
            //cout << ligne << endl;
            int j = 0;
            exemples.push_back(vector<float>());
            while (getline (mots, mot, '/') && j <= nb_entrees)
            {
                //cout << mot << endl;
                istringstream stream(mot);
                //cout << stream << endl;
                stream >> exemple;
                //if (!stream >> exemple)
                //    throw "Erreur: donnees non numeriques";
                //cout << exemple;
                exemples[i].push_back(0);
                exemples[i][j] = exemple;
                j++;
            }
            i++;
        }
    //exemples[1].push_back(0);
    return exemples;
    }

    catch (const char *message)
    {
        cerr << message << endl;
        return vector <vector <float> > (NULL);
    }


}

void affiche_exemples(vector <vector <float> > exemples)
{
    // affiche les exemples
    for (unsigned i = 0; i < exemples.size(); i++)
    {
        for (unsigned j = 0; j < exemples[i].size(); j++)
        {
            cout << exemples[i][j] << " ";
        }
        cout << "\n";
    }
    cout << "\n";
}

bool compatibilite_NE(neurone N, vector <vector <float> > exemples)
{
    for (unsigned i = 0; i < exemples.size(); i++)
    {
        if (N.nombre_entrees() != exemples[i].size() - 1)
        {
            return 0;
        }
    }
    return 1;
}
