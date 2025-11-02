#include "c_primes.h"

std::vector<int> c_primes(int nb_primes) {
    if (nb_primes > 1000) {
        nb_primes = 1000;
    }

    int p[1000];           // tableau statique pour stocker les nombres premiers
    int len_p = 0;         // nombre de premiers trouvés
    int n = 2;             // nombre à tester

    while (len_p < nb_primes) {
        bool is_prime = true;

        // Vérifie si n est divisible par un des nombres premiers précédents
        for (int i = 0; i < len_p; ++i) {
            if (n % p[i] == 0) {
                is_prime = false;
                break;
            }
        }

        // Si aucun diviseur trouvé, c’est un nombre premier
        if (is_prime) {
            p[len_p] = n;
            ++len_p;
        }

        ++n;
    }

    // Copier dans un vecteur pour le retour
    std::vector<int> result(p, p + len_p);
    return result;
}
