import numpy as np
from iteration_utilities import random_permutation
from random import seed
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == '__main__':
    # seedy
    #seed(22)  # dla iteration_utilities
    #np.random.seed(22)
    #nMiast = 50
    #nOsob = 10 * nMiast
    #pKrzyz = 0.9
    #pMut = 0.01
    #maxIter = 2000
    #maxBrakZmian = 300
    #nazwa_plota = 'plot5091.png'
    print("Algorytm w trakcie działania będzie pokazywał następujące wartości:")
    print("Nr. iteracji | nr. iteracji bez zmian w wyniku | odległość")
    nMiast = int(input("Wprowadź ilość miast: "))
    nOsob  = int(input("Wprowadź ilość osobników jednej generacji: "))
    pKrzyz = float(input("Wprowadź prawdopodobieństwo krzyżowania (np. dla 50% wprowadź 0.5): "))
    pMut  = float(input("Wprowadź prawdopodobieństwo mutacji (np. dla 1% wprowadź 0.01): "))
    maxIter = int(input("Wprowadź maksymalną ilość iteracji algorytmu: "))
    maxBrakZmian  = int(input("Wprowadź ilość iteracji bez zmian w wyniku po której algorytm się zatrzyma: "))
    nazwa_plota = input("Wprowadź nazwę pliku do którego zostaną zapisane wykresy (np. plot.png): ")

    osobniki = np.zeros((nOsob, nMiast))
    miasta = np.random.randint(0, 500, (nMiast, 2))
    odleglosci = np.zeros((nOsob, 1))
    distMinIter = np.zeros(maxIter)
    fig, ax = plt.subplots(3, 1, figsize=(10,10))
    fig.tight_layout(pad=3.0)

    # początkowe permutacje dla osobników
    for i in range(nOsob):
        osobniki[i] = random_permutation(list(range(nMiast)))

    # początkowa permutacja
    osobnikPocz = osobniki[0].astype(int)
    ax[0].scatter(miasta[:, 0], miasta[:, 1], color="red")
    linia = mpl.lines.Line2D(miasta[osobnikPocz][:, 0], miasta[osobnikPocz][:, 1], linewidth=0.5)
    ax[0].add_line(linia)

    cBrakZmian = 0
    sMinOdleglosc = 0
    nIter = 1

    # główna pętla - jeżeli brak zmian 50 iteracji pod rząd - koniec
    while (cBrakZmian < maxBrakZmian) and (nIter < maxIter):
        print(nIter,cBrakZmian, sMinOdleglosc)
        # obliczanie funkcji przystosowania (suma odległości przebytej)
        odleglosci = np.zeros((nOsob, 1))
        for i in range(nOsob):
            for j in range(1, nMiast):
                x = np.abs(miasta[int(osobniki[i][j])][0]-miasta[int(osobniki[i][j-1])][0])
                y = np.abs(miasta[int(osobniki[i][j])][1]-miasta[int(osobniki[i][j-1])][1])
                odleglosci[i] += np.sqrt(np.square(x) + np.square(y))

        # counter do warunku zakończenia
        if sMinOdleglosc == np.min(odleglosci):
            cBrakZmian += 1
        else:
            cBrakZmian = 0

        # zachowanie najlepszego
        osobniki[0] = osobniki[np.argmin(odleglosci)]
        osobniki[1] = osobniki[0]  # żeby było parzyście potem z krzyżowania

        # losowanie najlepszych - metoda ruletki
        prawdopodobienstwa = odleglosci
        prawdopodobienstwa = np.max(prawdopodobienstwa) - prawdopodobienstwa  # bo te najmniejsze mają być najbardziej prawd.
        prawdopodobienstwa = prawdopodobienstwa / np.sum(prawdopodobienstwa)
        prawdopodobienstwa = np.transpose(prawdopodobienstwa)
        prawdopodobienstwa = np.square(prawdopodobienstwa)
        prawdopodobienstwa = prawdopodobienstwa / np.sum(prawdopodobienstwa)

        count = 2
        while count < nOsob:
            # wylosowanie 2 rodziców
            wylosowane = np.random.choice(np.array(range(nOsob)), 2, p=prawdopodobienstwa[0])  # np. [492 953]
            if pKrzyz < np.random.rand():
                continue
            # wylosowanie locus
            sLocus = np.random.randint(1, nMiast-1)
            kLocus = np.random.randint(1, nMiast-1)
            if sLocus == kLocus:
                sLocus -= 0
            if sLocus > kLocus:
                (sLocus, kLocus) = (kLocus, sLocus)

            # tworzenie dzieci
            p1 = osobniki[wylosowane[0]]
            p2 = osobniki[wylosowane[1]]

            d1 = p2.copy()
            i = 0
            j = 0
            # tworzenie + funkcja legalizacji
            for m in range(len(p1)):
                if i == sLocus:
                    d1[sLocus:kLocus] = p1[sLocus:kLocus]
                    i += kLocus - sLocus
                elif not p2[j] in p1[sLocus:kLocus]:
                    d1[i] = p2[j]
                    i += 1
                    j += 1
                else:
                    j += 1

            d2 = p1.copy()
            i = 0
            j = 0
            # tworzenie + funkcja legalizacji
            for m in range(len(p1)):
                if i == sLocus:
                    d2[sLocus:kLocus] = p2[sLocus:kLocus]
                    i += kLocus - sLocus
                elif not p1[j] in p2[sLocus:kLocus]:
                    d2[i] = p1[j]
                    i += 1
                    j += 1
                else:
                    j += 1

            # mutacja
            for i in range(nMiast):
                if pMut > np.random.rand():
                    g1 = np.random.randint(0, nMiast - 1)
                    g2 = np.random.randint(0, nMiast - 1)
                    temp = d1[g2]
                    d1[g2] = d1[g1]
                    d1[g1] = temp
                if pMut > np.random.rand():
                    g1 = np.random.randint(0, nMiast - 1)
                    g2 = np.random.randint(0, nMiast - 1)
                    temp = d2[g2]
                    d2[g2] = d2[g1]
                    d2[g1] = temp

            osobniki[count] = d1
            osobniki[count+1] = d2
            count += 2

        sMinOdleglosc = np.min(odleglosci)
        distMinIter[nIter] = sMinOdleglosc
        nIter += 1

    # wizualizacja
    osobnikAlfa = osobniki[0].astype(int)
    ax[1].scatter(miasta[:, 0], miasta[:, 1], color="red")
    linia = mpl.lines.Line2D(miasta[osobnikAlfa][:,0], miasta[osobnikAlfa][:,1], linewidth=0.5)
    ax[1].add_line(linia)
    distMinIter[0] = distMinIter[1]
    ax[2].plot(distMinIter)

    ax[0].set_title("Początkowa przykładowa droga (całkowicie losowa)")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[1].set_title("Ostateczna najlepsza droga")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")
    ax[2].set_title("Zmniejszanie się drogi przebytej przez komiwojażera")
    ax[2].set_xlabel("Pokolenie")
    ax[2].set_ylabel("Droga pokonana")
    ax[2].set_xlim([0,nIter])
    ax[2].set_ylim([0,np.max(distMinIter)])
    plt.savefig(nazwa_plota)
