# Tesi Argentieri Luca

La tesi consiste nello studio di modifiche architetturali di modelli di tipo ESN.

Tutti i modelli hanno il reservoir della stessa dimensione cambia come viene inizializzato il reservoir.

Per chiarezza definisco le varie modifiche in questo modo:
- ESN1: ESN di tipo standard
- ESN2: Il reservoir è composto da N sub-reservoir ognuno con dimensione uguale a UNITS / N, non comunicanti tra di loro
- ESN3: Il reservoir è composto da N sub-reservoir ognuno con dimensione uguale a UNITS / N, tra di loro connessi
- ESN4: Il reservoir è composto da N sub-reservoir ognuno con dimensione variable, tra di loro connessi

## Risultati ottenuti

I seguenti risultati sono stati ottenuti tramite il dataset 'character trajectories', ricostruendo il modello 5 volte con gli stessi iperparametri.
ESN1 è il modello di riferimento.

| UNITS | ESN 1       | ESN 2       | ESN 3       | ESN 4       |
|-------|-------------|-------------|-------------|-------------|
| 25    | 71.28±3.35% | 76.38±1.59% | 74.40±4.28% | 83.11±4.60% |
| 50    | 77.66±2.66% | 82.98±2.94% | 84.83±1.85% | 87.80±4.68% |
| 75    | 82.74±3.63% | 85.58±2.60% | 80.75±2.79% | 90.65±3.31% |
| 100   | 84.46±2.36% | 89.47±0.83% | 92.06±1.13% | 90.95±2.03% |
| 125   | 85.40±2.47% | 87.66±3.43% | 93.91±1.95% | 93.52±0.71% |
| 150   | 85.81±1.56% | 90.03±2.43% | 93.75±1.70% | 94.67±0.69% |
| 175   | 86.85±2.49% | 83.31±4.26% | 93.43±2.91% | 93.38±2.47% |
| 200   | 87.49±2.33% | 90.74±2.49% | 94.28±0.74% | 95.18±1.33% |

A quante unità si arriva a un accuratezza X

| Accuratezza | ESN 1         | ESN 2 | ESN 3 | ESN 4 |
|-------------|---------------|-------|-------|-------|
| 85          | 125           | 75    | 50    | 50    |
| 90          | Non raggiunta | 200   | 100   | 75    |

Grafici e immagini dei modelli si trovano nella cartella ```images```

## Implementazione

Tutti modelli sono formati da due modelli di tipo keras.Sequential:
- un reservoir costituito da un livello di masking e da un livello ESN definito in ```lib/esn.py```.
- il readout costituito da un solo livello Denso.

I modelli vengono definiti all'interno del file ```lib/models.py```, ed ereditano dalla classe "ESNInteface" i metodi di call, fit e evaluate,
le differenze tra i modelli sono le funzioni d'inizializzazione del reservoir nella funzione di init.
Le inizializzazioni dei kernel e dei recurrent kernel vengono definite in ```lib/initializers.py```. 

Per i kernel esistono 2 inizializzatori:
- ```Kernel``` per generare kernel "standard"
- ```SplitKernel``` genera kernel tali da fornire la feature di input soltanto a un sottoinsieme di unita nel recurrent kernel. 
Per ottenere ciò si creano N matrici della giusta dimensione e si uniscono tramite la funzione tf.linalg.LinearOperatorBlockDiag
che andrà a generare una nuova matrice avendo sulla diagonale le matrice precendentemente generate e zero da le altre parti.

Per i recurrent kernel :
- ```RecurrentFullConnected``` usato per ESN1
- ```RecurrentKernel``` usato per generare i kernel ricorrenti con i N sub-reservoir, andando a generare NxN matrici ognuna con le proprietà necessarie,
per poi andarle a riunificare in una sola matrice.
- ```Type2, Type3, Type4``` Ereditano da ```RecurrentKernel``` e forniscono i parametri necessari per poter creare i reservoir per i modelli ESN2, ESN3 e ESN4 