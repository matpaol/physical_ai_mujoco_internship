# OSSERVA: implementazione corrente

La API pubblica e `Observer.observe(source, task) -> Observation`. L'osservazione
contiene `SceneState`, `PhysicalRelationState` e `UncertaintyState`. Gli input
intermedi (`PerceptualState` e `SensorEvidence`) restano interni a OSSERVA.
Il ramo `ExactObserver.privileged_state(...)` produce separatamente il vettore
storico a 17 valori per oggetto usato dal PPO teacher della fase 1A.

## Sorgenti

| Sorgente | Ingresso | Stato |
|---|---|---|
| `ExactObserver` | stato MuJoCo attraverso la API `Simulator` | operativo in 0B/1A |
| `DegradedObserver` | stato MuJoCo perturbato con seed riproducibile | componente iniziale per 1C |
| `StereoObserver` | `SensorBundle` | baseline deprecata stereo-only/RGB-D |
| `SensorObserver` | packet sincronizzato, maschere B/N tracciate e punti LiDAR | percorso operativo del laboratorio e delle fasi 1B/1C |

`StereoObserver` puo ricevere una coppia da camera simulata o reale perche non
dipende dal backend. Per il banco di prova, `SimulatedStereoCamera` produce
immagini B/N e calibrazione. `OracleDetector` fornisce la baseline ideale;
`LearnedDetector` carica un modello YOLO-seg e identifica ostacoli e PFM-1
senza leggere pose o ID MuJoCo. `SensorObserver` proietta i punti LiDAR nelle
maschere per localizzare e stimare la geometria. `ObjectTracker` rende
persistenti gli ID; `CADMatcher` completa l'ingombro PFM-1 dalla STL quando il
fit e valido; `ObservationEncoder` assegna slot stabili e produce il vettore
fisso per PPO. Restano la sorgente reale/ROS 2 e la validazione su dati reali.

Il banco di prova autonomo e descritto in [OBSERVE_TEST.md](OBSERVE_TEST.md).

## Relazioni e incertezza

`GeometricRelationEstimator` produce **candidati** di supporto da posizione e
ingombro stimati. Lo `relation_score` misura solo la vicinanza verticale nella
regola geometrica ed e non calibrato. Il `dependency_graph` e derivato dalle
relazioni dirette e non contiene ordini di rimozione. Questa e una baseline:
non implementa il predittore di dinamica locale di Li et al. L'adattamento del
metodo degli autori richiede point cloud segmentate, maschere e dati di
interazione; il loro codice usa Isaac Sim e un ambiente software diverso.

`ExactUncertaintyProvider` produce qualita piena. Il provider degradato tiene
memoria degli ID gia osservati ma non visibili al passo corrente, senza
inventare una posa. Il provider stereo lascia le qualita numeriche ignote e
segnala spazio non osservato. Non e un belief calibrato ne una riproduzione del
modello CNABU di Marques et al. Il codice pubblicato dagli autori dichiara
ancora incompleta la pubblicazione di training e valutazione; per un adapter
fedele serviranno un dataset e metriche di calibrazione del nostro dominio.

## Invarianti

- `ObservationBuilder` controlla ID, frame e timestamp dei tre prodotti.
- Massa, attrito e velocita stanno solo in `PrivilegedState`.
- DECIDE riceve `Observation` e seleziona un `object_id`; l'environment traduce
  l'ID nell'indice richiesto da Gymnasium.
- I modelli PPO 1A restano teacher su `PrivilegedState`. 1B/1C usano
  `ObservationEncoder`; richiedono modelli nuovi e non caricano quelli 1A.
- Il provider viene scelto all'avvio. Degrado sintetico e stereo sono sorgenti
  alternative, non degradazioni concatenate automaticamente.

Riferimenti implementativi: [Li et al.](https://github.com/lyttttt3333/Broadcast_Support_Relation),
[Marques et al.](https://github.com/NilsDengler/manipulation_enhanced_map_prediction).
