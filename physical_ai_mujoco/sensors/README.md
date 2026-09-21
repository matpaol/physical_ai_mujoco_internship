# Modulo sensori: stereo, RGB-D e LiDAR

Questo package e' il confine tra il mondo fisico/simulato e i dati di ingresso
di OSSERVA. Produce **misure e rilevazioni**, non decide quali oggetti rimuovere.
MuJoCo rimane in `simulation/`: qui si chiamano metodi pubblici del simulatore,
senza importare il motore fisico. In futuro gli adapter reali potranno produrre
gli stessi contratti senza cambiare OSSERVA.

## Perche' e' nato

All'inizio OSSERVA riceveva lo stato esatto di MuJoCo. Era utile per chiudere
il ciclo del progetto, ma non mostrava cosa accade quando un oggetto e'
occluso, una detection manca o la posizione stimata e' imprecisa. La prima
camera stereo simulata ha quindi aggiunto due viste **monocromatiche**. Il
simulatore fornisce le immagini e, separatamente, maschere e ID ideali usati
come etichette di test. Questa separazione impedisce di confondere una camera
con un detector: la camera non conosce il target e non riconosce oggetti.

Nel primo benchmark stereo, su tre scene con dieci oggetti, il recall medio
era 0,40 e il F1 dei supporti non era misurabile. Alcune coppie di maschere
venivano scartate per disallineamento verticale dei centroidi; inoltre la
triangolazione produceva la posizione, ma non forma e dimensioni necessarie
allo stimatore geometrico. Per poter misurare questi problemi separatamente
sono stati aggiunti rig, detector sostituibile, depth, nuvole di punti,
disturbi, LiDAR e calibrazione.

## Flussi dei dati

```text
MuJoCo Simulator API
  render_stereo() ──> SimulatedStereoCamera ──> StereoFrame (grigio)
  render_depth_stereo() ──> SimulatedStereoCamera(with_depth=True) ──> StereoFrame (+depth)
  render_instance_masks() ──> OracleDetector ─> StereoDetections (solo test)
  raycast() ──> lidar.scan() ──────────────────> LidarFrame

StereoFrame + StereoDetections + disturbi ──> BundleBuilder ──> SensorBundle
SensorBundle senza depth ──> StereoObserver ──> triangolazione
SensorBundle con depth ────> StereoObserver(mode="depth") ──> point cloud
                                                └─────────> Observation

Percorso operativo di fusione:

SimulatedSensorSource ──> SynchronizedSensorPacket (stereo + LiDAR + calibrazione)
                                  │
Oracle/Learned Detector ──────────┤
             ObjectTracker ───────┤
                                  ▼
                         SensorObserver
                 mask 2D + punti LiDAR proiettati
                                  ▼
                              Observation
```

`SimulatedSensorSource` applica i disturbi immagine/LiDAR a runtime prima della
fusione. `ObjectTracker` associa classe, IoU e centroide e mantiene ID stabili
durante brevi occlusioni. `OracleDetector` resta soltanto l'adapter di test.


## File e responsabilita'

| File | Cosa implementa | Perche' e' separato |
|---|---|---|
| `rig.py` | `StereoRig`: K, baseline, posa, proiezione/retroproiezione e caricamento calibrazione JSON | Geometria uguale in simulazione e su camera calibrata |
| `capture.py` | `SimulatedStereoCamera`, `StereoFrame`, baseline di default | Immagini/depth grezze senza ID o maschere |
| `detector.py` | API `Detector`, rilevazioni e `OracleDetector` | Isola gli ID ideali di MuJoCo dal futuro detector |
| `noise.py` | Drop, erosione/dilatazione, fusioni, falsi positivi, confusione di tipo, errore di depth e calibrazione | Disturbi indipendenti e riproducibili con seed |
| `cloud.py` | Pixel depth + maschera → punti 3D; stima approssimata della geometria | OSSERVA non riceve shape/size vere dal simulatore |
| `bundle.py` | Composizione e validazione delle maschere nel `SensorBundle` | Punto unico di assemblaggio, detector sostituibile |
| `source.py` | `SensorSource` e acquisizione simulata stereo+LiDAR | Confine sostituibile con registrazioni e ROS 2 |
| `tracking.py` | associazione temporale e ID persistenti | Il detector non definisce l'identita tra frame |
| `lidar/` | Pattern di scansione, frame, disturbo, test visivo e API calibrazione | Il LiDAR e' un sensore autonomo, non un campo nascosto della camera |
| `calibration/` | Perturbazione delle pose e stima rigida da punti corrispondenti | Separa errore di misura e stima della trasformazione |

I contratti grezzi (`StereoFrame`, `LidarFrame`, `SensorCalibration` e
`SynchronizedSensorPacket`) sono in `contracts/sensing.py`; i vecchi percorsi
di import restano disponibili. Il `SensorBundle` precedente resta come
**baseline deprecata** esclusivamente per i confronti `stereo` e `rgbd`;
`BundleBuilder` emette una `DeprecationWarning`. Conserva i campi
`rgb_left/right` HxWx3 richiesti dal percorso precedente: i tre canali hanno
la **stessa intensita'**. Le proprieta' `gray_left/right` espongono HxW.
`depth_left/right` e `detection_source` sono aggiunte opzionali. `StereoObserver` seleziona esplicitamente `stereo` o `depth` alla costruzione;
il contenuto del bundle non cambia la modalità a runtime. **La depth renderizzata da MuJoCo
non e' calcolata dalle due viste stereo**: confrontare i due percorsi serve a
isolare l'errore della triangolazione.

Il LiDAR non restituisce ID di oggetti. `LidarFrame` conserva un campione per
raggio: `valid=False`, range infinito e punto NaN significano nessun ritorno;
`to_dict()` li serializza come `null`. La calibrazione LiDAR-camera usa punti
3D gia' **corrispondenti** e restituisce `camera_from_lidar`. Trovare tali
corrispondenze da due misure grezze e' un problema ulteriore, non risolto da
quella funzione. `compare_lidar_depth` confronta invece raggi e depth usando
una calibrazione nota, per diagnosticare scarti della catena sensoriale.

Per la PFM-1, i punti LiDAR associati alla maschera possono essere allineati
alla STL tramite `CADMatcher`: inizializzazione PCA, ICP trimmed e rifiuto del
fit se l'errore normalizzato e eccessivo. Un fit accettato restituisce
l'ingombro nominale completo; un fit rifiutato conserva la stima parziale.

## Avvio e test

```bash
python -m physical_ai_mujoco.sensors.test_stereo_camera --seed 42
python -m physical_ai_mujoco.sensors.test_sensor_suite --seed 42 --objects 3
python -m physical_ai_mujoco.sensors.lidar.test_lidar --seed 42 --objects 3
python -m physical_ai_mujoco.observe.main_test_osserva --mode rgbd --scenes 3 --objects 10 --headless
python -m physical_ai_mujoco.observe.main_test_osserva --mode fusion_oracle --scenes 3 --objects 3 --headless
python -m physical_ai_mujoco.evaluation.sensor_calibration --scenes 3 --objects 10
python -m pytest tests/test_sensors_extended.py tests/test_simulated_stereo.py
```

I tre file di test di sensori e OSSERVA possono anche essere avviati con
`python /percorso/del/progetto/tests/test_sensors_extended.py` (analogamente
`test_simulated_stereo.py` e `test_observe_pipeline.py`). In questo caso lo
script mette la propria radice di progetto davanti alle altre installazioni
Python e chiama pytest. Eseguire un test con `python` senza questo passaggio
potrebbe importare un'altra copia installata di `physical_ai_mujoco`.

I test visivi aprono il viewer MuJoCo/Gymnasium e mostrano immagini e
scansione. `q` o Esc chiudono; `--seconds N` limita la durata e `--headless`
salva soltanto un PNG. Il test LiDAR risiede nel package `lidar/`.

I test automatici verificano geometria del rig, allineamento depth, maschere,
sostituibilita' del detector, disturbi e seed, scansioni con hit/no-hit,
calibrazione, serializzazione JSON e integrazione con OSSERVA. Il benchmark
registra posizione, dimensioni, forma, recall e supporti sulle stesse scene.
I profili in `configs/observe_tests/` scelgono `clean`, `fixed` o `random`.

## Cosa significano i risultati

Nelle tre scene da dieci oggetti usate per diagnosticare il vecchio benchmark,
il percorso RGB-D pulito ha ottenuto recall 0,967, errore posizione medio
0,019 m e F1 supporti 0,391. Il grafo e' quindi **calcolabile**, ma non ancora
accurato: una nuova sorgente sensoriale non corregge da sola il criterio del
`GeometricRelationEstimator`. La stima della forma da una superficie
parzialmente visibile puo' sbagliare; `confidence` e' un indicatore empirico,
non una probabilita' calibrata. Il renderer locale ha anche avvisato che la
precisione del depth buffer OpenGL e' limitata (`ARB_clip_control unavailable`).

La segmentazione oracle resta un vantaggio ideale della simulazione. Nessun
risultato qui dimostra che un detector reale identifichi il target, ne' che
il rumore sintetico riproduca automaticamente il comportamento di una camera
o di un LiDAR specifici. Per il sim2real si sostituiscono gli adapter e si
calibrano i disturbi su misure reali, mantenendo gli stessi contratti.
