# Physical AI con MuJoCo

Banco di prova sim2real per scegliere l'ordine di rimozione di oggetti da un
mucchio, raggiungere un target e misurare il disturbo provocato agli altri.
Il progetto cresce per de-idealizzazione: prima decisioni con stato esatto e
rimozione ideale, poi percezione stereo, robustezza, UR5 simulato e sistema reale.

## Avvio

Dalla cartella del progetto, nell'ambiente Conda gia usato:

```bash
conda activate mujoco-tirocinio
python main.py
```

Il banco di prova autonomo di OSSERVA si avvia con
`python -m physical_ai_mujoco.observe.main_test_osserva`. Chiede quale
configurazione e quale sorgente confrontare. Dopo il report permette di
riaprire ogni scena nel viewer MuJoCo/Gymnasium con le viste stereo affiancate.
Il menu consente di scegliere il disturbo delle rilevazioni stereo e un numero
fisso o casuale di oggetti per scena; il report registra i valori effettivi.
Il test visivo separato della stereocamera si avvia con
`python -m physical_ai_mujoco.sensors.test_stereo_camera`: apre il viewer e
mostra le due immagini monocromatiche affiancate. La distanza tra le camere
si imposta in `physical_ai_mujoco/sensors/capture.py` tramite
`STEREO_BASELINE_M`. I profili di disturbo sono in `configs/observe_tests/`
e i report vanno in `outputs/observe_tests/`.
Il banco include ora anche la sorgente `rgbd`, che usa depth renderizzata
da MuJoCo e riporta errore di dimensioni e F1 dei supporti. Il test congiunto
di stereo, depth e LiDAR e' `python -m physical_ai_mujoco.sensors.test_sensor_suite`;
il LiDAR ha anche `python -m physical_ai_mujoco.sensors.lidar.test_lidar`.
La modalita `fusion_learned` usa il detector della PFM-1 sulla camera B/N e il
LiDAR per posizione e geometria. Il menu mostra sempre il modello attivo; i
pesi sono salvati in `outputs/detector_weights/`. Il modello fornito con il
progetto e `pfm_1_seg_immersed_v2.pt`.
La storia, le responsabilita' e i limiti del modulo sono nel suo
[README](physical_ai_mujoco/sensors/README.md); i contratti e le prove di
calibrazione sono descritti anche in [docs/SENSORS.md](docs/SENSORS.md).

Il menu rende eseguibili 0A, 0B, 1A, 1B e 1C. In 1B PPO riceve il vettore
prodotto da camera B/N, detector e LiDAR; in 1C la stessa catena riceve
disturbi randomizzati a runtime. I modelli sensoriali usano il prefisso
`ppo_sensor_` per non essere confusi con quelli teacher di 1A.

I profili vivono in `configs/experiments/`: selezionano i componenti nel builder.
Anche i processi avviati dal menu ereditano la selezione. Le librerie scientifiche
non leggono nomi o numeri delle fasi. La fase 3 e assorbita da 1B e 2.
`main.py` si puo ancora avviare dal pulsante Run di VS Code.

Per una nuova installazione (Python >= 3.10):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test,train,video,detector]"
python main.py
```

Anche `python -m pip install -r requirements.txt` usa le stesse dipendenze.
`train` installa Stable-Baselines3; `video` aggiunge il supporto MP4;
`detector` installa Ultralytics per `fusion_learned` e per il training della
segmentazione.
L'installazione supportata e editable dalla copia del repository: configurazioni
e dataset rimangono nella radice, fuori dal pacchetto Python.

## Comandi disponibili

```bash
python -m scripts.run_phase_0a --seed 42
python -m scripts.run_phase_0b --objects 6 --headless --episodes 3
python -m scripts.inspect_episode --objects 6 --policy top --seed 0 --episodes 20 --quiet
```

Per allenare usare il menu oppure:

```bash
python scripts/allena.py --oggetti 6 --passi 25000 --paralleli 2
python scripts/quanto_margine.py --oggetti 3 --scene 2 --finisci-al-target
python scripts/verifica_fisica.py
python -m scripts.record_phase_0b --objects 3 --format gif
```

I precedenti comandi `scripts/*` sono mantenuti come punti di ingresso; il codice
operativo risiede nel package. I modelli esistenti in `outputs/modelli/` e i loro
file `_normalizzazione.pkl` restano utilizzabili con lo stesso numero di oggetti.
Un numero diverso viene ora segnalato con un errore esplicito.

## Organizzazione

| Package | Responsabilita |
|---|---|
| `contracts` | Observation, PrivilegedState, decisione ed esiti |
| `scene` | Dataset, descrizione e campionamento della scena |
| `simulation` | MuJoCo, sessione della scena, assestamento, pool, viewer |
| `sensors` | Stereo B/N, LiDAR, detector, tracking, disturbi e calibrazione |
| `observe` | Fusione, CAD PFM-1, scena/relazioni/incertezza e encoding PPO |
| `decide` | Classi decisionali casuale, altezza, target e PPO |
| `execute` | Rimozione ideale tramite API del simulatore |
| `task` | Disturbo, reward, crollo, successo e fine episodio |
| `envs` | Adapter Gymnasium; percorso storico mantenuto |
| `infrastructure` | Builder e adapter dei callback esistenti |
| `experiments` | Rollout, training, registrazione, monitor e metadati |
| `evaluation` | Ispezione, test OSSERVA, ricerca esaustiva e verifiche fisiche |
| `ui` | Menu e interazione con l'utente |

Ogni ruolo pubblica la sua API tramite `__init__.py`. DECIDE riceve dati, produce
un `ObjectDecision` e non legge il simulatore. Il runner passa la decisione
all'ambiente, che coordina gli altri componenti.

## Stato attuale e limiti

- Fasi 0A, 0B e 1A operative; 1B/1C sono collegate end-to-end ma richiedono
  ancora training e validazione statistica delle policy sensoriali.
- `state` conserva i 17 valori originali per oggetto; `stereo` restituisce due
  immagini RGB; `both` restituisce stato e immagini sincronizzati.
- La stereo simulata produce immagini B/N e maschere visibili dal renderer
  MuJoCo. `fusion_oracle` usa le etichette ideali per la baseline;
  `fusion_learned` usa un detector YOLO-seg addestrato sulla PFM-1 e sugli
  ostacoli sintetici. Manca ancora la validazione su dati reali.
- OSSERVA espone ora una `Observation` strutturata con scena, candidati di
  relazione e incertezza. Il vettore esatto a 17 valori per oggetto resta il
  canale privilegiato per il PPO teacher; `SensorObserver` combina maschere
  camera e punti LiDAR. `ObjectTracker` stabilizza gli ID, `CADMatcher` usa la
  STL PFM-1 per completare la geometria parziale e `ObservationEncoder`
  costruisce il vettore fisso consumato da PPO in 1B/1C.
- Massa e attrito restano nel ramo privilegiato del teacher; `Observation` non
  li espone. La stereo a centroidi stima le posizioni ma non ancora dimensioni
  e relazioni di supporto affidabili; il banco di prova rende visibile il limite.
- La rimozione resta ideale: il robot non e ancora presente.
- `SensorBundle`/`StereoObserver` restano soltanto come baseline deprecata per
  i confronti stereo-only e RGB-D; il percorso deployable usa
  `SynchronizedSensorPacket`.
- Con la configurazione attuale l'episodio termina al target, prosegue dopo un
  crollo e ha limite di 64 azioni. `is_success` richiede target rimosso senza
  crolli precedenti. Il disturbo considera la traslazione finale massima degli
  oggetti rimasti, escludendo il target; non misura il percorso o la rotazione.
- La validazione scientifica di 1A deve ancora definire split e gate. I seed
  della valutazione del trainer sono registrati, ma non dimostrano un test set
  indipendente dal training.

## Test e riproducibilita

```bash
python -m pytest -q
python scripts/verifica_fisica.py
python -m scripts.verifica_osserva
```

`verifica_osserva` esegue tutte le sorgenti sui profili `clean`, `fixed` e
`random`, continua dopo un errore isolato e salva il riepilogo in
`outputs/observe_tests/verifica_completa_*.json`.

I test grafici richiedono accesso al display macOS. Su Linux senza display il
backend offscreen va configurato in base alla macchina (EGL oppure OSMesa).
Un errore CoreGraphics di accesso al display non certifica un errore della stereo.

Ogni nuovo training salva anche `<modello>_run.json`: parametri, seed di
valutazione, versioni, commit, hash del codice, configurazioni, dataset e risultati.
La migrazione e documentata in [docs/RESTRUCTURING.md](docs/RESTRUCTURING.md).

## Documenti guida

- [Architettura](docs/ARCHITECTURE.md)
- [De-idealizzazione](docs/DEIDEALIZATION.md)
- [Implementazione OSSERVA](docs/OBSERVE_IMPLEMENTATION.md)
- [Stato completo/parziale/non implementato dei sensori](docs/SENSOR_COMPLETION_STATUS.md)
- [Banco di prova OSSERVA](docs/OBSERVE_TEST.md)
- [Registro delle modifiche](MODIFICHE.md)
- [Documentazione precedente, storica](docs/history/)

Gli output e le vecchie copie in `_to_delete/` rimangono locali, esclusi da Git.
