# Stereo, RGB-D e LiDAR simulati

Il modulo `sensors/` non importa MuJoCo. Il simulatore espone render RGB,
segmentazione ideale, depth metrica e ray casting; i sensori li consumano
tramite l'API del simulatore. La segmentazione oracle e' una sorgente di test:
gli ID non sono riconosciuti dalle immagini.

## Componenti e contratti

- `rig.py`: `StereoRig` contiene intrinsics, baseline, extrinsics e dimensioni;
  proietta/retroproietta in ciascun occhio e puo essere costruito dalla
  descrizione di scena o da JSON di calibrazione.
- `capture.py`: `SimulatedStereoCamera` produce due immagini monocromatiche.
  Con `with_depth=True` aggiunge le due depth renderizzate. Per compatibilita le immagini
  sono rappresentate come HxWx3 `uint8` con canali identici; `gray_left` e
  `gray_right` forniscono la vista HxW. **La depth non e' triangolata dalle due
  immagini.**
- `detector.py`: `Detector.detect(frame)` non accetta un simulatore.
  `OracleDetector` e' l'adapter di test che usa la segmentazione MuJoCo.
- `noise.py`: disturbi configurabili per maschere, tipo, calibrazione e depth.
  Un `numpy.random.Generator` esplicito rende riproducibile ogni prova.
- `cloud.py`: retroproiezione dei pixel validi di una maschera a punti 3D;
  stima approssimata di centro, forma e dimensioni dall'inviluppo visibile.
  Occlusioni possono sottostimare le dimensioni; la confidence e' un
  indicatore empirico, non una probabilita calibrata.
- `bundle.py`: `BundleBuilder` combina frame, detector e disturbi in
  `SensorBundle`; e una baseline deprecata per confronti stereo/RGB-D.
- `source.py`: crea il packet stereo+LiDAR sincronizzato e applica a runtime
  disturbi riproducibili alle misure, non alle etichette.
- `tracking.py`: mantiene ID di istanza stabili con classe, IoU e centroide.
- `lidar/`: scansione 2D o 3D sparse tramite `simulator.raycast`, frame con
  un elemento per raggio, `valid` per i ritorni e `NaN`/`inf` per i no-hit.
  La serializzazione JSON usa `null` per i no-hit.
- `calibration/`: perturbazione delle extrinsics e stima rigida
  `camera_from_lidar` da **corrispondenze 3D gia associate**. La stima non
  risolve da sola il problema di trovare le corrispondenze.

`StereoObserver` riceve `reconstruction_mode="stereo"` oppure `"depth"` all’avvio.
La prima modalità usa la triangolazione; la seconda richiede entrambe le depth.
La scelta non cambia implicitamente in base ai dati del singolo frame. Le osservazioni restano nello stesso
contratto pubblico `Observation`; `SensorEvidence` conserva la nuvola di punti.
Il percorso deployable usa invece `SensorObserver` e
`SynchronizedSensorPacket`. La PFM-1 puo essere allineata alla STL per
recuperare l'ingombro completo da punti parziali; i fit incompatibili vengono
rifiutati.

## Avvio dei test visivi e del benchmark

```bash
python -m physical_ai_mujoco.sensors.test_stereo_camera --seed 42
python -m physical_ai_mujoco.sensors.test_sensor_suite --seed 42 --objects 3
python -m physical_ai_mujoco.sensors.lidar.test_lidar --seed 42 --objects 3
python -m physical_ai_mujoco.observe.main_test_osserva --mode rgbd --scenes 3 --objects 10 --headless
python -m physical_ai_mujoco.evaluation.sensor_calibration --scenes 3 --objects 10
```

I test visivi aprono il viewer MuJoCo/Gymnasium e una finestra OpenCV; `q` o
Esc chiudono. `--seconds N` chiude dopo N secondi. `--headless` salva il PNG
senza aprire finestre. I programmi in `sensors/` possono essere eseguiti
anche passando direttamente il percorso del file.

I profili `clean`, `fixed` e `random` in `configs/observe_tests/` definiscono
anche `rgbd.drop_probability`, `depth_sigma`, `translation_sigma` e
`rotation_sigma_deg`. La curva di calibrazione esegue le stesse scene e seed
per ogni combinazione di errori e salva recall, errori di posizione/dimensione
e F1 supporti. `compare_lidar_depth` proietta gli hit LiDAR nel frame camera
con calibrazione nota e misura il residuo della depth senza usare ID.

La depth del renderer MuJoCo nella prova locale ha restituito valori metrici,
ma il driver OpenGL ha segnalato precisione limitata del depth buffer
(`ARB_clip_control unavailable`). Non considerare un singolo F1 o residuo come
calibrazione di un sensore reale. La scansione LiDAR e' geometrica; il modello
di rumore attuale e' sintetico e va tarato su un sensore concreto prima di
interpretarlo come prestazione sim2real.
