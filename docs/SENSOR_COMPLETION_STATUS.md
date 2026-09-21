# Stato esplicito della pipeline sensoriale

Aggiornamento: 2026-09-21 (visore); resto del documento al 2026-09-20. Questo documento distingue codice completato,
implementazioni ancora da validare e lavoro non eseguito. Non sostituisce i
report numerici del benchmark.

## Completato e collegato al runtime

- `ObjectTracker`: ID persistenti per classe, IoU e distanza del centroide;
  tollera brevi occlusioni ed e usato da `LearnedDetector`.
- `ObservationEncoder`: slot stabili, feature oggetto, matrice delle relazioni,
  flag di spazio ignoto e action mask a dimensione fissa.
- Environment Gymnasium sensoriale: `obs_mode="sensor"`, acquisizione tramite
  `SensorSource`, encoding PPO e associazione controllata fra track e corpo
  simulato soltanto nell'adapter di esecuzione.
- Contratto `Observer`: tutte le implementazioni ricevono `(source,
  TaskContext)`; non coesistono piu firme incompatibili.
- Disturbi runtime: fotometria B/N e rumore/dropout LiDAR sono applicati da
  `SimulatedSensorSource`, con seed riproducibile. I profili OSSERVA includono
  parametri distinti per la fusione.
- Target configurabile: `LidarGeometryEstimator` non contiene piu la stringa
  magica `pfm_1_target`.
- Metriche: rapporti delle tre dimensioni e del volume stimato rispetto al
  vero, sia globali sia per target e fascia di esposizione.
- Fasi 1B e 1C: selezionabili dal menu; i modelli PPO sensoriali hanno nomi
  separati dai teacher 1A. 1C applica i disturbi a runtime.

## Implementato, ma ancora da validare scientificamente

- `CADMatcher`: carica la STL binaria, applica scala metrica, inizializza con
  PCA, esegue ICP trimmed su osservazioni parziali e rifiuta fit con errore
  normalizzato eccessivo. Se il fit e rifiutato, la pipeline ricade sulla
  geometria visibile senza inventare una posa completa.
- Il CAD matcher passa prove sintetiche e usa correttamente le dimensioni della
  STL PFM-1. Non e ancora validato su point cloud del LiDAR reale, su tutte le
  percentuali di interramento o su tutte le simmetrie della mina.
- 1B/1C sono eseguibili e pronte per il training, ma non e stata addestrata ne
  validata una nuova policy PPO sensoriale in questo aggiornamento.
- I disturbi 1C sono operativi ma i loro intervalli non sono ancora calibrati
  su registrazioni dei sensori reali.

## Mantenuto intenzionalmente, quindi non eliminato

- `SensorBundle`, `BundleBuilder` e `StereoObserver` non sono stati rimossi:
  servono ai confronti storici `stereo` e `rgbd`. Sono marcati come baseline
  deprecata e `BundleBuilder` emette `DeprecationWarning`. Il runtime 1B/1C usa
  invece `SynchronizedSensorPacket` e non dipende da quel percorso.

## Non implementato in questo aggiornamento

- FPFH + RANSAC globale tramite Open3D: il CAD matcher usa PCA + ICP trimmed,
  evitando una nuova dipendenza. Va aggiunto soltanto se i dati reali mostrano
  che le inizializzazioni correnti non coprono pose/occlusioni sufficienti.
- `RealSensorSource`, ROS 2 e driver dell'hardware reale.
- Calibrazione reale stereo-camera/LiDAR e caratterizzazione sperimentale del
  rumore.
- Miglioramento del `GeometricRelationEstimator`: il grafo resta una baseline
  geometrica non calibrata.
- Nuovo training di PPO: in accordo con il workflow, resta da eseguire sul PC
  dell'utente e da riportare nei report.
- Training del detector: dal 2026-09-21 esiste il modulo `vision_training`
  (ricette, domain randomization, scheda del modello, valutazione per % di
  target visibile), verificato sul Mac con un training di prova. Il modello
  addestrato sul dataset randomizzato `sim_dr_v1` non esiste ancora.

## Verifiche eseguite

- 131 test automatici passati nella copia Desktop, inclusi rendering MuJoCo.
- Smoke test `fusion_oracle` e `fusion_learned` completati con invarianti
  validi.
- Smoke test fase 1B: vettore e observation space `(169,)`, sei slot, azione
  sensoriale associata e rimossa correttamente.
- Smoke test fase 1C: disturbi immagine e LiDAR presenti nella sorgente runtime.

Uno smoke test dimostra che la catena gira; non misura recall, robustezza o
sim2real. Per quelle conclusioni servono benchmark su molte scene e dati reali.
