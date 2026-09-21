# Banco di prova autonomo di OSSERVA

`python -m physical_ai_mujoco.observe.main_test_osserva` avvia un menu
separato dal ciclo operativo. Sceglie una configurazione da
`configs/observe_tests/` e una sorgente: Exact, Degraded, Stereo, RGB-D,
fusione camera-LiDAR oracle o fusione camera-LiDAR learned. Si puo scegliere un numero
fisso di oggetti oppure `R`, con un intervallo minimo/massimo: in questo caso
il numero viene estratto separatamente per ogni scena. Dopo il report apre subito la prima scena nel
viewer MuJoCo/Gymnasium con le due viste stereo; le finestre restano aperte
finche si preme `q` o Esc nella finestra stereo. Dopo si puo scegliere un'altra
scena oppure uscire. Non chiama DECIDE o ESEGUE.
Il test visivo della camera ha un avvio indipendente
nel package `sensors/`.
Per esecuzioni riproducibili senza domande:

```bash
python -m physical_ai_mujoco.observe.main_test_osserva --config configs/observe_tests/fixed.json --mode all --scenes 10 --seed 42
python -m physical_ai_mujoco.sensors.test_stereo_camera --seed 42
python -m physical_ai_mujoco.observe.main_test_osserva --config configs/observe_tests/random.json --mode all --stereo-profile fixed --objects random --min-objects 2 --max-objects 6 --scenes 10 --headless
python -m physical_ai_mujoco.observe.main_test_osserva benchmark --config configs/observe_tests/random.json --mode fusion_learned --scenes 1 --objects 3 --headless
python -m scripts.verifica_osserva
python -m physical_ai_mujoco.sensors.test_stereo_camera --noise-px 1 --drop-probability 0.1
```

Sono supportati anche i percorsi diretti, per esempio dal pulsante Run
dell'editor: `python physical_ai_mujoco/observe/main_test_osserva.py` e
`python physical_ai_mujoco/sensors/test_stereo_camera.py`.

Il comando `main_test_osserva` produce metriche e grafi, poi apre la prima scena per la
revisione visiva se avviato in un terminale interattivo. Si puo riaprire in seguito
un report gia salvato con
`python physical_ai_mujoco/observe/main_test_osserva.py --review-report outputs/observe_tests/NOME.json --scene 1`.
La revisione salva `scene_000_stereo_preview.png` accanto ai grafi del report.
Gli ID sulle maschere e `[T]` indicano la verita ideale di MuJoCo per
l'ispezione, non un riconoscimento visivo effettuato da OSSERVA;
`--review-seconds N` limita la durata per prove automatiche. `--headless`
disattiva la revisione. Il comando `test_stereo_camera` apre
il viewer e una finestra con le immagini sinistra/destra monocromatiche e
maschere sovrapposte, e salva un PNG. Si chiude con `q` o Esc nella finestra
delle camere; `--seconds N` imposta una durata automatica. La baseline fisica
e la calibrazione usano entrambe `STEREO_BASELINE_M` definita in
`physical_ai_mujoco/sensors/capture.py`.

## Detector appreso e pesi

La modalita `fusion_learned` richiede l'extra opzionale `detector`:

```bash
python -m pip install -e ".[detector]"
```

I modelli esportati sono in `outputs/detector_weights/`. Il menu mostra la
cartella e il modello attivo all'avvio. La selezione segue questo ordine:
percorso passato con `--weights`, `pfm_1_seg.pt`, modello versionato
`pfm_1_seg*.pt` modificato piu di recente. Il modello attuale e
`pfm_1_seg_immersed_v2.pt`. Il training interattivo stampa sempre il percorso
assoluto del file prodotto.

Gli errori operativi vengono mostrati senza traceback. Un errore interno
imprevisto produce un messaggio breve e salva il traceback completo in
`outputs/logs/osserva_error_*.log`.

`python -m scripts.verifica_osserva` esegue una scena per ognuna delle sei
sorgenti su tutti i profili `clean`, `fixed` e `random`. Ogni combinazione e
isolata: un fallimento non impedisce di verificare le altre. Il report completo
viene scritto in `outputs/observe_tests/verifica_completa_*.json`.

Per ogni seed, il runner crea una scena assestata e la consegna alle sorgenti.
Solo il valutatore legge le pose e i contatti MuJoCo come riferimento. Il
report JSON in `outputs/observe_tests/` conserva configurazione, seed di
ingresso, seed della scena, parametri campionati, metriche per scena, medie,
deviazioni standard e seed dei casi peggiori. Ogni scena produce anche DOT e,
se Graphviz e disponibile, SVG dei supporti: nero = riferimento, verde =
relazione confermata, arancione = aggiunta, rosso tratteggiato = mancante.

La configurazione usa valori fissi o intervalli `[min, max]` campionati una
volta per scena. `degraded` applica rumore gaussiano alle coordinate 3D e
omissione degli oggetti. La scelta `--stereo-profile clean|fixed|random` e
indipendente dal profilo Degraded: `stereo` trasla separatamente le maschere
delle due viste di pixel gaussiani interi e puo omettere una detection per
vista. **Questo disturbo riguarda le rilevazioni, non i pixel delle immagini**:
le maschere sono ancora etichette ideali MuJoCo. `--objects N` sceglie un
numero fisso; `--objects random` estrae il numero nell'intervallo inclusivo
`--min-objects`/`--max-objects` per ogni scena. Ogni scena mantiene lo stesso
numero e stato fisico per tutte le sorgenti. Il seed rende riproducibili sia
il numero di oggetti sia i parametri di disturbo; il report salva i valori
effettivi per ogni scena.

`SimulatedStereoCamera` converte in scala di grigi il render MuJoCo e produce
un `StereoFrame` con immagini e calibrazione, senza ID. Le immagini restano a
tre canali identici per il contratto esistente. `BundleBuilder` con `OracleDetector` e il
componente separato che aggiunge maschere e ID da segmentazione MuJoCo e
applica i disturbi. Le maschere rappresentano solo i pixel visibili e sono
allineate alle immagini, ma il mapping istanza/ID e ideale. Il test misura acquisizione simulata,
triangolazione e blocchi successivi di OSSERVA. Con le maschere oracle non
misura il riconoscimento visivo; questa capacita viene valutata separatamente
da `fusion_learned`.
Nel percorso `fusion_oracle` l'identita resta ideale. Nel percorso
`fusion_learned`, invece, maschere e classi provengono dal modello addestrato;
MuJoCo viene letto solo dal valutatore per calcolare le metriche.

Il menu e `--mode rgbd` offrono inoltre una sorgente con **depth renderizzata
da MuJoCo**, distinta dalla triangolazione stereo. La depth viene
retroproiettata con le maschere oracle in nuvole di punti visibili; forma e
dimensioni sono stime, quindi il grafo diventa calcolabile senza ricevere
direttamente le dimensioni vere dal simulatore. Il report aggiunge
`mean_size_error_m` e `shape_accuracy`. Si vedano [SENSORS.md](SENSORS.md) e
`python -m physical_ai_mujoco.evaluation.sensor_calibration` per il test dei
disturbi di calibrazione.

Nel profilo attuale il centro del rig e `(0, -0.55, 0.32)` m, orientato verso
`(0, 0, 0.07)` m. Le camere sono parallele, a 0.15 m di distanza, con immagini
320x240 e campo verticale di 50 gradi. Sono parametri iniziali di scena, non
una posa ottimizzata per rendere ogni oggetto visibile. La baseline usata dai
test e sovrascritta da `STEREO_BASELINE_M` nel modulo della camera.

Le metriche sono recall/precision degli oggetti, errore euclideo delle pose
riconosciute, precision/recall/F1 dei candidati di supporto e stato delle
invarianti. Nel confronto dei supporti si esclude il terreno e si converte
`support_graph()` (oggetto alto -> supporto basso) nel verso delle relazioni
OSSERVA (basso -> alto). `support_graph()` e un riferimento approssimato dai
contatti; il F1 non e una misura delle dipendenze dinamiche di manipolazione.
Se la sorgente non offre dimensioni degli oggetti, le metriche di relazione
sono `null`, non zero. Un denominatore vuoto e riportato come `null`.

Le invarianti del contratto sono verificate dal builder e da
`validate_observation()`, senza ricostruire l'output dell'observer. Il benchmark
registra per ogni scena `invariants_ok` e `invariant_violations`. Se
l'Observation e malformata, conserva il motivo nel JSON, esporta un grafo
diagnostico al posto del grafo dei supporti e continua con le altre sorgenti e
scene. Le metriche di accuratezza escludono quell'output: non viene assegnato
uno zero che sembrerebbe una misura fisica. Il comando termina con codice 1 se
almeno una scena viola un'invariante. Un errore diverso, per esempio in
configurazione o acquisizione, viene lasciato emergere e non viene presentato
come errore del contratto. Il JSON viene scritto prima dell'esportazione dei
grafi, cosi rimane disponibile anche se Graphviz fallisce.

I test binari controllano a parte ID, relazioni, frame, timestamp, visibilita
degli oggetti ricordati e copertura dell'incertezza, oltre al confine dei dati
privilegiati: solo Exact espone `privileged_state`, gli oggetti di Observation
non contengono massa, attrito o velocita, e `observe()` non legge lo stato
privilegiato. Questi test hanno tolleranza zero; il report scientifico misura
invece l'accuratezza senza soglie arbitrarie, che richiedono una baseline
osservata e una decisione esplicita.

Il report contiene anche `object_uncertainty` per ogni osservazione:
`visible`, `remembered`, `detection_quality` e `pose_quality`, oltre al
booleano globale `unknown_space`. Questi numeri non sono confidenze calibrate:
Exact usa 1, Degraded una qualita sintetica legata al drop, Stereo lascia la
qualita numerica sconosciuta. Il benchmark crea un osservatore nuovo per ogni
scena, quindi non misura ancora la memoria tra frame consecutivi; quella e
coperta dai test unitari.
