# Addestramento del visore (`vision_training`)

Questo pacchetto produce il **visore**: la rete di segmentazione (YOLO-seg) che
in OSSERVA riconosce ostacoli e target nelle immagini della camera. Genera i
dataset da MuJoCo, addestra la rete e misura fino a che punto riconosce il
target quando e' parzialmente coperto.

Non e' codice che gira sul robot. Il robot usa solo il risultato, cioe' i pesi
`.pt` caricati da `sensors/learned_detector.py`.

## Perche' e' nato

Il 21 settembre 2026 il benchmark `fusion_learned` (4 scene, 3-8 oggetti) dava
richiamo del target 25%. Il visore di allora, `pfm_1_seg_immersed_v2.pt`:

- era stato addestrato su 80 scene con 1-6 oggetti, sempre con lo stesso
  pavimento a scacchi, la stessa luce, la stessa camera e lo stesso colore del
  target;
- non aveva una scheda: non si sapeva piu' con quante epoche ne su quali dati;
- non si poteva riaddestrare dal menu, che chiedeva solo scene, oggetti e seed.

Prova fatta lo stesso giorno sulle stesse scene: senza variazioni visive trova
il target nel 100% dei casi, con luci, terreno e colori variati nello 0%. Un
visore che funziona solo nelle condizioni su cui e' stato addestrato non puo'
funzionare nel mondo reale.

Le decisioni tecniche sono annotate in `MODIFICHE.md`, voci del 2026-09-21.

## Il ciclo in quattro passi

```text
 ricetta dataset (JSON)                       ricetta training (JSON)
        |                                              |
        v                                              v
 1. GENERA  dataset.py  ----> datasets/generated/<nome>/ ----> 2. ADDESTRA  training.py
    MuJoCo + randomizzazione      train / val / test              YOLO (Ultralytics)
                                                                   |
                          outputs/detector_weights/<nome>.pt  <----+
                          outputs/detector_weights/<nome>.json (scheda)
                                                                   |
 3. VALUTA  evaluation.py  <---------------------------------------+
    richiamo per fascia di % visibile --> soglia minima scritta nella scheda
        |
        v
 4. USA  OSSERVA (LearnedDetector) e benchmark: la soglia decide quando il
         target conta come "osservabile"
```

Per il sim2real il passo 2 si ripete due volte: prima su immagini MuJoCo
(pre-training), poi sulle foto reali partendo dai pesi sim (fine-tuning).

## Mappa dei file

| File | Cosa fa |
|---|---|
| `recipe.py` | Legge e valida le ricette JSON. Una chiave sconosciuta e' un errore, non un default silenzioso. |
| `randomization.py` | Estrae le condizioni visive di ogni vista (camera, luci, terreno, sfondo, colori) e applica i disturbi d'immagine. Solo numeri, niente simulatore. |
| `dataset.py` | Costruisce le scene, applica le condizioni, fotografa con le due camere, misura la % visibile del target, scrive il dataset. |
| `labels.py` | Formato su disco: PNG, maschere, poligoni YOLO, JSON per immagine, `data.yaml`. Vale anche per le foto reali. |
| `training.py` | Addestra con Ultralytics, sceglie il device (CUDA, poi MPS, poi CPU), scrive la scheda del modello, valuta sul test. |
| `evaluation.py` | Richiamo del target per fasce di % visibile, falsi positivi, soglia minima; aggiorna la scheda. |

Pezzi fuori da questo pacchetto che il visore usa:

| Dove | Cosa |
|---|---|
| `simulation/visual_conditions.py` | Le condizioni visive come dati puri (`VisualConditions` e sottoclassi). |
| `simulation/simulator.py` | `apply_visual_conditions()` cambia l'aspetto sul modello gia' compilato; `hold_object()` tiene fermo il target interrato mentre il resto della pila ricade. |
| `evaluation/target_exposure.py` | `measure_target_visibility()`: pixel visibili / pixel del target da solo, senza modificare la scena. |
| `evaluation/observe_benchmark.py` | Legge la soglia dalla scheda del modello per decidere quando il target e' "osservabile". |
| `sensors/learned_detector.py` | Il consumatore finale: carica i pesi a runtime. Non e' stato toccato. |
| `observe/main_test_osserva.py` | Menu: 2 genera, 3 addestra, 5 valuta. |
| `configs/vision_training/` | Le ricette. |

Regola di confine: questo pacchetto **non importa MuJoCo**. L'aspetto della
scena si cambia solo con `Simulator.apply_visual_conditions()`. Lo controlla
`tests/test_architecture.py`.

## Come si usa

Guida passo passo con tutte le opzioni, come leggere i risultati e i problemi
frequenti: `docs/ADDESTRAMENTO_VISORE.md`.

Dalla cartella del progetto, sempre con lo stesso ambiente Python:

```bash
# 1. dataset
python -m scripts.genera_dataset_target configs/vision_training/dataset_sim_dr_v1.json
# 2. training (a fine training valuta da solo sul test)
python -m scripts.train_detector configs/vision_training/training_sim_dr_v1_m3.json
# 3. valutazione di un modello qualsiasi su un dataset qualsiasi
python -m physical_ai_mujoco.vision_training.evaluation \
    outputs/detector_weights/sim_dr_v1_m3.pt datasets/generated/sim_dr_v1 --split test
```

Oppure dal menu: `python physical_ai_mujoco/observe/main_test_osserva.py`,
voci 2, 3 e 5.

Prima di un run lungo conviene sempre la prova veloce: ricette
`dataset_smoke.json` e `training_smoke.json`, pochi minuti in tutto.

Tempi misurati su MacBook Air M3 16 GB il 21/09/2026: dataset smoke (6 scene)
4 s; dataset `sim_dr_v1` (800 scene, 3.392 immagini) 24 minuti; training smoke
(2 epoche, 16 immagini) circa 15 s su MPS. La generazione non apre finestre:
l'avanzamento compare ogni 1/20 delle scene.

## Le ricette

Un esperimento = una ricetta = un `name`. Per cambiare qualcosa si **copia la
ricetta e si cambia il nome**: dataset e modelli esistenti non vengono mai
sovrascritti (il codice si rifiuta).

Ricette presenti:

| Ricetta | A cosa serve |
|---|---|
| `dataset_smoke` / `training_smoke` | Prova che la pipeline gira. I numeri non valgono. |
| `dataset_sim_dr_v1` | Dataset principale: 800 scene, 1-10 oggetti, randomizzazione completa. |
| `dataset_sim_nodr_v1` | Stesse scene di `sim_dr_v1` senza randomizzazione: serve a misurare quanto guadagna la randomizzazione. |
| `training_sim_dr_v1_m3` | Training sul Mac (yolo11n-seg, 640 px, 100 epoche). |
| `training_sim_dr_v1_gpu` | Training su GPU NVIDIA (yolo11s-seg, 200 epoche). |
| `training_real_finetune_v1` | Fine-tuning sulle foto reali partendo dai pesi sim. Serve il dataset reale. |

Chiavi principali di una ricetta dataset:

| Chiave | Significato |
|---|---|
| `scene_rules` | Regole di scena (tipi di oggetti, target). |
| `scenes.count`, `scenes.object_count` | Quante scene; oggetti per scena `[min, max]`. |
| `scenes.target_immersion` | Quanto e' interrato il target `[min, max]`, 0 = appoggiato, 1 = sepolto. |
| `scenes.views_per_scene` | Quante foto diverse (condizioni diverse) per ogni scena. |
| `scenes.target_absent_fraction` | Quota di viste senza target (esempi negativi). |
| `scenes.hard_visibility` | Viste extra se il target e' poco visibile (range di % visibile). |
| `splits` | Quota di scene per validazione e test. Lo split e' per scena. |
| `labels.minimum_visible_pixels` | Sotto questi pixel un'istanza non viene etichettata. |
| `randomization` | `null` = nessuna variazione. Altrimenti sezioni `camera`, `lighting`, `ground`, `background`, `objects`, `image`; una sezione assente = quell'aspetto resta nominale. |

Chiavi principali di una ricetta training: `dataset`, `base_model` (un nome
Ultralytics come `yolo11n-seg.pt` oppure un nostro `.pt`), `epochs`,
`image_size`, `batch`, `device` (`auto`), `ultralytics` (parametri passati
cosi' come sono a Ultralytics), `evaluation` (come valutare a fine training).

Attenzione: con l'ottimizzatore automatico di Ultralytics il `lr0` della
ricetta viene ignorato. Se il learning rate conta, scrivere anche
`"optimizer": "AdamW"` (o `"SGD"`) nella sezione `ultralytics`.

## Cambiare lo scenario (oggetti, target)

Non e' una cosa che si tocca in questo pacchetto: si tocca la ricetta di
scena (`scene_rules`, es. `configs/observe_tests/immersed_scene_rules.json`)
e, se serve, il catalogo oggetti. `vision_training` non sa nulla dei tipi
specifici: etichetta sempre e solo due classi, `obstacle` e il
`target_type_id` dichiarato — qualunque tipo non sia il target diventa
"obstacle". Per questo cambiare lo scenario non richiede toccare `dataset.py`
ne' `labels.py`.

**Caso 1 — usare tipi gia' nel catalogo, cambiare solo quali/quanti.**
Il catalogo e' `datasets/object_dataset/geometric_objects.json` (forma,
dimensioni, massa, attrito, colore di ogni tipo). Nella ricetta di scena si
tocca solo `object_selection`:

- `required_type_ids`: i tipi che possono comparire. Non e' un limite al
  numero di oggetti — a runtime vengono ciclati per riempire quanti oggetti
  servono (`--objects` o `scenes.object_count` della ricetta dataset); vedi
  `Session._rules_with_object_count` in `simulation/session.py`.
- `target_type_id`: quale di questi e' il target.

**Caso 2 — un tipo che non esiste ancora nel catalogo.** Si aggiunge una voce
in `geometric_objects.json` (forma, `size_range`, `friction`, `rgba`; per una
mesh anche `mesh_file` STL e `mesh_scale`) e poi la si richiama da
`required_type_ids` come nel caso 1. E' una modifica del modulo scena/
simulazione (`scene/dataset_loader.py` valida il file), non del visore.

**Se cambia il target**, oltre a `target_type_id` vanno ricontrollate due
cose nella ricetta dataset, altrimenti si randomizzano colori o soglie
pensati per il target vecchio:

- `randomization.objects.target_palette`: oggi sono i colori realistici della
  PFM-1. Va rifatta per il colore vero del nuovo target.
- `labels.minimum_visible_pixels` e `scenes.hard_visibility.range`: dipendono
  da quanto e' piccolo il target e da come si "nasconde" quando interrato.

Limite noto: il sistema assume sempre un solo target e un'unica classe
generica di ostacoli. Piu' tipi di target, o classi di ostacolo distinte da
riconoscere separatamente, sono un cambiamento piu' grande e non sono
supportati oggi.

## Cosa si trova su disco

```text
datasets/generated/<nome>/
  images/{train,val,test}/*.png     foto in grigio
  labels/{train,val,test}/*.txt     poligoni YOLO-seg
  masks/{train,val,test}/*.png      maschera di ogni istanza, senza perdite
  annotations/{train,val,test}/*.json   per ogni foto: istanze, % visibile del
                                    target, oggetti, condizioni visive usate
  summary.json                      statistiche per split e per fascia
  recipe.json                       copia della ricetta usata
  data.yaml                         per Ultralytics (riscritto al training)

outputs/detector_weights/<nome>.pt     pesi
outputs/detector_weights/<nome>.json   scheda: ricetta, dataset, genitore,
                                       versioni, commit git, metriche, valutazioni
outputs/detector_training/<nome>/      cartella di lavoro di Ultralytics
outputs/detector_evaluations/*.json    report completi, anche immagine per immagine
```

## Come si legge la valutazione

- **Fasce di % visibile.** Per ogni fascia (0-10%, 10-20%, ...) si conta in
  quante foto il target e' riconosciuto (IoU della maschera >= 0.5).
- **Soglia minima.** Si parte dalla fascia piu' visibile e si scende; ci si
  ferma alla prima fascia con richiamo sotto l'80%. Il limite inferiore
  dell'ultima fascia buona e' la **% minima riconoscibile**. Contano solo le
  fasce con abbastanza foto (`min_samples_per_band`).
- **Falsi positivi.** Quota di foto senza target in cui il visore ne vede uno.
- La soglia misurata finisce nella scheda (`observability`). Il benchmark di
  OSSERVA la usa per dire se il target era "osservabile": ordine di priorita'
  scheda del modello, poi profilo del benchmark (5%), poi default del codice.

## Sim2real: le foto reali

Il formato del dataset e' lo stesso per sim e reale. Per il fine-tuning serve
`datasets/real/pfm1_real_v1/` con:

- `images/{train,val,test}/*.png` (in grigio, come la camera);
- `labels/...` nel formato YOLO-seg (classe 0 ostacolo, 1 target);
- `annotations/<split>/*.json` con almeno `sample_id`, `image` e `instances`
  (per ogni istanza `class_index` e `mask`), se si vuole usare `evaluation.py`;
- `summary.json` con `"class_names": ["obstacle", "pfm_1_target"]`.

Nelle foto reali la % visibile non si puo' misurare: la valutazione mette
tutto in una fascia `sconosciuta`. Uno strumento per convertire le etichette
fatte a mano in questo formato non esiste ancora.

## Manutenzione

- **Nuovo esperimento**: copia una ricetta, cambia `name`, lancia. Mai
  modificare una ricetta gia' usata per un modello che si tiene.
- **Nuovo aspetto da randomizzare** (per esempio foto reali del terreno come
  texture):
  1. aggiungere il campo in `simulation/visual_conditions.py`;
  2. applicarlo in `Simulator.apply_visual_conditions()`;
  3. aggiungere la sezione in `recipe.py` e l'estrazione in `randomization.py`;
  4. un test in `tests/test_vision_training.py`.
- **Flussi casuali**: ogni scopo ha un numero fisso (`STREAM_*` in
  `randomization.py`). Non riusarli per altro: e' cio' che fa si' che la
  stessa ricetta con e senza randomizzazione generi le stesse scene.
- **Test**: `python main_test.py --suite visore` (piu' `architettura` se si
  toccano i confini).
- **Registro**: ogni modifica va in `MODIFICHE.md` (vedi `CLAUDE.md`).

## Limiti noti

- Le texture del terreno sono macchie generiche, non terreni reali specifici.
- Il bordo del terreno (1 m) e lo sfondo entrano nell'inquadratura quando la
  camera e' inclinata; lo sfondo e' randomizzato, il bordo no.
- Con `target_immersion` fino a 0.95 il target e' spesso del tutto sepolto: in
  `sim_dr_v1` lo e' in 177 immagini di test su 480. Controllare
  `visibility_bands` nel `summary.json` prima di un training lungo.
- Le viste extra "poco visibile" dipendono dalla camera: fra una ricetta con e
  una senza randomizzazione il loro numero puo' differire.
- La % visibile ha un'incertezza di qualche pixel (bordi, rasterizzazione).
- Su MPS il training non e' riproducibile bit per bit (avvisi di PyTorch).
- I dataset generati prima di questo modulo non hanno la % visibile nei JSON.

## Checklist "fra un mese"

1. `python main_test.py --suite visore`: tutto verde?
2. Quale modello e' attivo? Senza indicazioni, benchmark e OSSERVA usano
   `outputs/detector_weights/pfm_1_seg.pt` se esiste, altrimenti il
   `pfm_1_seg*.pt` piu' recente: i modelli di questo modulo (es.
   `sim_dr_v1_m3.pt`) **non** vengono presi da soli. Per usarli: `--weights`
   nel benchmark, oppure `sensor_observation.detector_weights` nella config.
   Poi leggi la scheda `.json` del modello: ricetta, dataset, soglia misurata.
3. Da quale dataset viene? `resolved.dataset` nella scheda, poi
   `recipe.json` e `summary.json` nella cartella del dataset.
4. Cosa e' cambiato da allora? `MODIFICHE.md`, voci dopo la data della scheda.
5. Vuoi rifare o migliorare? Copia la ricetta, cambia il nome, rilancia.
