# Addestramento del visore — come si lancia

Guida pratica per addestrare il detector che riconosce target e ostacoli nelle
immagini delle telecamere. Solo comandi e ricette: il *perche'* delle scelte e'
in `physical_ai_mujoco/vision_training/README.md` e in `MODIFICHE.md`.

Tutti i comandi si lanciano dalla cartella `mujoco_deploy`.

---

## Il ciclo in 3 passi

```
ricetta dataset  ──►  1. genera il dataset   (MuJoCo: immagini + etichette)
ricetta training ──►  2. addestra il modello (YOLO11-seg)
                      3. valuta per % di target visibile
```

Ogni passo prende **una ricetta JSON** in `configs/vision_training/`.
Un esperimento nuovo = una ricetta nuova con un `name` nuovo.

---

## 0. Prova veloce (farla sempre dopo una modifica)

```bash
python -m scripts.genera_dataset_target configs/vision_training/dataset_smoke.json
python -m scripts.train_detector configs/vision_training/training_smoke.json
```

Pochi minuti. Serve solo a vedere che tutto gira; i numeri non contano.
Per rilanciarla, cancellare prima `datasets/generated/smoke/` e
`outputs/detector_weights/smoke.*` (i comandi rifiutano di sovrascrivere).

---

## 1. Generare il dataset

```bash
python -m scripts.genera_dataset_target configs/vision_training/dataset_sim_dr_v1.json
```

| | |
|---|---|
| Output | `datasets/generated/sim_dr_v1/` |
| Tempo | ~24 min su MacBook M3 (800 scene, ~3400 immagini) |
| Finestre | nessuna: stampa `scene N/800` ogni 40 scene |
| Opzione | `--output CARTELLA` per salvare altrove |

Warning `AGX: exceeded compiled variants footprint limit` sul Mac: innocuo.

**Da controllare alla fine** (stampato e in `summary.json`):

- `visibility_bands` per `train`, `val`, `test`: quante immagini per fascia di %
  visibile. Se una fascia ha meno di 10 immagini nel test, la sua riga nella
  valutazione non conta.
- `target_absent`: viste senza target (servono per i falsi positivi).
- `hard_visibility_views`: viste extra delle scene col target poco visibile.

Contenuto della cartella:

| File | Cosa contiene |
|---|---|
| `images/{train,val,test}/*.png` | immagini in grigio, come la camera |
| `labels/{train,val,test}/*.txt` | poligoni YOLO-seg (0 = ostacolo, 1 = target) |
| `masks/{train,val,test}/*.png` | maschera esatta di ogni oggetto |
| `annotations/{train,val,test}/*.json` | per ogni immagine: % visibile, interramento, condizioni visive estratte |
| `manifest.jsonl` | tutte le annotazioni in un file |
| `data.yaml` | quello che legge Ultralytics |
| `recipe.json` | copia della ricetta usata |
| `summary.json` | il riassunto |

---

## 2. Addestrare

```bash
python -m scripts.train_detector configs/vision_training/training_sim_dr_v1_m3.json    # Mac M3
python -m scripts.train_detector configs/vision_training/training_sim_dr_v1_gpu.json   # GPU NVIDIA
python -m scripts.train_detector configs/vision_training/sim_dr_v1_rtx3050.json        # ROG, RTX 3050 4 GB
```

`sim_dr_v1_rtx3050` e' la ricetta GPU con `batch: 8` per stare nei 4 GB di
VRAM della RTX 3050 Laptop. **Il modello consegnato oggi e'
`outputs/detector_weights/sim_dr_v1_rtx3050.pt`** (22/09/2026, 190 epoche,
mAP50 maschere 0,84): e' in git; la cartella di training completa e' nell'archivio
Drive (`04_visore/training/`, vedi [SINCRONIZZAZIONE.md](SINCRONIZZAZIONE.md)).

| | Mac M3 | GPU NVIDIA |
|---|---|---|
| Modello di partenza | `yolo11n-seg` (piccolo) | `yolo11s-seg` (piu' capiente) |
| Epoche max / patience | 100 / 25 | 200 / 40 |
| Batch | 16 | 32 |

- Device scelto da solo: CUDA → MPS (Mac) → CPU. Se MPS da' errori:
  `"device": "cpu"` nella ricetta.
- Si ferma prima se per `patience` epoche non migliora.
- `--no-evaluation`: salta la valutazione finale.

Output:

| File | Cosa contiene |
|---|---|
| `outputs/detector_weights/<name>.pt` | i pesi |
| `outputs/detector_weights/<name>.json` | **scheda del modello**: ricetta, dataset, modello padre, versioni, commit git, metriche, soglia minima misurata |
| `outputs/detector_training/<name>/` | cartella di lavoro di Ultralytics (curve, `results.csv`) |

---

## 3. Valutare

A fine training la valutazione parte da sola. Per rifarla o valutare un altro
modello:

```bash
python -m physical_ai_mujoco.vision_training.evaluation \
    outputs/detector_weights/sim_dr_v1_rtx3050.pt datasets/generated/sim_dr_v1 --split test
```

| Opzione | Default | Significato |
|---|---|---|
| `--split` | `test` | `train`, `val` o `test` |
| `--iou` | 0.5 | sovrapposizione minima per dire "trovato" |
| `--confidence` | 0.25 | confidenza minima del detector |
| `--band-width` | 0.1 | larghezza delle fasce di % visibile |
| `--required-recall` | 0.8 | richiamo richiesto per la soglia minima |
| `--min-samples` | 10 | immagini minime perche' una fascia conti |
| `--output FILE` | | dove scrivere il report |

Report in `outputs/detector_evaluations/`. Valutando sul `test`, la scheda del
modello viene aggiornata con la soglia misurata.

### Come si legge

Forma dell'output (numeri inventati, solo per leggere):

```
Split: test | campioni: 480 | target etichettati: 250
Richiamo del target complessivo: 78.0%
  % visibile   campioni   richiamo   IoU medio
    0.00-0.10         40      15.0%        0.21
    0.10-0.20         35      51.4%        0.48
    ...
    0.50-0.60         30      90.0%        0.77
Scene senza target: 48, falsi positivi 4.2%
Soglia minima misurata: 50% di sagoma visibile
```

- **richiamo** = in quante immagini di quella fascia il target e' stato trovato
  (IoU ≥ 0.5).
- **IoU medio** = quanto bene la maschera trovata copre quella vera.
- **soglia minima** = la fascia piu' bassa da cui in su il richiamo resta
  ≥ 80%. E' la % che il benchmark di OSSERVA usa per dire "target osservabile"
  quando si usa quel modello.
- **falsi positivi** = viste senza target in cui il modello ne vede uno.

### Confronti utili

```bash
# il modello vecchio sul test randomizzato
python -m physical_ai_mujoco.vision_training.evaluation \
    outputs/detector_weights/pfm_1_seg_immersed_v2.pt datasets/generated/sim_dr_v1 --split test

# randomizzazione si' / no: stesso test, due modelli
python -m physical_ai_mujoco.vision_training.evaluation \
    outputs/detector_weights/sim_nodr_v1_m3.pt datasets/generated/sim_dr_v1 --split test
```

Per il secondo serve prima `dataset_sim_nodr_v1` e una ricetta di training su
di esso (copia di `training_sim_dr_v1_m3.json` con `name` e `dataset` cambiati).

---

## 4. Usare il modello in OSSERVA

Il benchmark **non** prende da solo i modelli nuovi; va passato:

```bash
python -m physical_ai_mujoco.evaluation.observe_benchmark \
    --config configs/observe_tests/random.json --mode fusion_learned \
    --weights outputs/detector_weights/sim_dr_v1_rtx3050.pt --scenes 20 --headless
```

La prima riga dei risultati dice quale soglia di "target osservabile" e' in uso
(dalla scheda del modello, dal profilo o il 5% di default).

---

## 5. Sim2real: fine-tuning sulle foto reali

```bash
python -m scripts.train_detector configs/vision_training/training_real_finetune_v1.json
```

Parte dai pesi `sim_dr_v1_rtx3050.pt`, congela i primi 10 blocchi (`freeze: 10`),
learning rate basso (`lr0: 0.001` con `optimizer: AdamW`: con `auto`
Ultralytics ignorerebbe `lr0`). La scheda registra il modello padre.

Serve prima il dataset reale in `datasets/real/pfm1_real_v1/`, **stesso
formato** del sintetico:

- `images/{train,val,test}/*.png` in grigio;
- `labels/{train,val,test}/*.txt` YOLO-seg, classe 0 ostacolo, 1 target;
- `data.yaml`.

Sulle foto reali la % visibile non si misura: la valutazione mette tutto in
una fascia `sconosciuta`. Il convertitore dalle etichette fatte a mano non
esiste ancora. Per addestrare su GPU partendo dai pesi GPU, cambiare
`base_model` in una copia della ricetta.

---

## 6. Fare un esperimento nuovo

1. Copiare la ricetta piu' vicina in `configs/vision_training/`.
2. Cambiare `name` (e `dataset` se e' una ricetta di training) e cio' che si
   vuole provare. **Una cosa alla volta**, cosi' il confronto dice cosa ha
   fatto la differenza.
3. Lanciare dataset → training → valutazione come sopra.
4. Copiare la tabella per fasce in `05_tesi_appunti.md`.

Non modificare una ricetta gia' usata per un modello che si tiene.

Chiavi che si cambiano piu' spesso:

| Ricetta | Chiave | Effetto |
|---|---|---|
| dataset | `scenes.count` | numero di scene |
| dataset | `scenes.object_count` | [min, max] oggetti per scena |
| dataset | `scenes.target_immersion` | [min, max] interramento del target (0-1) |
| dataset | `scenes.views_per_scene` | viste per scena |
| dataset | `scenes.target_absent_fraction` | quota di viste senza target |
| dataset | `scenes.hard_visibility` | fascia di % visibile che riceve viste extra |
| dataset | `randomization` | `null` = nessuna randomizzazione; altrimenti camera, luci, terreno, sfondo, colori, effetti immagine |
| dataset | `seed` | stesso seed = stesse scene |
| training | `base_model` | `yolo11n-seg.pt`, `yolo11s-seg.pt` o un `.pt` gia' addestrato |
| training | `epochs`, `patience`, `batch`, `image_size` | durata e memoria |
| training | `ultralytics` | opzioni passate a Ultralytics (augmentation, `lr0`, `freeze`, …) |
| training | `evaluation` | parametri della valutazione finale |

Il significato completo di ogni chiave e' nel README del visore. Una chiave
scritta male fa fermare subito il comando con un errore.

---

## 7. Problemi frequenti

| Messaggio / sintomo | Causa | Cosa fare |
|---|---|---|
| "esiste gia'" | stesso `name` di un dataset o modello esistente | nuovo `name`, o cancellare il vecchio |
| chiave sconosciuta nella ricetta | refuso o chiave non supportata | correggere il nome |
| sembra fermo durante la generazione | stampa solo ogni 1/20 delle scene | aspettare |
| errore MPS sul Mac | operazione non supportata da MPS | `"device": "cpu"` |
| il benchmark usa il modello vecchio | i modelli nuovi non sono presi di default | passare `--weights` |
| fasce senza soglia | meno di `min_samples` immagini nella fascia | dataset piu' grande o `--min-samples` piu' basso (dichiararlo) |

## Test del modulo

```bash
python main_test.py --suite visore
```
