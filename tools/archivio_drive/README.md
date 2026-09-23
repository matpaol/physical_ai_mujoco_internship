# `tools/archivio_drive` — archivio Drive dei risultati pesanti

Git tiene codice, configurazioni e pesi consegnati. Tutto il resto che serve
alla tesi — dataset generati, cartelle di training, video, benchmark,
calibrazioni, dati reali — va nella cartella `physical_ai_mujoco_tesi/` di
Google Drive. Questo modulo la gestisce: la cataloga per modulo del progetto,
tiene un registro di ogni operazione e riporta i file su qualsiasi PC.

Il quadro generale (cosa va in git, cosa su Drive, come si lavora su più PC) è
in [docs/SINCRONIZZAZIONE.md](../../docs/SINCRONIZZAZIONE.md).

---

## File del modulo

| File | Contenuto |
|---|---|
| `archivio_drive.py` | tutto il codice: comandi, registro, README generato. Solo libreria standard. |
| `__main__.py` | permette `python -m tools.archivio_drive` |
| `README.md` | questo file |
| `../../configs/archivio_drive.json` | moduli, categorie, cartelle, compressione, esclusioni |
| `../../tests/test_archivio_drive.py` | test (suite `archivio` di `main_test.py`); una cartella temporanea fa da Drive |

Il modulo **non importa nulla** di `physical_ai_mujoco` e non dipende
dall'ambiente del progetto: gira con qualsiasi Python ≥ 3.10.

---

## Dove si trova Drive

Il modulo lo trova da solo:

| PC | Percorso |
|---|---|
| Windows (ROG) | `G:\Il mio Drive` (app Google Drive per desktop) |
| Mac | `~/Library/CloudStorage/GoogleDrive-<account>/Il mio Drive` |
| Colab | `/content/drive/MyDrive`, dopo `drive.mount('/content/drive')` |
| Altro | `--drive PERCORSO` oppure variabile d'ambiente `ARCHIVIO_DRIVE` |

Non serve nessuna credenziale: scrive nella cartella locale di Drive e l'invio
lo fa l'app.

---

## Struttura dell'archivio

Ricalca i moduli di [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md). Le
categorie sono definite in `configs/archivio_drive.json`.

```
physical_ai_mujoco_tesi/
├── README.md              generato: struttura, catalogo, vista per fase, registro
├── registro.jsonl         una riga JSON per operazione, append-only
├── 01_scena/
│   └── magazzini_scene/   [zip]
├── 02_simulazione/
│   ├── verifiche_fisica/
│   └── video/
├── 03_osserva/
│   ├── benchmark/
│   ├── calibrazione/
│   └── catture/           [zip]
├── 04_visore/
│   ├── dataset/           [zip]
│   ├── training/
│   ├── pesi/
│   └── valutazioni/
├── 05_decidi/
│   ├── training/
│   └── valutazioni/
├── 06_esegui/
│   └── ur5/               dalla fase 2
├── 07_dati_reali/
│   ├── foto/              [zip]
│   └── acquisizioni/      [zip]
└── 90_tesi/
    ├── figure/
    ├── tabelle/
    └── video_demo/
```

`python -m tools.archivio_drive categorie` stampa l'elenco aggiornato con la
descrizione di ogni categoria.

### Il run

Ogni caricamento è un **run**: una cartella `AAAA-MM-GG_nome` dentro la sua
categoria, per esempio `04_visore/training/2026-09-22_sim_dr_v1_rtx3050/`.
Contiene i file e una `scheda.json` con:

- operazione (`aggiunta` o `nuova_versione`) e run precedente;
- data e ora (Europe/Rome), PC, autore;
- categoria, fase, nota;
- sorgenti (percorsi nel repository) e dove riportarle con `scarica`;
- commit git, branch, presenza di modifiche non committate;
- elenco dei file con dimensione e SHA-256, più un'impronta dell'insieme.

Le categorie `[zip]` contengono migliaia di file piccoli: il run contiene un
solo `<nome>.zip`. Drive sincronizza male i file sciolti; uno zip si carica in
un colpo e si verifica.

### Registro e README

`registro.jsonl` ha una riga per ogni operazione (`aggiunta`,
`nuova_versione`, `annotazione`). Non si modifica e non si cancella nessuna
riga. Il `README.md` dell'archivio è **rigenerato** dal registro a ogni
operazione: non va modificato a mano.

---

## Comandi

Dalla radice del repository. Tutti accettano `--drive`, `--config`,
`--autore`, `--pc`.

### `categorie`

```bash
python -m tools.archivio_drive categorie
```

### `carica` — archiviare

```bash
python -m tools.archivio_drive carica SORGENTE [SORGENTE ...] \
    --categoria CATEGORIA --nota "..." [--fase 1B] [--nome NOME] \
    [--zip | --no-zip] [--forza] [--prova]
```

- Una sola cartella: il suo contenuto va alla radice del run. Più sorgenti
  (file o cartelle): ognuna entra con il suo nome.
- `--nota` **obbligatoria**: cosa contiene e perché lo archivi, con i numeri.
- `--fase`: un profilo di `configs/experiments/` (`0A`, `0B`, `1A`, `1B`, `1C`, `2`, …).
- `--nome`: nome del run (default: nome della prima sorgente).
- `--zip` / `--no-zip`: scavalca la scelta della categoria.
- `--prova`: dice cosa farebbe, senza scrivere nulla.
- `--forza`: archivia anche se un run identico esiste già.

Esempi:

```bash
# cartella di training del visore
python -m tools.archivio_drive carica outputs/detector_training/sim_dr_v1_rtx3050 \
    --categoria visore/training --fase 1B \
    --nota "Training su RTX 3050, 190 epoche, mAP50 maschere 0,84"

# dataset generato (compresso da solo)
python -m tools.archivio_drive carica datasets/generated/sim_dr_v1 \
    --categoria visore/dataset --fase 1B --nota "..."

# pesi + scheda in un run
python -m tools.archivio_drive carica outputs/detector_weights/X.pt outputs/detector_weights/X.json \
    --categoria visore/pesi --nota "..."
```

### `stato` — cosa c'è

```bash
python -m tools.archivio_drive stato             # elenco dei run
python -m tools.archivio_drive stato --verifica  # ricalcola gli hash: file mancanti o alterati
```

### `scarica` — riportare un run su questo PC

```bash
python -m tools.archivio_drive scarica 2026-09-22_sim_dr_v1_rtx3050
python -m tools.archivio_drive scarica 04_visore/training/2026-09-22_sim_dr_v1_rtx3050 --dest altrove/
```

Basta il nome del run se non è ambiguo. Senza `--dest` i file tornano dove
stavano nel repository. I file locali identici vengono saltati; quelli
**diversi** non vengono toccati: il comando li elenca ed esce con codice 1
(`--sovrascrivi` per rimpiazzarli). Ogni file scaricato viene verificato con
l'hash.

### `annota` — correggere o completare una nota

```bash
python -m tools.archivio_drive annota 2026-09-22_sim_dr_v1_rtx3050 --nota "..."
```

Aggiunge una riga `annotazione` al registro; il README la mostra accanto alla
nota originale.

### `prepara` e `readme`

```bash
python -m tools.archivio_drive prepara   # crea tutte le cartelle vuote e il README
python -m tools.archivio_drive readme    # rigenera il README dal registro
```

---

## Cosa fa da solo

- **Niente doppioni:** se il contenuto (impronta degli hash) è identico a un
  run già presente nella stessa categoria, non carica nulla.
- **Niente sovrascritture:** se il contenuto è cambiato crea un run nuovo
  (`nuova_versione`, con `_v2`, `_v3`… se nello stesso giorno) e lascia intatto
  il precedente.
- **Copie verificate:** ogni file copiato viene riletto e confrontato con
  l'hash; lo zip viene scritto come `.parziale`, controllato e poi rinominato.
  Se qualcosa va storto il run non entra nel registro.
- **Git in sola lettura:** legge commit e branch con `GIT_OPTIONAL_LOCKS=0`,
  senza toccare l'indice.

---

## Regole

1. Nell'archivio si scrive **solo** con questo modulo: niente trascinamenti,
   rinomine o cancellazioni a mano.
2. Ogni run ha una nota che dice **perché** esiste, come le voci di
   `MODIFICHE.md`: *"190 epoche, mAP50 maschere 0,84, prima del fine-tuning
   reale"*, non *"training"*.
3. La fase non è una cartella: è `--fase`. Il README ha una vista per fase.
4. `90_tesi/` contiene materiale finito per il documento, non run grezzi.
5. I pesi **consegnati** stanno in git (`outputs/detector_weights/`); in
   `04_visore/pesi` va l'archivio completo, anche intermedi e scartati.

---

## Estendere

- **Una categoria nuova** (es. quando arriva l'UR5): si aggiunge in
  `configs/archivio_drive.json` sotto `categorie`, con `cartella` dentro un
  modulo dichiarato in `moduli`, `descrizione` e, se contiene migliaia di file,
  `"comprimi": true`. Il codice non si tocca. Voce in `MODIFICHE.md`.
- **Un modulo nuovo**: una riga in `moduli` (numero davanti per l'ordine).
- **Il codice**: dopo ogni modifica `python main_test.py --suite archivio`, e
  `VERSIONE_STRUMENTO` va incrementata se cambia il formato di scheda o registro.

---

## Per le IA che lavorano sul repository

- Quando una sessione produce risultati pesanti utili alla tesi, proponi di
  archiviarli e scrivi tu la `--nota`, con il problema o lo scopo e i numeri.
- Firma con `--autore "<IA> (<modello>) su richiesta di <persona>"`.
- Usa prima `--prova` e mostra alla persona cosa verrà caricato.
- Non modificare a mano `README.md`, `registro.jsonl` o le `scheda.json`
  dell'archivio; per correggere usa `annota`.
