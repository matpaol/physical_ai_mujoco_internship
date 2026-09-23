# Istruzioni per chi lavora su questo repository

Valgono per le persone e per i modelli linguistici. Gli strumenti di sviluppo
basati su IA leggono questo file all'avvio: se stai leggendo, riguarda te.

## 1. Il registro delle modifiche non è facoltativo

**Ogni modifica al codice richiede una voce in `MODIFICHE.md`, scritta nella
stessa sessione in cui il codice viene toccato.**

Codice modificato senza la sua voce è lavoro incompleto, anche a test verdi. Il
file contiene le regole per scriverla; il campo che conta è **«Perché»**, con il
problema osservato e i numeri che lo mostravano — non l'intenzione.

Data e ora al secondo, fuso `Europe/Rome`:

```bash
TZ=Europe/Rome date '+%Y-%m-%d %H:%M:%S'
```

Un'IA firma come tale, indicando il modello e la persona che l'ha istruita. Non
si firma mai come se fosse l'autore umano.

## 2. Le regole di struttura

- **Solo `physical_ai_mujoco/simulation/` importa `mujoco`.** Gli altri moduli
  parlano per interfacce. È ciò che permetterà di sostituire il simulatore con
  il robot vero senza riscrivere il resto.
- **Fase 0A è congelata.** Deve continuare a produrre scene byte-identiche a
  parità di seed. Fase 0B ha config, dataset e regole di scena propri: se serve
  cambiare qualcosa, si cambia lì.
- **La fase vive nella configurazione**, non nel codice. Nessun modulo deve
  contenere un `if` sulla fase o sulla modalità in cui sta girando.

## 3. Verificare, non supporre

`pytest` prima di consegnare. Se una modifica non è stata verificata, la voce in
`MODIFICHE.md` deve dirlo: è un'informazione utile, nasconderla no.

## 4. Dove si trova cosa

| File | Contenuto |
|---|---|
| `README_oa.md` | come si usa il codice, e perché funziona così |
| `MODIFICHE.md` | cosa è cambiato, quando, per mano di chi e per quale motivo |
| `configs/phase_0b/` | i parametri di Fase 0B: task, scena, simulazione, dataset |
| `docs/SINCRONIZZAZIONE.md` | lavorare su più PC: cosa va in git, cosa su Drive, comandi git |
| `tools/archivio_drive/README.md` | archivio Drive dei risultati pesanti: struttura, comandi, regole |

## 5. Risultati pesanti e più PC

In git vanno codice, configurazioni e **solo i pesi consegnati**
(`outputs/detector_weights/`). Dataset generati, cartelle di training, video,
benchmark e dati reali vanno nell'archivio Drive con
`python -m tools.archivio_drive carica ... --nota "..."`: la nota è
obbligatoria e segue le regole del «Perché» di `MODIFICHE.md`. Non modificare a
mano i file dell'archivio. Le operazioni git che scrivono (commit, push, merge)
le decide la persona: un'IA prepara i file e propone i comandi.
