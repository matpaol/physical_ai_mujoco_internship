# Piano di lavoro

Cosa si sta facendo, in quale branch, cosa e' finito e cosa resta aperto.
Per chi riprende il lavoro su DECIDI: [HANDOFF_DECIDI.md](HANDOFF_DECIDI.md);
le decisioni di progetto sono in [PROGETTO_DECIDI.md](PROGETTO_DECIDI.md).
Si aggiorna a ogni passo, insieme a `MODIFICHE.md` (che invece dice cosa e'
cambiato e perche'). Ultimo aggiornamento: 25/09/2026.

## Decisioni in vigore

- **OSSERVA reale congelato** (tag `osserva-v1`, stato in
  [OSSERVA_STATO.md](OSSERVA_STATO.md)). DECIDI ed ESEGUI si sviluppano
  sull'osservazione dell'oracolo MuJoCo (profilo `configs/experiments/oracolo.json`).
- **DECIDI restituisce un solo oggetto** per passo (`ObjectDecision`), non una lista.
- **Teacher e student separati**: lo student riceve solo `Observation`; il
  teacher anche `PrivilegedState` (solo simulazione).
- **Un branch per componente**; `osserva-oracolo` per ora **non** si unisce in `main`.
- Codice, test, commit e tag in inglese; documentazione in italiano.
- **Progetto di DECIDI** (25/09): confini dei moduli, ciclo del passo,
  `DecisionInput` = `Observation` + `DecisionContext`, K = 12 slot dal profilo,
  maschera solo strutturale, TASK come modulo di tutte le reward, training PPO
  + imitazione. Dettagli in [PROGETTO_DECIDI.md](PROGETTO_DECIDI.md), figure in
  [diagrammi/architettura_progetto.drawio](diagrammi/architettura_progetto.drawio).

## Stato

| Passo | Branch | Stato |
|---|---|---|
| Osservatore oracolo (grafo dai contatti via `PrivilegedState`) | `osserva-oracolo` | fatto (`876eacc`) |
| Main dell'oracolo: `Observation`, `PrivilegedState`, grafo DOT/SVG, JSON | `osserva-oracolo` | fatto (`0e4ae4f`) |
| `PrivilegedState` piu' ricco: centro di massa, tre attriti, forma e dimensioni | `osserva-oracolo` | fatto (`0e4ae4f`) |
| Main generale `main_pipeline.py`: scena → OSSERVA → DECIDI → ESEGUI → TASK, componenti dal profilo; `run_episode` unico ciclo | `decidi` (da `osserva-oracolo`) | fatto (`1ef115d`) |
| Interfacce student (`Decider.decide(observation)`) e teacher (`TeacherDecider.decide(observation, privileged)`) | `decidi` | fatto (`1ef115d`) |
| Struttura di `Observation`/`PrivilegedState` documentata ([STRUTTURA_OSSERVAZIONE.md](STRUTTURA_OSSERVAZIONE.md)), esempio `esempi/oracolo_seed7/` | `decidi` | fatto (`1ef115d`) |
| Grafo dell'oracolo: verso invertito in 19 archi su 147 (13%), target coperto sbagliato in 6 scene su 30; correggere con la forza verticale? | `osserva-oracolo` o `decidi` | da decidere |
| Progettazione del decisore: confini, contratti, slot, maschera, reward, training ([PROGETTO_DECIDI.md](PROGETTO_DECIDI.md)) | `decidi` | fatto, da committare; aperti al §7 |
| Diagrammi dell'architettura, nove pagine ([diagrammi/](diagrammi/)) | `decidi` | fatto, da committare |
| `ObservationEncoder` spostato in `decide/encoding.py` | `decidi` | fatto, da committare |
| Modalità `observation` dell'environment con l'oracolo, K = 12 slot dal profilo, `reset()` del decisore | `decidi` | in corso |
| Decisore student: encoder (vettore + maschera) → policy → decoder → `object_id` | `decidi` | dopo la modalità |
| Training con `MaskablePPO` e confronto con le baseline sugli stessi semi | `decidi` | dopo il decisore |
| Esito precedente nell'input di DECIDI (richiesta non eseguibile → stessa scelta ripetuta) | `decidi` | da decidere |
| `oracle_degraded`: rumore riproducibile su pose, oggetti e supporti (robustezza 1C), seme separato dalla scena | da decidere | dopo la baseline del decisore |
| ESEGUI | `esegui` | da progettare |

## Aperti, non bloccanti

- Merge di `osserva-oracolo` in `main`: lo decide Matteo.
- Detector: valutazione per fascia di sagoma visibile mai eseguita;
  benchmark `fusion_learned` da ripetere sulle stesse scene di `fusion_oracle`
  (`--objects random --min-objects 4 --max-objects 10`, stesso `--seed`).
- `pyproject.toml` dichiara Python >= 3.10 ma il codice richiede 3.12.
- Scostamenti noti di OSSERVA elencati in `observa.md`: l'encoder è stato
  spostato in `decide/` il 24/09; resta il seme dell'osservatore non separato.
