# Piano di lavoro

Cosa si sta facendo, in quale branch, cosa e' finito e cosa resta aperto.
Si aggiorna a ogni passo, insieme a `MODIFICHE.md` (che invece dice cosa e'
cambiato e perche'). Ultimo aggiornamento: 23/09/2026.

## Decisioni in vigore

- **OSSERVA reale congelato** (tag `osserva-v1`, stato in
  [OSSERVA_STATO.md](OSSERVA_STATO.md)). DECIDI ed ESEGUI si sviluppano
  sull'osservazione dell'oracolo MuJoCo (profilo `configs/experiments/oracolo.json`).
- **DECIDI restituisce un solo oggetto** per passo (`ObjectDecision`), non una lista.
- **Teacher e student separati**: lo student riceve solo `Observation`; il
  teacher anche `PrivilegedState` (solo simulazione).
- **Un branch per componente**; `osserva-oracolo` per ora **non** si unisce in `main`.
- Codice, test, commit e tag in inglese; documentazione in italiano.

## Stato

| Passo | Branch | Stato |
|---|---|---|
| Osservatore oracolo (grafo dai contatti via `PrivilegedState`) | `osserva-oracolo` | fatto (`876eacc`) |
| Main dell'oracolo: `Observation`, `PrivilegedState`, grafo DOT/SVG, JSON | `osserva-oracolo` | fatto, da committare |
| `PrivilegedState` piu' ricco: centro di massa, tre attriti, forma e dimensioni | `osserva-oracolo` | fatto, da committare |
| Main generale scena → OSSERVA → DECIDI → ESEGUI → TASK, componenti dal profilo; `run_episode` | `decidi` (da `osserva-oracolo`) | prossimo |
| Interfacce student (`decide(observation)`) e teacher (`decide(observation, privileged)`) | `decidi` | prossimo |
| Progettazione del decisore (le 12 domande del §20 delle regole) | `decidi` | da fare |
| `oracle_degraded`: rumore riproducibile su pose, oggetti e supporti (robustezza 1C), seme separato dalla scena | da decidere | dopo la baseline del decisore |
| ESEGUI | `esegui` | da progettare |

## Aperti, non bloccanti

- Merge di `osserva-oracolo` in `main`: lo decide Matteo.
- Detector: valutazione per fascia di sagoma visibile mai eseguita;
  benchmark `fusion_learned` da ripetere sulle stesse scene di `fusion_oracle`
  (`--objects random --min-objects 4 --max-objects 10`, stesso `--seed`).
- `pyproject.toml` dichiara Python >= 3.10 ma il codice richiede 3.12.
- Scostamenti noti di OSSERVA elencati in `observa.md` (encoder in `observe/`,
  seme dell'osservatore non separato, ...).
