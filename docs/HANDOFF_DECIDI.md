# Handoff — progettazione e implementazione di DECIDI

Documento di passaggio per la prossima sessione (persona o IA) che lavora sul
decisore. Scritto il 23/09/2026 alla fine della sessione che ha preparato il
branch `decidi`. Per lo stato aggiornato vale sempre
[PIANO_LAVORO.md](PIANO_LAVORO.md).

---

## 1. Da leggere prima di tutto

| Ordine | File | Perché |
|---|---|---|
| 1 | `CLAUDE.md` | regole del repository per le IA |
| 2 | questo file | dove siamo e cosa resta da decidere |
| 3 | [PIANO_LAVORO.md](PIANO_LAVORO.md) | stato dei passi e decisioni in vigore |
| 4 | [STRUTTURA_OSSERVAZIONE.md](STRUTTURA_OSSERVAZIONE.md) | cosa riceve DECIDI, campo per campo, e i limiti del grafo dell'oracolo |
| 5 | `MODIFICHE.md`, prime 4-5 voci | cosa è cambiato e perché |
| 6 | `observa.md`, `interfaces.md`, `decide.md` | specifiche di Matteo. **Non sono nel repository**: Matteo le allega alla chat |

---

## 2. Il progetto in breve

Tesi / tirocinio di Matteo Paolini, "Physical AI con MuJoCo". Compito:
**estrarre un bersaglio (PFM-1) da una pila**, togliendo un oggetto alla volta
finché il bersaglio si può prendere, disturbando la pila il meno possibile.

Architettura a monolite modulare: SCENE, SIMULATION, OBSERVE, DECIDE, EXECUTE,
TASK, VISION_TRAINING. Il ciclo di un passo:

```
scena → OSSERVA → Observation (+ PrivilegedState) → DECIDI → ObjectDecision
      → ESEGUI → ExecutionOutcome → TASK → reward, disturbo, fine episodio
```

Decisione del capo: **OSSERVA reale congelato** (tag `osserva-v1`). DECIDI ed
ESEGUI si sviluppano sull'osservazione esatta dell'oracolo MuJoCo.

---

## 3. Branch e commit

| Branch | Contenuto | Stato |
|---|---|---|
| `main` | versione stabile, con OSSERVA reale | non toccare |
| `osserva-oracolo` | osservatore oracolo, main dell'oracolo, `PrivilegedState` più ricco (`876eacc`, `0e4ae4f`) | pushato. **Non si unisce in `main`** finché lo dice Matteo |
| `decidi` | creato da `osserva-oracolo` | vedi sotto |

Sul branch `decidi`, al momento della scrittura, il lavoro è **non committato**.
Matteo lancia i comandi git; se non l'ha ancora fatto:

```
python main_test.py --suite tutti
```
```
git add main_pipeline.py physical_ai_mujoco tests docs MODIFICHE.md
```
```
git status
```
```
git commit -m "DECIDE: pipeline main, run_episode, teacher interface, observation structure doc"
```
```
git push
```

---

## 4. Cosa esiste già per DECIDI

| Pezzo | Dove | Cosa fa |
|---|---|---|
| `Decider` (student) | `decide/core.py` | `decide(observation) -> ObjectDecision` |
| `TeacherDecider` | `decide/core.py` | `decide(observation, privileged) -> ObjectDecision` (solo simulazione) |
| `RandomDecider`, `HighestObjectDecider`, `ImmediateTargetDecider` | `decide/core.py` | baseline banali, preesistenti |
| `PPODecider` | `decide/core.py` | teacher PPO storico sul vettore privilegiato a 17 valori per oggetto; non implementa `decide` |
| `ComponentBuilder.decider(name)` | `infrastructure/builder.py` | costruisce `random`, `highest`, `immediate_target` |
| `run_episode(env, decider, seed, on_step)` | `experiments/episode.py` | **unico ciclo** di un episodio; distingue da solo student e teacher |
| `pipeline_main.py` + `main_pipeline.py` | `experiments/`, radice | main interattivo: scene casuali, passo per passo |
| `ObservationEncoder` | `observe/encoding.py` | vettore a slot fissi (22 valori per slot + adiacenza N×N, maschera azioni). Concettualmente è di DECIDI: va spostato in `decide/` se lo si usa |
| `train_teacher.py` | `experiments/` | training PPO storico |

**Profilo in uso:** `configs/experiments/oracolo.json`, con osservatore
`oracle`, esecutore `ideal_removal` e decisore di default `random`. Il profilo
chiude l'episodio quando il target è rimosso e non quando la pila crolla.

**ESEGUI oggi è ideale:** `IdealRemovalExecutor` toglie qualsiasi oggetto
presente, **target compreso, anche se ha altri oggetti sopra**. Il costo di una
scelta sbagliata si vede solo nel disturbo.

**TASK** (`task/core.py`, parametri in `configs/phase_0b/env.json`):

- il disturbo è lo spostamento massimo in metri degli oggetti non target
  dopo una rimozione;
- si ha un crollo se il disturbo supera 0,05 m;
- la reward vale `-0,05` per ogni rimozione, `-4 × disturbo`, `+1` quando si
  rimuove il target e `-1` a ogni crollo.

Con l'esecutore ideale, quindi, `ImmediateTargetDecider` finisce sempre in un
passo. Il problema interessante è il **compromesso fra numero di rimozioni e
disturbo**, e dipende da come si definiscono il successo e l'esecutore.

---

## 5. Cosa riceve DECIDI

Dettaglio in [STRUTTURA_OSSERVAZIONE.md](STRUTTURA_OSSERVAZIONE.md), con un
esempio in `docs/esempi/oracolo_seed7/`. In sintesi:

- **`Observation`** (student) contiene tre parti:
  - `scene`: oggetti presenti, con posa, forma, dimensioni, tipo e ruolo;
  - `relations`: archi `source → target`, dove `source` sostiene `target`;
  - `uncertainty`: con l'oracolo tutto a 1,0.
- Gli oggetti rimossi spariscono dall'`Observation`, quindi N cambia a ogni passo.
- **`PrivilegedState`** (teacher) contiene anche velocità, massa, i tre attriti,
  centro di massa, oggetti rimossi e `contact_supports`.

**Limite misurato del grafo dell'oracolo.** La regola "sostiene chi ha il
centro più basso" è stata confrontata con le forze di contatto su 30 scene:

| Misura | Valore |
|---|---|
| archi totali | 147 |
| archi con il verso invertito | 19 (13%) |
| scene in cui sbaglia se il target è coperto | 6 su 30 |

Inoltre il contatto con il pavimento è scartato e gli archi sono binari.
Nell'esempio con seed 7, l'arco `object_002 → object_000` è invertito e il
target in realtà è libero. Gli script dell'analisi erano temporanei e non sono
nel repository.

---

## 6. Decisioni aperte, in ordine

1. **Il grafo dell'oracolo va corretto prima di DECIDI?** Proposta: ricavare
   il verso dalla forza verticale (`mj_contactForce`), usare come
   `relation_score` la quota di peso portata e tenere il pavimento. È una
   modifica di OSSERVA-oracolo: va deciso anche su quale branch.
2. **Qual è l'obiettivo:**
   - meno rimozioni?
   - meno disturbo?
   - nessun crollo?
   - ha senso un esecutore ideale che estrae il target anche se coperto?
3. **Quale decisore per primo.** Le opzioni sono in §6 di
   STRUTTURA_OSSERVAZIONE.md:
   - pianificatore sul grafo, come baseline;
   - PPO a slot fissi;
   - GNN;
   - teacher con privilegi.
4. Le domande di progettazione del §20 delle regole di architettura di Matteo.

**Non implementare niente prima di aver concordato con Matteo le risposte.**
Lo ha chiesto esplicitamente: DECIDI (ed ESEGUI) si progettano insieme.

---

## 7. Regole di lavoro

- **Git:** commit, push e merge li lancia Matteo; l'IA prepara i file e
  propone i comandi. Letture solo con `GIT_OPTIONAL_LOCKS=0` o leggendo i file
  in `.git/`: git lanciato dalla VM ha lasciato due volte `.git/index.lock`.
- **Mai cancellare:** si sposta in `_to_delete/` (Matteo ha negato il
  permesso di cancellazione).
- **Lingue:** codice, docstring, commenti, messaggi, test, commit e tag in
  inglese. Documentazione (`docs/`, `MODIFICHE.md`, README) in italiano. I file
  italiani esistenti si traducono solo dove si toccano.
- **`MODIFICHE.md`:** una voce per ogni modifica, in cima, con i campi Chi /
  Cosa / Perché (con numeri) / File / Verifica. Ora con
  `TZ=Europe/Rome date '+%Y-%m-%d %H:%M:%S'`. Firma:
  "Claude (<modello>, Anthropic), su richiesta di Matteo Paolini."
  Una voce già committata non si tocca (regola 8).
- **`docs/PIANO_LAVORO.md`:** si aggiorna a ogni passo.
- **Architettura:**
  - solo `simulation/` importa mujoco;
  - `observe/`, `decide/`, `contracts/` e `task/` non importano gymnasium né
    `envs` (c'è un test che lo controlla);
  - i contratti nuovi vanno in `contracts/`;
  - la fase sta nella configurazione, non nel codice;
  - ogni cosa nel suo modulo.
- **Scelta del decisore:** DECIDI restituisce **un solo** oggetto per passo.
  Lo student vede solo l'`Observation`.
- **Comandi per Matteo:**
  - sul Mac, un comando alla volta e senza commenti `#` in linea: zsh non li
    accetta;
  - lui preferisce il tasto Run di VS Code;
  - `pip` sul ROG è rotto: usare `python -m pip`.
- **Windows:** ogni `open`/`write_text` con `encoding="utf-8"`. Un test lo
  controlla.

---

## 8. Come si prova

- **Tasto Run di VS Code** (Mac, `.vscode/launch.json`, non versionato):
  - "Pipeline (scene -> OBSERVE -> DECIDE -> EXECUTE)";
  - "OBSERVE oracle main";
  - entrambe con `--seed 7`.
- **Da terminale:** `python main_pipeline.py --seed 7`, con `--profile` per
  cambiare profilo. A ogni passo:
  - Invio: passo successivo;
  - `a`: fino alla fine;
  - `o`: `Observation` e `PrivilegedState` completi;
  - `q`: ferma l'episodio.
- **Debugger:** lanciare `main_pipeline.py`, non `episode.py`. Breakpoint in
  `experiments/episode.py` alla riga 93 (`decision, given = decide(...)`):
  lì `observation` e `privileged` sono già definite.
- **Test:**
  - `python main_test.py --suite decidi`;
  - `python main_test.py --suite osserva`;
  - `python main_test.py --suite tutti`: 190 passati sul Mac con il lavoro
    di `decidi` (184 a `0e4ae4f` + 6 di `test_pipeline`).

---

## 9. Aperti, non bloccanti

- `oracle_degraded`: rumore riproducibile su pose, oggetti e supporti per la
  robustezza della fase 1C, con un seme separato. Si fa dopo la baseline del
  decisore.
- ESEGUI da progettare (branch `esegui`).
- Detector: manca la valutazione per fascia di sagoma visibile; il benchmark
  `fusion_learned` va ripetuto sulle stesse scene di `fusion_oracle`.
- `pyproject.toml` dichiara Python >= 3.10, ma il codice richiede 3.12.
- `pfm_1_seg_immersed_v2.pt` esiste solo sul Mac.
- Non tracciati sul Mac: `Claude outputs/` e `training_sim_nodr_v1_m3.json`.
