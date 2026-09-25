# Registro delle modifiche

Ogni modifica al codice di questo progetto va annotata qui. Il registro serve a
rispondere, mesi dopo, a una domanda sola: **perché questa riga è così?**

---

## ⚠️ ISTRUZIONE PERMANENTE — leggere prima di toccare il codice

> **A chiunque modifichi questo repository, persona o modello linguistico:**
>
> **Aggiornare questo file fa parte della modifica, non è un passo successivo.**
> Una modifica al codice senza la sua voce qui è una modifica incompleta, anche
> se i test passano.
>
> Regole:
>
> 1. **Una voce per ogni cambiamento di comportamento.** Refactoring che non
>    cambia nulla di osservabile: una riga basta. Un parametro, una soglia, una
>    regola: voce completa.
> 2. **Le voci nuove vanno in cima**, subito sotto «Registro». Le vecchie non si
>    modificano e non si cancellano mai — se una decisione viene ribaltata, si
>    scrive una voce nuova che la supera e si cita quella vecchia.
> 3. **Il campo obbligatorio è «Perché».** Il «cosa» si ricostruisce dal diff;
>    il «perché» no, ed è l'unica cosa che questo file conserva davvero. Scrivere
>    il problema osservato, non l'intenzione: *«su 25 scene il target restava
>    scoperto 9 volte»*, non *«migliorata la generazione»*.
> 4. **Data e ora al secondo**, fuso `Europe/Rome`. Da terminale:
>    `TZ=Europe/Rome date '+%Y-%m-%d %H:%M:%S'`
> 5. **Chi**: nome della persona, oppure nome e modello dell'IA più la persona
>    che l'ha istruita. Un'IA non firma mai come se fosse l'autore umano.
> 6. **Verifica**: come si è stabilito che la modifica funziona. «34 test
>    passati» è una verifica; «sembra funzionare» non lo è. Se non è stata
>    verificata, scriverlo — è un'informazione utile, nasconderlo no.
> 7. **Niente voci a posteriori in blocco.** Si annota mentre si lavora.
> 8. **Una voce gia' committata non si tocca piu'.** Correzioni e ripensamenti
>    diventano voci nuove (regola 2). Prima del commit si possono solo
>    completare i campi — tipicamente la «Verifica» con i risultati arrivati
>    dopo — oppure riscrivere la voce, **dichiarandolo nel campo «Chi»** con
>    il motivo. La data in testa resta quella della prima stesura.

---

## Convenzioni

Ogni voce ha sei campi:

| Campo | Contenuto |
|---|---|
| **Data e ora** | `AAAA-MM-GG HH:MM:SS`, fuso `Europe/Rome` |
| **Chi** | chi ha materialmente scritto la modifica, e su richiesta di chi |
| **Cosa** | il cambiamento, in termini di comportamento osservabile |
| **Perché** | il problema che lo ha causato, con i numeri se ci sono |
| **File** | i file toccati |
| **Verifica** | come si è stabilito che funziona |

Documenti collegati, che restano la fonte per le decisioni di progetto:
`README_oa.md` per come si usa il codice, e il registro delle decisioni nei
documenti di tesi (`02_decisioni.md`) per il **perché architetturale**. Qui si
annota il perché *tecnico*: le due cose non vanno confuse.

---

# Registro

## 2026-09-25 15:00:49 — Documento di progetto di DECIDI

**Chi:** Claude (claude-opus-5-5, Anthropic), su richiesta di Matteo Paolini.

**Cosa:** Branch `decidi`. Solo documentazione. Nuovo `docs/PROGETTO_DECIDI.md`:
confini dei moduli, ciclo di un passo, contratti (`DecisionInput`,
`DecisionContext`, `FeasibilityAssessment`, `ExecutionOutcome`), struttura
interna di DECIDI (encoder con maschera, K = 12 slot, decoder, `reset()`),
reward attuale, training e valutazione, questioni aperte, ordine di
implementazione. Aggiornati `docs/PIANO_LAVORO.md` (decisioni, stato, aperti)
e `docs/HANDOFF_DECIDI.md` (rimando). In testa a `PROGETTO_DECIDI.md`, il
rimando alle decisioni canoniche D47 e D48 (`software_architecture/02_decisioni.md`
v22) e la nota che `ObjectDecision` è l'`ObjectPlan` delle regole di
architettura con lunghezza uno.

**Perché:** la progettazione del 23-25/09 era solo nella chat e nei diagrammi;
senza un documento testuale la prossima sessione avrebbe dovuto ricostruire
le scelte, come la maschera solo strutturale, o il perché di `DecisionInput`
unico e dell'assenza di `remaining_steps`.

**File:** `docs/PROGETTO_DECIDI.md` (nuovo), `docs/PIANO_LAVORO.md`,
`docs/HANDOFF_DECIDI.md`, `MODIFICHE.md`.

**Verifica:** valori della reward riletti da `task/core.py` e
`configs/phase_0b/env.json`; troncamento da `envs/__init__.py`.

---

## 2026-09-25 13:45:03 — Suite di diagrammi dell'architettura (draw.io)

**Chi:** Claude (claude-opus-5-5, Anthropic), su richiesta di Matteo Paolini.

**Cosa:** Branch `decidi`. Solo documentazione. `docs/diagrammi/architettura_progetto.drawio`,
nove pagine, che uniscono la suite di Claude e la `architettura_progetto_v2.drawio`
di Matteo (lasciata intatta accanto). Dalla v2, corrette: (1) architettura,
ora con i ritorni verso l'environment, `DecisionInput`, teacher, verso
MuJoCo → PrivilegedState → OracleObserver e legenda dei colori; (2) un passo
della pipeline con i tre tempi di osservazione (Oₜ, Fₜ,ₖ, Oₜ₊₁), fase A solo al
reset; (3) dati per la reward, con la formula e i valori reali; (4) dentro
ESEGUI, con le decisioni a rombo. Dalla suite di Claude: (5) OSSERVA dentro,
(6) DECIDI dentro, (7) contratti dati, (8) training e valutazione, (9) il BPMN
`decidi_pipeline_v6`. Palette unica; bordo tratteggiato = previsto. Due punti
segnati come "da decidere": chi fornisce Fₜ,ₖ a ESEGUI (OSSERVA o sensori del
robot) e se l'`ExecutionOutcome` arriva prima o dopo l'assestamento.

**Perché:** Matteo ha chiesto figure per capire struttura e flusso dei dati. Il
BPMN (`decidi_pipeline_v6`) mostra il processo ma non cosa contengono i dati né
cosa esiste solo in simulazione. draw.io lascia controllare la disposizione
(limite di Mermaid) e si apre in VS Code con l'estensione "Draw.io Integration".

**File:** `docs/diagrammi/architettura_progetto.drawio` (nuovo), `MODIFICHE.md`.

**Verifica:** XML valido; nomi di classi e campi controllati su
`observe/core.py`, `observe/pipeline.py`, `contracts/observation.py`,
`contracts/core.py`, `task/core.py`; anteprima di ogni pagina controllata per
sovrapposizioni.

---

## 2026-09-24 15:13:40 — `ObservationEncoder` spostato da OSSERVA a DECIDI

**Chi:** Claude (claude-opus-5-5, Anthropic), su richiesta di Matteo Paolini.

**Cosa:** Branch `decidi`. Refactoring senza cambio di comportamento:
`observe/encoding.py` diventa `decide/encoding.py` (`ObservationEncoder`,
`EncodedObservation`, `OBJECT_FEATURE_NAMES`). `observe/__init__.py` non li
esporta piu'; `decide/__init__.py` si'. Aggiornati gli import di
`envs/target_extraction.py` e `tests/test_sensor_pipeline_completion.py`.
Riga corrispondente in `docs/HANDOFF_DECIDI.md`.

**Perché:** è lo scostamento n. 1 elencato in `observa.md` v11: il Blocco 5
esclude che OSSERVA scelga feature per una rete, e `interfaces.md` assegna la
trasformazione dell'`Observation` a DECIDI. Il decisore student che stiamo per
scrivere usa l'encoder: tenerlo in `observe/` avrebbe fatto dipendere DECIDI da
un dettaglio interno di OSSERVA.

**File:** `physical_ai_mujoco/decide/encoding.py` (spostato),
`physical_ai_mujoco/decide/__init__.py`, `physical_ai_mujoco/observe/__init__.py`,
`physical_ai_mujoco/envs/target_extraction.py`,
`tests/test_sensor_pipeline_completion.py`, `docs/HANDOFF_DECIDI.md`, `MODIFICHE.md`.

**Verifica:** nessun altro riferimento a `observe.encoding` nel codice;
`from physical_ai_mujoco.decide import ObservationEncoder` funziona. Suite
completa da lanciare sul Mac (`python main_test.py --suite tutti`).

---

## 2026-09-23 16:37:12 — Documento di passaggio per DECIDI

**Chi:** Claude (claude-opus-5-5, Anthropic), su richiesta di Matteo Paolini.

**Cosa:** Branch `decidi`. Solo documentazione. `docs/HANDOFF_DECIDI.md`: stato
dei branch, cosa esiste per DECIDI (interfacce, baseline, `run_episode`, main,
encoder, esecutore ideale, reward di TASK), limiti misurati del grafo
dell'oracolo, decisioni aperte in ordine, regole di lavoro, come si prova.
Riga di rimando in `docs/PIANO_LAVORO.md`.

**Perché:** Matteo apre una chat nuova per progettare il decisore; questa
sessione era lunga e già compattata. Il documento evita che la nuova sessione
riparta da zero o ripeta errori già fatti (lock di git dalla VM, cancellazioni,
commenti `#` in zsh).

**File:** `docs/HANDOFF_DECIDI.md` (nuovo), `docs/PIANO_LAVORO.md`, `MODIFICHE.md`.

**Verifica:** riferimenti controllati sul codice del branch `decidi`
(`decide/core.py`, `execute/core.py`, `task/core.py`, `configs/experiments/oracolo.json`,
riga 93 di `experiments/episode.py`).

---

## 2026-09-23 16:32:10 — Struttura di Observation e PrivilegedState documentata, con esempio e verifica del grafo dell'oracolo

**Chi:** Claude (claude-opus-5-5, Anthropic), su richiesta di Matteo Paolini.

**Cosa:** Branch `decidi`. Solo documentazione, nessun cambiamento di codice.
- `docs/STRUTTURA_OSSERVAZIONE.md`: tutti i campi di `Observation` (scena,
  relazioni, incertezza) e di `PrivilegedState`, con tipi, unità, frame, verso
  degli archi e valori prodotti dall'oracolo; tabella "chi vede cosa";
  valutazione della struttura per DECIDI; opzioni di input per DECIDI come
  domande aperte.
- `docs/esempi/oracolo_seed7/`: `observation.json`, `privileged_state.json`,
  `support_graph.dot/.svg` di una scena riproducibile (`main_test_oracle.py
  --seed 7`: seed 1390851128, 5 oggetti, target `object_002`).

**Perché:** Matteo vuole fissare, anche per la tesi, la struttura che OSSERVA
consegna a DECIDI e capire se regge prima di progettare il decisore. La
verifica con le forze di contatto (`mj_contactForce`, componente verticale;
le somme per oggetto coincidono con `m·g`) mostra che la regola dell'oracolo
"il centro più basso sostiene" inverte il verso di **19 archi su 147 (13%)**
su 30 scene casuali e sbaglia se il target è coperto in **6 scene su 30**.
Nella scena d'esempio l'arco `object_002 → object_000` è invertito: `000`
regge il 27% del peso del target e sul target non preme nulla. Il contatto
con il pavimento viene scartato e gli archi sono binari (`relation_score`
sempre 1,0). Nessuna correzione è stata fatta: la decisione è di Matteo.

**File:** `docs/STRUTTURA_OSSERVAZIONE.md` (nuovo), `docs/esempi/oracolo_seed7/`
(nuovo), `docs/PIANO_LAVORO.md`, `MODIFICHE.md`.

**Verifica:** campi confrontati con `contracts/observation.py` e
`contracts/core.py`. Forze: per ogni oggetto della scena d'esempio la somma
delle forze verticali ricevute è uguale al peso (es. `object_004`:
1,807 + 1,102 + 0,747 = 3,656 N = 0,3727 kg · 9,81). Analisi su 30 scene
eseguita sul Mac con script temporanei fuori dal repository.

---

## 2026-09-23 15:47:51 — Main generale della pipeline e interfaccia teacher di DECIDI

**Chi:** Claude (claude-opus-5-5, Anthropic), su richiesta di Matteo Paolini.

**Cosa:** Branch `decidi`.
- `main_pipeline.py` (radice, avvio leggero) e
  `physical_ai_mujoco/experiments/pipeline_main.py`: per ogni scena casuale
  (seed e 4-10 oggetti) esegue il ciclo scena → OSSERVA → DECIDI → ESEGUI →
  TASK con i componenti scelti dal profilo (default `oracolo.json`:
  osservatore oracle, decisore `default_decider`, esecutore del profilo) e
  mostra ogni passo: grafo e oggetti liberi, oggetto scelto e ruolo, esito
  dell'esecuzione, reward, disturbo, crollo, target rimosso. Chiede se aprire
  il viewer per la scena; a ogni passo: Invio, `a` fino alla fine, `o`
  `Observation` e `PrivilegedState` completi, `q` ferma l'episodio. Alla fine
  riepilogo (esito, ordine di rimozione, disturbo totale).
- `physical_ai_mujoco/experiments/episode.py`: `run_episode(env, decider)`,
  l'unico ciclo del sistema, con `StepRecord`/`EpisodeRecord`; il decisore
  riceve solo cio' che la sua interfaccia ammette.
- `decide.TeacherDecider`: `decide(observation, privileged)`, separato da
  `Decider` (student, `decide(observation)`).
- Environment: metodo pubblico `privileged_state()` (indipendente
  dall'osservatore configurato; il vecchio vettore del teacher passa da li')
  e `info["execution"]` con l'`ExecutionOutcome` di ESEGUI a ogni `step`.
- `ComponentBuilder.decider(nome)`: `random`, `highest`, `immediate_target`.

**Perche':** per progettare DECIDI serviva fissare cosa riceve e da chi. Il
ciclo esisteva solo spezzato: l'environment faceva scena, esecuzione e task,
ma DECIDI veniva chiamato da script diversi (menu 0B/1A, `politiche`), con
funzioni che ricevevano l'intero environment, e nessuno rendeva impossibile
dare lo stato privilegiato a uno student. L'esito di ESEGUI non usciva
dall'environment (solo `removed` implicito nel reward). Matteo ha chiesto un
main generale in cui entreranno anche ESEGUI e, piu' avanti, i decisori
teacher/student: un unico `run_episode` evita che training, valutazione e
visualizzazione abbiano cicli diversi.

**File:** `main_pipeline.py` (nuovo), `physical_ai_mujoco/experiments/episode.py`
(nuovo), `physical_ai_mujoco/experiments/pipeline_main.py` (nuovo),
`physical_ai_mujoco/decide/{core,__init__}.py`,
`physical_ai_mujoco/envs/target_extraction.py`,
`physical_ai_mujoco/infrastructure/builder.py`, `tests/test_pipeline.py`
(nuovo), `tests/suites.py`, `docs/PIANO_LAVORO.md`.

**Verifica:** 6 test nuovi in `tests/test_pipeline.py` (suite `decidi`):
il teacher riceve `PrivilegedState` a ogni passo e lo student no, anche se
il ramo privilegiato viene comunque registrato; ogni passo ha un
`ExecutionOutcome`, l'oggetto rimosso al passo k manca dall'osservazione del
passo k+1, il successo coincide con le regole del task; `on_step` puo'
fermare l'episodio; errore se si passa qualcosa che non e' un decisore; il
builder costruisce i decisori per nome; il main gira senza terminale su una
scena vera. Prova da terminale `python main_pipeline.py --seed 7`: 5
oggetti, 3 passi, target estratto senza crollo. Nella VM Cowork del Mac:
`test_pipeline`, `test_architecture`, `test_test_runner`,
`test_observe_pipeline`, `test_oracle_preview` 49/49; `test_phase_0b`,
`test_learned_detector`, `test_observe_menu`, `test_phase_0a` senza
rendering 62/62. Viewer e suite completa **da provare sul Mac**. Completato
prima del commit, suite completa sul Mac (macOS, conda `mujoco-tirocinio`):
`python main_test.py --suite tutti` → **190 passati, 0 falliti** in 43,0 s
(184 del commit precedente + 6 nuovi). Viewer non ancora provato.

## 2026-09-23 15:42:47 — Regola 8 del registro e piano di lavoro

**Chi:** Claude (claude-opus-5-5, Anthropic), su richiesta di Matteo Paolini.

**Cosa:** Solo documentazione. Nuova regola 8 in cima a questo file: una
voce committata non si modifica; prima del commit si possono completare i
campi o riscriverla dichiarandolo nel «Chi». Nuovo `docs/PIANO_LAVORO.md`
(cosa si fa in quale branch, cosa e' fatto, cosa resta aperto), citato nella
tabella di `CLAUDE.md`.

**Perche':** controllando il registro su richiesta di Matteo e' emerso che
due voci di oggi erano state riscritte prima del commit (oracolo e
anteprima), e solo una lo dichiarava; la regola 2 vieta di modificare le voci
vecchie ma non diceva nulla di quelle non ancora committate. Inoltre il
registro dice cosa e' cambiato ma non cosa manca: il piano del lavoro
(branch `osserva-oracolo`, poi `decidi`, poi robustezza ed ESEGUI) esisteva
solo nella conversazione.

**File:** `MODIFICHE.md`, `docs/PIANO_LAVORO.md` (nuovo), `CLAUDE.md`.

**Verifica:** nessun codice cambiato. Controllato che, rispetto al commit
`876eacc`, questo file abbia solo righe aggiunte nelle voci gia' committate.

## 2026-09-23 15:37:39 — `PrivilegedState` piu' ricco per il teacher: centro di massa, attriti, forma

**Chi:** Claude (claude-opus-5-5, Anthropic), su richiesta di Matteo Paolini.

**Cosa:** `ObjectObservation` (contratto del ramo privilegiato,
`contracts/core.py`) ha sei campi nuovi, facoltativi e in coda: `type_id`,
`shape`, `size` (ingombri x, y, z veri in metri), `center_of_mass`
(scostamento dal centro geometrico, assi del corpo), `torsional_friction`,
`rolling_friction`. `ExactObserver.privileged_state()` li riempie dalla
descrizione della scena. Il vettore storico del teacher PPO
(`as_vector()`, 17 valori per oggetto) e' invariato. Il main dell'oracolo
(`evaluation/oracle_preview.py`) li mostra nella tabella PRIVILEGED STATE
(attriti radente/torsionale/volvente, centro di massa in mm) e li salva in
`privileged_state.json`.

**Perche':** preparazione di DECIDI con teacher e student: il teacher puo'
usare il ramo privilegiato, e Matteo ha chiesto attriti, masse e centri di
massa. Il `PrivilegedState` esportava solo l'attrito radente e nessun centro
di massa, benche' la scena li randomizzi gia' (su 5 oggetti della scena di
prova, seed 7: centri di massa spostati fino a 13,5 mm, attrito volvente fra
0,001 e 0,005): un teacher non poteva sapere perche' un oggetto si ribalta.
Campi in coda con default per non cambiare i costruttori esistenti ne' il
vettore del PPO. Forze di contatto e altro si aggiungono quando il teacher ne
avra' bisogno.

**File:** `physical_ai_mujoco/contracts/core.py`, `physical_ai_mujoco/observe/core.py`,
`physical_ai_mujoco/evaluation/oracle_preview.py`, `tests/test_observe_pipeline.py`,
`tests/test_oracle_preview.py`.

**Verifica:** test nuovo
`test_privileged_state_exposes_physical_truth_without_changing_teacher_vector`
(campi riempiti dalla scena; vettore ancora 17 valori per oggetto, attrito
radente all'indice 14). Nella VM Cowork del Mac: `test_observe_pipeline`,
`test_oracle_preview`, `test_architecture`, `test_test_runner` 43/43;
`test_phase_0b` senza i test di rendering 37/37 (il vettore del teacher passa
da `privileged_state()`). Completato prima del commit, suite completa sul
Mac (macOS, conda `mujoco-tirocinio`): `python main_test.py --suite tutti` →
**184 passati, 0 falliti** in 39,5 s (177 del commit precedente + 7 nuovi).

## 2026-09-23 15:08:41 — Anteprima dell'osservatore oracolo su scene casuali

**Chi:** Claude (claude-opus-5-5, Anthropic), su richiesta di Matteo Paolini.
Voce riscritta due volte nella stessa sessione, prima del commit: la prima
stesura descriveva uno script in `evaluation/`; Matteo lo ha voluto come main
del modulo in `observe/`, poi con grafo dei supporti e `PrivilegedState` oltre
all'`Observation`. La data e' quella della prima stesura.

**Cosa:** Nuovo main della pipeline sintetica di OSSERVA,
`physical_ai_mujoco/observe/main_test_oracle.py`, accanto al laboratorio
della percezione `main_test_osserva.py` e con la stessa struttura: il main e'
un avvio leggero, la logica sta in `physical_ai_mujoco/evaluation/oracle_preview.py`
(in inglese), eseguibile anche direttamente (pulsante Run dell'editor).
Per ogni scena, seed e numero di oggetti casuali (4-10), environment costruito
dal profilo `configs/experiments/oracolo.json`, e le **due uscite di OSSERVA**
in simulazione:
- `Observation` per DECIDI, validata con `validate_observation`: oggetti
  (tipo, ruolo, posa, dimensioni), grafo dei supporti, oggetti solo sul
  pavimento, oggetti liberi sopra, cosa poggia sul target direttamente o
  attraverso altri oggetti, riepilogo dell'incertezza;
- `PrivilegedState` (solo simulazione: teacher, oracolo, valutazione):
  massa, attrito, velocita' residua, presenza, target, e controllo che le
  coppie di contatto coincidano con il grafo dell'`Observation`.
Entrambe vengono salvate in `outputs/observe_tests/oracle_<data_ora>/scene_NNN/`
(`observation.json`, `privileged_state.json`) insieme al grafo dei supporti
(`support_graph.dot` e, se c'e' Graphviz, `support_graph.svg`: dal basso verso
l'alto, target in giallo, oggetti liberi sopra bordati di verde). Poi chiede se
aprire l'immagine del grafo, se aprire il viewer MuJoCo (target evidenziato,
finestra reattiva finche' non si preme Invio) e se passare a una nuova scena.
`--seed` ripete la sequenza, `--output` cambia la cartella. Senza terminale
interattivo elabora una scena ed esce. Registrato nella suite `osserva` di
`tests/suites.py`.

**Perche':** l'unico test di OSSERVA era ancora il laboratorio della
percezione congelata (`main_test_osserva.py`), che chiede profilo, sorgenti,
disturbi, numero di scene e oggetti e confronta cinque sorgenti. Serviva un
main che producesse solo cio' che DECIDI e il teacher riceveranno, cioe'
`Observation` e `PrivilegedState`, e che li lasciasse su file: la prima
versione di questa anteprima stampava solo l'`Observation` e Matteo ha
chiesto esplicitamente anche il grafo e lo stato privilegiato. Il main sta in
`observe/`, come chiesto da Matteo; la logica sta in `evaluation/` perche'
costruisce l'environment (Gymnasium), che i file di `observe/` non importano
(`test_contract_and_policy_modules_do_not_import_backend_or_environment`): e'
lo stesso schema del vecchio main, che richiama `evaluation/observe_benchmark.py`.

**File:** `physical_ai_mujoco/observe/main_test_oracle.py` (nuovo),
`physical_ai_mujoco/evaluation/oracle_preview.py` (nuovo),
`tests/test_oracle_preview.py` (nuovo), `tests/suites.py`, `docs/OSSERVA_STATO.md`.

**Verifica:** 6 test nuovi: sequenza di scene riproducibile e numero di
oggetti nell'intervallo; descrizione di una pila a tre (target → middle →
top); ramo privilegiato mostrato a parte e coerente con il grafo; file JSON e
grafo DOT scritti e rileggibili; il main di OSSERVA avvia l'anteprima;
esecuzione non interattiva su una scena MuJoCo vera con seed 7. Nella VM
Cowork del Mac: `test_oracle_preview`, `test_observe_pipeline`,
`test_architecture`, `test_test_runner` 42/42. Grafo della scena di prova
(seed 7) reso in PNG e controllato a vista. Il viewer e l'apertura
dell'immagine non sono verificabili nella VM (niente display). Completato
prima del commit: Matteo ha provato il main e il viewer sul Mac ("il main
osserva oracolo funziona"); l'apertura dell'immagine del grafo resta da
provare. Suite completa sul Mac: 184 passati, 0 falliti in 39,5 s.

## 2026-09-23 14:57:54 — Regola della lingua: codice e messaggi git in inglese

**Chi:** Claude (claude-opus-5-5, Anthropic), su richiesta di Matteo Paolini.

**Cosa:** Solo documentazione. `CLAUDE.md` (sezione 2) fissa la lingua:
codice, test, messaggi di commit e di tag in inglese; i file esistenti si
traducono quando vengono modificati; documentazione in italiano. L'esempio
di commit in `docs/SINCRONIZZAZIONE.md` e' ora in inglese.

**Perche':** regola stabilita da Matteo il 23/09 insieme alle regole di
architettura del progetto; non era scritta da nessuna parte, quindi un'IA o
una persona nuova sul repository non poteva saperla.

**File:** `CLAUDE.md`, `docs/SINCRONIZZAZIONE.md`.

**Verifica:** nessun codice cambiato, nessun test da rilanciare.

## 2026-09-23 14:40:50 — Osservatore oracolo: grafo dei supporti dai contatti MuJoCo

**Chi:** Claude (claude-opus-5-5, Anthropic), su richiesta di Matteo Paolini.
Voce riscritta nella stessa sessione, prima del commit, dopo il confronto con
`observa.md` v11 e le regole di architettura (codice in inglese).

**Cosa:** Nuovo `OracleObserver` (`observe/core.py`), selezionabile con
`"observer": "oracle"` e usato dal nuovo profilo `configs/experiments/oracolo.json`.
Scena e incertezza identiche a `ExactObserver`; le relazioni fisiche vengono da
`OracleRelationEstimator` (`observe/pipeline.py`, il provider "simulation-only"
previsto dal Blocco 3 di `observa.md`), che copia nel contratto
`PhysicalRelationState` le coppie di appoggio (sotto → sopra, tipo `support`,
score 1,0, estimator `mujoco_contacts_oracle`). Le coppie arrivano dal ramo
privilegiato: `PrivilegedState` ha un nuovo campo `contact_supports`
(default vuoto) riempito da `ExactObserver.privileged_state()` con
`simulator.support_graph()`, solo fra oggetti presenti, pavimento escluso.
`PipelineObserver` ha un punto unico `_estimate_relations()` che l'oracolo
ridefinisce. Codice nuovo e parti toccate in inglese. Nuovo
`docs/OSSERVA_STATO.md` (stato della percezione al congelamento); rimandi in
`docs/ARCHITECTURE.md` e `README_oa.md`.

**Perche':** con il tutor si e' deciso di congelare la percezione reale e
sviluppare DECIDI ed ESEGUI su un'osservazione di riferimento. Quella
esistente (`ExactObserver`) ha pose esatte ma supporti stimati da una regola
geometrica: nel benchmark `outputs/observe_tests/20260923_135955` (4 scene)
la precisione dei supporti con pose esatte era 0,43, e su tre scene di prova
(seed 3, 7, 11) la regola trova 14, 12 e 6 candidati contro 5, 3 e 3 appoggi
veri. Il simulatore aveva gia' il grafo dei contatti (`support_graph()`,
usato solo come verita' del benchmark). I contatti sono verita' del
simulatore: per `observa.md` stanno in `PrivilegedState`, che "puo'
alimentare la costruzione di un oracle delle relazioni", e non nei prodotti
deployable (`SensorEvidence`). E' un grafo degli appoggi da contatto, non la
verita' causale delle dipendenze: e' dichiarato nell'estimator e nella doc.
L'oracolo si aggiunge agli osservatori esistenti: la pipeline reale resta
selezionabile dal profilo.

**File:** `physical_ai_mujoco/contracts/core.py`,
`physical_ai_mujoco/observe/{pipeline,core,__init__}.py`,
`physical_ai_mujoco/infrastructure/builder.py`,
`configs/experiments/oracolo.json` (nuovo), `tests/test_observe_pipeline.py`,
`docs/OSSERVA_STATO.md` (nuovo), `docs/ARCHITECTURE.md`, `README_oa.md`.

**Verifica:** 6 test nuovi in `tests/test_observe_pipeline.py`:
`PrivilegedState` porta solo appoggi fra oggetti presenti; relazioni prese dai
contatti anche quando la geometria direbbe il contrario; scena e incertezza
identiche a `ExactObserver`; errore esplicito se all'estimatore non arriva un
`PrivilegedState`; su una scena MuJoCo vera (seed 3) il grafo coincide con
`support_graph()` e dopo una rimozione non contiene piu' l'oggetto tolto; il
profilo `oracolo.json` costruisce un `OracleObserver`. Nella VM Cowork del Mac
(Linux aarch64, Python 3.12.14, MuJoCo 3.14): `test_observe_pipeline`,
`test_architecture`, `test_phase_0a`, `test_learned_detector`,
`test_observe_menu`, `test_test_runner` 68/68; `test_phase_0b` 32 passati, 1
fallito per rendering OpenGL assente nella VM (`test_frame_capture_returns_frames`).
Suite completa sul Mac (macOS, conda `mujoco-tirocinio`, Python 3.12):
`python main_test.py --suite tutti` → **177 passati, 0 falliti** in 40,3 s
(171 di prima + 6 nuovi).

## 2026-09-23 13:19:14 — Testo sempre in UTF-8: il benchmark OSSERVA falliva su Windows

**Chi:** Claude (claude-opus-5-5, Anthropic), su richiesta di Matteo Paolini.

**Cosa:** `encoding="utf-8"` aggiunto a tutte le 73 chiamate `read_text()` /
`write_text()` che non lo avevano, in 23 file (`physical_ai_mujoco/`,
`scripts/`, `tools/`, `tests/`), e alle chiamate `subprocess.run(..., text=True)`
che leggono l'output di git (`experiments/metadata.py`,
`vision_training/training.py`, `tools/archivio_drive`). Nuovo test
`test_text_files_are_read_and_written_as_utf8` in `tests/test_architecture.py`:
fallisce se una `read_text`/`write_text` o un `open()` in modo testo non indica
l'encoding. Le modifiche sono state fatte da uno script che inserisce solo
l'argomento, usando le posizioni dell'AST, e ricontrolla che ogni file resti
Python valido.

**Perche':** prima suite completa sul ROG (Windows 11, Python 3.12.6): 168
passati, 2 falliti, entrambi in `tests/test_simulated_stereo.py`
(`test_benchmark_compares_all_sources_without_decide_or_execute`,
`test_benchmark_records_invariant_failure_and_continues`) con
`UnicodeEncodeError: 'charmap' codec can't encode character '\u2192'` in
`observe_benchmark.export_graphs`. Senza `encoding` Python usa la codifica
del sistema: UTF-8 su Mac e Linux, cp1252 su Windows, che non ha la freccia
dei grafi `.dot`. Lo stesso difetto era in altre 71 chiamate: su Windows i
file UTF-8 scritti dal Mac (es. `"1B · Policy"` nei profili di
`configs/experiments/`) vengono letti storpiati senza errore, e il codice
scritto sul Mac lo reintrodurrebbe senza accorgersene: da qui il test.

**File:** `physical_ai_mujoco/evaluation/{observe_benchmark,sensor_calibration,sensor_validation}.py`,
`physical_ai_mujoco/experiments/metadata.py`, `physical_ai_mujoco/infrastructure/{builder,experiment}.py`,
`physical_ai_mujoco/observe/main_test_osserva.py`, `physical_ai_mujoco/sensors/{rig,lidar/scanner}.py`,
`physical_ai_mujoco/vision_training/{dataset,evaluation,labels,recipe,training}.py`,
`scripts/verifica_osserva.py`, `tools/capture_baseline.py`, `tools/archivio_drive/archivio_drive.py`,
`tests/test_{architecture,learned_detector,observe_menu,phase_0b,sensors_extended,simulated_stereo,vision_training}.py`.

**Verifica:** nella VM Cowork del ROG (Linux, Python 3.12.14) con
`-X warn_default_encoding` ed `EncodingWarning` trasformato in errore per i
moduli del progetto: `test_architecture` 14/14 (compreso il nuovo),
`test_archivio_drive` 12/12, `test_test_runner` 3/3, `test_learned_detector`
8/8, `test_observe_menu` 18/18, `test_mesh_target` 6/6, `test_phase_0a` 6/6,
`test_sensor_validation` 1/1, `test_vision_training` 13 passati prima dei test
che fanno rendering. I test con rendering MuJoCo non girano nella VM (nessun
display OpenGL). Poi, sul ROG (Windows 11, Python 3.12.6):
`python main_test.py --suite tutti` → **171 passati, 0 falliti** in 198,8 s
(prima della correzione: 168 passati, 2 falliti).

## 2026-09-23 12:56:08 — Archivio Drive: esclusi i file `._*` del Mac, niente run a meta'

**Chi:** Claude (claude-opus-5-5, Anthropic), su richiesta di Matteo Paolini.

**Cosa:** `configs/archivio_drive.json`: `._*` aggiunto a `escludi`.
`tools/archivio_drive`: se `carica` viene interrotto (Ctrl+C) o fallisce
durante copia o zip, la cartella del run appena creata viene rimossa e nulla
entra nel registro; avanzamento stampato ogni 2.000 impronte e ogni 500 copie.
`VERSIONE_STRUMENTO` passa a 2. `.gitignore`: aggiunto `._*`.

**Perche':** il primo caricamento del dataset `sim_dr_v1` sul ROG ha contato
53.056 file invece di circa la meta': ogni file ha un gemello AppleDouble
`._nome` da 4 KB (verificato: `images/train` 2.432 immagini e 2.432 `._`,
`labels/train` idem), creato dal Mac quando il dataset e' stato copiato su un
filesystem non Apple. Sono metadati senza contenuto: sarebbero finiti nello zip
su Drive. L'unico modo di fermare il caricamento era Ctrl+C, che nella
versione 1 poteva lasciare su Drive una cartella di run incompleta e non
registrata, da cancellare a mano.

**File:** `configs/archivio_drive.json`, `tools/archivio_drive/archivio_drive.py`,
`tests/test_archivio_drive.py`, `.gitignore`.

**Verifica:** `pytest tests/test_archivio_drive.py`: 12 passati, fra cui uno
nuovo che simula Ctrl+C durante la copia (registro vuoto, nessuna cartella
rimasta) e il file `._results.csv` escluso nel test di copia.

## 2026-09-23 12:47:53 — Archivio Drive per i risultati pesanti: modulo `tools/archivio_drive/`

**Chi:** Claude (claude-opus-5-5, Anthropic), su richiesta di Matteo Paolini.

**Cosa:** Nuovo modulo `tools/archivio_drive/` (codice, `__main__.py`,
`README.md`), avviabile con `python -m tools.archivio_drive`. Archivia file e
cartelle nella cartella `physical_ai_mujoco_tesi/` di Google Drive, montata
dall'app desktop (trovata da sola su Windows, Mac e Colab; altrimenti
`--drive`/`ARCHIVIO_DRIVE`). Comandi: `carica`, `scarica`, `stato
[--verifica]`, `annota`, `categorie`, `prepara`, `readme`. Ogni caricamento e'
un run `AAAA-MM-GG_nome` con `scheda.json` (commit, branch, PC, autore, fase,
SHA-256 di ogni file, nota obbligatoria); `registro.jsonl` append-only; il
`README.md` dell'archivio e' rigenerato dal registro. Contenuto identico non
viene ricaricato, contenuto cambiato diventa `nuova_versione` senza
sovrascrivere; le categorie con migliaia di file vengono zippate. Struttura
per modulo del progetto (01_scena ... 07_dati_reali, 90_tesi) definita in
`configs/archivio_drive.json`. Nuova suite di test `archivio` in
`tests/suites.py`. Nuova guida `docs/SINCRONIZZAZIONE.md` (git su piu' PC,
cosa va dove); aggiornati `CLAUDE.md` (sezione 5 e tabella) e `README_oa.md`.

**Perche':** il training del visore del 22/09 e' stato fatto sul ROG
(RTX 3050) perche' sul Mac M3 era troppo lento; i risultati (cartella di
training 67 MB con grafici e best/last.pt, dataset `sim_dr_v1` di 3.392
immagini) non entrano in git e non c'era un modo tracciato di portarli sugli
altri PC ne' di sapere, fra qualche mese, da quale commit e ricetta venivano.
Matteo ha chiesto un sistema su Drive catalogato per modulo, con un registro
di aggiunte e modifiche. La struttura segue i moduli e non le fasi per la
stessa regola del codice (la fase vive nella configurazione): la fase e' un
campo della scheda. Solo libreria standard perche' deve girare anche fuori
dall'ambiente del progetto (Colab, PC appena clonato). I dataset vanno zippati
perche' Drive sincronizza lentamente migliaia di file piccoli.

**File:** `tools/archivio_drive/archivio_drive.py`, `tools/archivio_drive/__init__.py`,
`tools/archivio_drive/__main__.py`, `tools/archivio_drive/README.md`,
`configs/archivio_drive.json`, `tests/test_archivio_drive.py`, `tests/suites.py`,
`docs/SINCRONIZZAZIONE.md`, `CLAUDE.md`, `README_oa.md` (modificati: `tests/suites.py`,
`CLAUDE.md`, `README_oa.md`; gli altri sono nuovi).

**Verifica:** `pytest tests/test_archivio_drive.py tests/test_test_runner.py`:
14 passati (11 dell'archivio: copia, zip, doppioni, nuova versione, scarica
senza sovrascrivere, annota, verifica di file alterati, errori d'uso), sia
nell'ambiente Linux della sessione sia nella VM Cowork del ROG. **Non** e'
stata eseguita la suite completa: in quegli ambienti MuJoCo non e' installato.
Da lanciare sul ROG: `python main_test.py --suite tutti`.

## 2026-09-23 12:47:53 — Pesi del visore in git; fine-tuning reale sui pesi del ROG

**Chi:** Claude (claude-opus-5-5, Anthropic), su richiesta di Matteo Paolini.

**Cosa:** `.gitignore`: `outputs/` diventa `outputs/*` con l'eccezione
`!outputs/detector_weights/`, quindi i pesi consegnati del visore (e la loro
scheda) entrano in git; esclusi i modelli base di Ultralytics (`/yolo*.pt`,
`/weights/`). `training_real_finetune_v1.json`: `base_model` passa da
`outputs/detector_weights/sim_dr_v1_m3.pt` a `sim_dr_v1_rtx3050.pt`.
Documentata la ricetta `configs/vision_training/sim_dr_v1_rtx3050.json`
(creata sul ROG il 22/09 senza voce in questo registro) in
`docs/ADDESTRAMENTO_VISORE.md` e `physical_ai_mujoco/vision_training/README.md`,
dove anche gli esempi di valutazione e benchmark ora usano `sim_dr_v1_rtx3050.pt`.

**Perche':** il modello del visore e' stato addestrato sul ROG (RTX 3050,
batch 8 per i 4 GB di VRAM; 190 epoche in circa 7 ore; mAP50 maschere 0,84,
recall 0,78) e serve su tutti i PC, ma `outputs/` era interamente ignorato:
l'unico modo di portarlo sul Mac era a mano. I pesi sono 20 MB, sotto il
limite di GitHub; le cartelle di training e le versioni intermedie restano
fuori da git e vanno nell'archivio Drive. La ricetta di fine-tuning puntava a
`sim_dr_v1_m3.pt`, un modello mai prodotto: il fine-tuning sulle foto reali
sarebbe fallito all'avvio.

**File:** `.gitignore`, `configs/vision_training/training_real_finetune_v1.json`,
`configs/vision_training/sim_dr_v1_rtx3050.json` (gia' presente, ora da committare),
`docs/ADDESTRAMENTO_VISORE.md`, `physical_ai_mujoco/vision_training/README.md`.

**Verifica:** `git check-ignore -v`: `outputs/detector_weights/sim_dr_v1_rtx3050.pt`
non e' ignorato, `outputs/detector_training/.../results.csv`, `yolo11s-seg.pt`
e `weights/yolo26n.pt` si'. JSON della ricetta riletto correttamente. Il
fine-tuning non e' stato lanciato (manca ancora il dataset reale).

## 2026-09-22 08:45:20 — Documentato come cambiare scenario e oggetti nel README del visore

**Chi:** Claude (claude-sonnet-5, Anthropic), su richiesta di Matteo Paolini.

**Cosa:** Solo documentazione, nessun cambiamento di comportamento. Nuova
sezione "Cambiare lo scenario (oggetti, target)" in
`physical_ai_mujoco/vision_training/README.md`: differenza fra riusare tipi
gia' nel catalogo (`datasets/object_dataset/geometric_objects.json`) e
aggiungerne uno nuovo, come `required_type_ids`/`target_type_id` nella ricetta
di scena scelgono e ciclano i tipi, cosa ricontrollare se cambia il target
(`target_palette`, `minimum_visible_pixels`, `hard_visibility`), e il limite
noto (un solo target, una sola classe di ostacoli).

**Perche':** Matteo ha chiesto come adattare il sistema cambiando gli oggetti
della scena; la domanda ha mostrato che ne' il README del visore ne'
`docs/ADDESTRAMENTO_VISORE.md` spiegavano questo passaggio, nonostante sia
proprio il tipo di domanda a cui questi file devono rispondere fra un mese.

**File:** `physical_ai_mujoco/vision_training/README.md`.

**Verifica:** contenuto controllato sul codice —
`scene/dataset_loader.py` (formato e validazione del catalogo),
`simulation/session.py` (`_rules_with_object_count`: come `required_type_ids`
viene ciclato secondo l'object_count), `configs/observe_tests/
immersed_scene_rules.json` (struttura reale di una ricetta di scena). Nessun
test da rilanciare: non cambia codice.

## 2026-09-21 21:06:17 — Guida ai comandi per l'addestramento del visore

**Chi:** Claude (claude-opus-5, Anthropic), su richiesta di Matteo Paolini.

**Cosa:** Solo documentazione, nessun cambiamento di comportamento. Nuovo
`docs/ADDESTRAMENTO_VISORE.md`: prova veloce, generazione del dataset,
training (Mac M3 / GPU NVIDIA), valutazione per % di target visibile con tutte
le opzioni e come leggerne l'output, uso del modello nel benchmark di OSSERVA
(`--weights`), fine-tuning sulle foto reali, come fare un esperimento nuovo,
problemi frequenti. Una riga in `physical_ai_mujoco/vision_training/README.md`
rimanda alla guida.

**Perché:** Matteo ha chiesto un file specifico per il visore con i comandi da
lanciare e le ricette da usare: il README del pacchetto spiega il perché delle
scelte, ma per rilanciare dataset, training e valutazione fra un mese servono i
comandi esatti, le opzioni e dove finiscono i risultati in un solo posto.

**File:** `docs/ADDESTRAMENTO_VISORE.md` (nuovo),
`physical_ai_mujoco/vision_training/README.md` (una riga di rimando).

**Verifica:** comandi e opzioni confrontati con gli `argparse` di
`vision_training/dataset.py`, `training.py`, `evaluation.py`,
`evaluation/observe_benchmark.py` e con le ricette in
`configs/vision_training/`; formato dell'output di valutazione preso da
`evaluation.format_report`. Tempi di generazione dal run di `sim_dr_v1` del
2026-09-21 (1437,6 s). Nessun test da rilanciare: non cambia codice.

## 2026-09-21 20:58:21 — Documentazione del visore e collegamento con i documenti di progetto

**Chi:** Claude (claude-opus-5, Anthropic), su richiesta di Matteo Paolini.

**Cosa:** Solo documentazione, nessun cambiamento di comportamento. Nuovo
`physical_ai_mujoco/vision_training/README.md` (ciclo, mappa dei file, uso,
ricette, output, manutenzione, limiti, checklist). Aggiornati
`docs/OBSERVE_TEST.md` (ricette, menu 5, soglia di osservabilità),
`docs/SENSOR_COMPLETION_STATUS.md` (il training del detector ora esiste),
`docs/OBSERVE_IMPLEMENTATION.md` (rimandi) e `docs/ARCHITECTURE.md` (sezione
VISION_TRAINING). Fuori dal repo, con i rispettivi storici:
`software_architecture/observa.md` v11 (mappa blocchi → codice e quattro
scostamenti), `software_architecture/02_decisioni.md` v21 (D46),
`service_documents/03_stato.md` v13, `service_documents/05_tesi_appunti.md` v10.

**Perché:** Il modulo del visore non aveva documentazione d'uso, e i documenti
del repo dicevano ancora che il training del detector non era implementato.
Matteo ha chiesto di poter capire fra un mese come e' collegato, come si
lancia e come si mantiene.

**File:** vedi *Cosa*.

**Verifica:** controllati a mano i comandi e i percorsi citati nel README
contro il codice e contro i run fatti sul Mac il 21/09; nessun test da
rilanciare perche' non cambia codice.

## 2026-09-21 20:22:45 — Ricetta di fine-tuning reale: ottimizzatore esplicito

**Chi:** Claude (claude-opus-5, Anthropic), su richiesta di Matteo Paolini.

**Cosa:** `training_real_finetune_v1.json` dichiara `"optimizer": "AdamW"`.

**Perché:** Il primo training sul Mac (ricetta `smoke`, Ultralytics 8.4.156)
ha stampato «optimizer=auto found, ignoring 'lr0'»: con l'ottimizzatore
automatico il learning rate della ricetta viene scartato. Per il fine-tuning
sulle foto reali il learning rate basso (0.001) è la scelta che evita di
cancellare quanto appreso in simulazione, quindi deve valere davvero.

**File:** `configs/vision_training/training_real_finetune_v1.json`.

**Verifica:** JSON valido; la ricetta non è ancora eseguibile (manca il
dataset reale), quindi l'effetto sul learning rate non è stato osservato.

## 2026-09-21 20:19:08 — `vision_training`: la vista senza target non dipende piu' dalla randomizzazione

**Chi:** Claude (claude-opus-5, Anthropic), su richiesta di Matteo Paolini.

**Cosa:** L'estrazione "questa vista e' senza target" usa un flusso casuale
proprio (`STREAM_ABSENT`) invece di quello delle condizioni visive. Ricette che
differiscono solo per la randomizzazione tolgono ora il target dalle stesse
viste. Restano possibili differenze nelle viste extra "poco visibile", perche'
dipendono dalla visibilita' misurata, che a sua volta dipende dalla camera.

**Perche':** Sul Mac, `smoke` (con randomizzazione) aveva 0 viste senza target
nello split val, la stessa ricetta senza randomizzazione ne aveva 2: togliendo
la randomizzazione non si consumavano estrazioni e cambiava l'esito. Il
confronto scena per scena fra `sim_dr_v1` e `sim_nodr_v1` promesso nella voce
precedente non era quindi vero.

**File:** `physical_ai_mujoco/vision_training/randomization.py`,
`physical_ai_mujoco/vision_training/dataset.py`, `tests/test_vision_training.py`.

**Verifica:** nuovo test che genera la stessa ricetta con e senza
randomizzazione (3 scene, 50% di viste senza target) e confronta la presenza
del target immagine per immagine; suite completa: 158 test passati
(container Linux, Python 3.12.3, MuJoCo 3.13.0).

## 2026-09-21 17:47:25 — Modulo `vision_training`: addestramento del visore da ricette, domain randomization e soglia di osservabilità misurata

**Chi:** Claude (claude-opus-5, Anthropic), su richiesta di Matteo Paolini.

**Cosa:**

- Nuovo pacchetto `physical_ai_mujoco/vision_training/` che sostituisce
  `experiments/synthetic_dataset.py` e `experiments/train_detector.py` (rimossi,
  il codice è stato spostato, non duplicato). Tutto parte da **ricette JSON** in
  `configs/vision_training/`: `dataset_*.json` (scene, split, randomizzazione) e
  `training_*.json` (modello di partenza, epoche, device). Le chiavi sconosciute
  sono un errore.
- **Dataset**: split per scena in train/val/**test** (il test non è mai visto in
  addestramento); più viste per scena; una quota di viste senza target
  (negativi); viste extra quando il target è poco visibile; la **% di sagoma
  visibile** del target è scritta in ogni annotazione, sempre. Le annotazioni
  JSON su disco ora contengono tutti i campi (prima solo il manifest li aveva).
- **Domain randomization** (solo se la ricetta la chiede): camera (posizione,
  punto guardato, rollio, fov), luce principale e headlight, terreno (scacchi,
  tinta unita, macchie a rumore; riflessione), sfondo, colore degli ostacoli e
  del target (tavolozza PFM-1), disturbi immagine su tutte le immagini
  (contrasto, luminosità, gamma, rumore, blur, vignettatura). Ogni aspetto ha
  il suo flusso casuale, derivato da (seed, scopo, scena, vista).
- `simulation/visual_conditions.py` (solo dati) e
  `Simulator.apply_visual_conditions(...)`: l'aspetto cambia sul modello già
  compilato; fisica e maschere non cambiano. `stereo_calibration()` legge ora
  il fov dal modello (identico al nominale senza randomizzazione).
- `Simulator.hold_object/release_object` e
  `place_target_at_immersion(..., settle_others=True)`: dopo aver interrato il
  target, il resto della pila ricade e si assesta. Il default resta `False`.
- `evaluation/target_exposure.measure_target_visibility`: pixel visibili /
  pixel del target da solo e non interrato, nella stessa posa, senza modificare
  la scena (compenetrazione di contatto sotto 2 mm non conta come interramento).
- **Training** (`vision_training/training.py`): device automatico CUDA → MPS →
  CPU; `data.yaml` riscritto con il percorso vero al momento del training;
  **scheda del modello** `<nome>.json` sempre scritta accanto ai pesi (ricetta,
  riassunto dataset, modello genitore, versioni, commit git, metriche); rifiuta
  di sovrascrivere un esperimento. Il modello di partenza può essere un nostro
  `.pt` (fine-tuning sulle immagini reali: `training_real_finetune_v1.json`).
- **Valutazione** (`vision_training/evaluation.py`): richiamo del target per
  fasce di % visibile, falsi positivi sulle viste senza target, richiamo per
  numero di oggetti e **% minima riconoscibile** (fascia più bassa da cui in su
  il richiamo è ≥ 80%, IoU ≥ 0.5). Il valore finisce nella scheda.
- **Benchmark OSSERVA**: `target_observable` non è più "almeno un pixel del
  target in immagine", ma `% visibile ≥ soglia`. La soglia viene dalla scheda
  del modello se misurata, altrimenti dal profilo
  (`target_observable_min_visible_fraction: 0.05`), altrimenti dal default
  0.05; il report dice quale. La curva per fasce di visibilità è ora popolata
  in ogni benchmark, non solo negli esperimenti di immersione.
- Menu del laboratorio OSSERVA: 2 e 3 scelgono una ricetta; nuova voce 5
  "Valuta il detector per % di target visibile". Nuova suite di test `visore`.

**Perché:**

- Il benchmark `fusion_learned` del 2026-09-21 16:46 (4 scene, 3–8 oggetti)
  dava richiamo del target 25% e F1 supporti 0.30. Il modello attivo
  `pfm_1_seg_immersed_v2.pt` era stato addestrato su 80 scene, 346 immagini,
  **1–6 oggetti**: le due scene in cui il target non è mai stato trovato avevano
  7 e 8 oggetti, fuori distribuzione.
- `target_observable` era `truth.any()`: uno spiraglio di un pixel bastava a
  contare come "osservabile" un target che a occhio era sepolto (scena 1 dello
  stesso run). La misura più corretta esisteva (`camera_visible_fraction`) ma
  si calcolava solo negli esperimenti di immersione: nel benchmark generico la
  curva per fasce era vuota (tutte le fasce con `scene_count: 0`).
- Il modello attivo non aveva scheda (`.json` assente, cartella di training
  assente) e il suo `data.yaml` puntava a
  `/Users/matteopaolini/Documents/Codex/.../pfm_1_dataset_immersed_v2`: non era
  né riaddestrabile così com'era né ricostruibile dal menu, che chiedeva solo
  scene, oggetti e seed.
- Nessuna randomizzazione visiva: camera fissa (stessa posa, fov 50°), una sola
  luce, pavimento sempre a scacchi, target sempre dello stesso grigio. Per il
  sim2real (pre-training in simulazione, poi fine-tuning su foto reali) il
  detector non deve poter imparare lo sfondo o una tonalità fissa.
- Interrando il target senza fisica, gli oggetti che gli poggiavano sopra
  restavano sospesi e le loro ombre indicavano dove stava il target.

**File:** `physical_ai_mujoco/vision_training/` (nuovo: `__init__`, `recipe`,
`randomization`, `labels`, `dataset`, `training`, `evaluation`),
`physical_ai_mujoco/simulation/visual_conditions.py` (nuovo),
`physical_ai_mujoco/simulation/simulator.py`,
`physical_ai_mujoco/evaluation/target_exposure.py`,
`physical_ai_mujoco/evaluation/observe_benchmark.py`,
`physical_ai_mujoco/observe/main_test_osserva.py`,
`physical_ai_mujoco/experiments/synthetic_dataset.py` (rimosso),
`physical_ai_mujoco/experiments/train_detector.py` (rimosso),
`scripts/genera_dataset_target.py`, `scripts/train_detector.py`,
`configs/vision_training/` (nuovo, 7 ricette),
`configs/observe_tests/{clean,fixed,random}.json`,
`tests/test_vision_training.py` (nuovo), `tests/test_learned_detector.py`,
`tests/test_observe_menu.py`, `tests/test_architecture.py`, `tests/suites.py`,
`tests/README.md`.

**Verifica:**

- Suite completa: **157 test passati** (prima delle modifiche: 139), in un
  container Linux con Python 3.12.3, MuJoCo 3.13.0 e rendering OSMesa. I test
  di Fase 0A passano invariati.
- Nuovi test architetturali: `vision_training/` non importa `mujoco`;
  `visual_conditions.py` è solo dati.
- Generazione della ricetta `dataset_smoke` (6 scene, 2–6 oggetti): 9 s, 3
  split, immagini ispezionate a vista.
- Benchmark `fusion_oracle` su 3 scene: curva per fasce popolata (nascosto,
  70–80%, 90–100%) e soglia riportata con la sua fonte.
- **Non verificato:** training reale con Ultralytics/PyTorch (non installabile
  nel container: `download.pytorch.org` bloccato dal proxy; il training è
  testato con un finto YOLO), training su MPS del Mac M3, tempi del dataset
  completo `sim_dr_v1` (800 scene), comportamento con la versione di MuJoCo
  dell'ambiente conda `mujoco-tirocinio`.

## 2026-09-21 11:26:00 — Menu test modulare, scena riproducibile e diagnostica CAD

**Chi:** Codex (GPT-5), su richiesta di Matteo Paolini.

**Cosa:** Aggiunto `main_test.py` con moduli logici scena, OSSERVA,
validazione sensori, pipeline dati, DECIDI e architettura. I nuovi test devono
essere assegnati in `tests/suites.py`. Il test visivo OSSERVA chiede ora numero
di scene, numero di oggetti e seed: Invio sul seed genera una nuova scena;
un numero la riproduce. Aggiunta una diagnostica CAD su 24 pose e cinque
frazioni di superficie sintetica con report JSON. L'encoder ricicla slot
obsoleti e segnala overflow come spazio incerto. La valutazione PPO usa gli
slot sensoriali quando richiesto; un adapter separato carica i modelli PPO
sensoriali e le loro statistiche di normalizzazione. Testata una detection
disturbata senza usare il ground truth per scartare falsi positivi.

**Perché:** Il menu di OSSERVA fissava una sola scena da tre oggetti e il seed
del profilo restava 42, dando l'impressione di una scena immutabile. La
copertura dei test era difficile da avviare per area. Gli slot potevano
esaurirsi con ID detector variabili. Un fit CAD accettato non dimostrava posa
corretta: nella diagnostica al 10% di superficie l'errore del centro e 3,7 cm
mediani, nonostante 24/24 fit accettati. Il rapporto dimensionale e 1 per
costruzione quando il CAD viene accettato.

**File:** `main_test.py`, `tests/`, `physical_ai_mujoco/evaluation/`,
`physical_ai_mujoco/observe/`, `physical_ai_mujoco/experiments/train_teacher.py`,
`physical_ai_mujoco/infrastructure/policy_adapter.py`, `MODIFICHE.md`.

**Verifica:** 139 test passati nel progetto Desktop, con accesso al renderer
MuJoCo; 20 warning non bloccanti. Diagnostica CAD eseguita su 120 combinazioni.
Non eseguito training PPO, ne validazione con LiDAR reale o detector appreso.

## 2026-09-20 17:46:01 — Pipeline sensoriale collegata a tracking, PPO e CAD PFM-1

**Chi:** Codex (GPT-5), su richiesta di Matteo Paolini.

**Cosa:** Il detector learned assegna ID temporali tramite `ObjectTracker`;
`ObservationEncoder` converte scena, incertezza e relazioni in slot e vettore
PPO a dimensione fissa. L'environment accetta `obs_mode="sensor"`, acquisisce
uno `SynchronizedSensorPacket`, espone maschere di azione sensoriali e associa
l'azione scelta al corpo simulato solo nell'adapter di esecuzione. Le fasi 1B e
1C sono selezionabili: 1C applica disturbi fotometrici e LiDAR a runtime. La
PFM-1 usa la STL per un allineamento CAD parziale con PCA, ICP trimmed e gate
sull'errore; un fit rifiutato ricade sulla geometria conservativa visibile. Il
benchmark registra rapporti dimensionali lineari e volumetrici, anche per
fascia di esposizione. `Observer.observe` usa ora un unico contratto
`(source, TaskContext)`. La vecchia catena `SensorBundle` emette una
`DeprecationWarning` ed e limitata ai benchmark stereo-only/RGB-D.

**Perché:** Gli ID `learned_###` dipendevano dall'ordine delle predizioni, PPO
non poteva consumare un numero variabile di oggetti, la dimensione della mina
interrata era quella della sola superficie visibile e i profili `fixed/random`
non modificavano la fusione runtime. Inoltre l'environment dichiarava un
observer iniettabile ma lo chiamava con una firma incompatibile.

**File:** `physical_ai_mujoco/{observe,sensors,envs,infrastructure,evaluation,
experiments,ui}/`, `configs/{experiments,observe_tests}/`, `tests/`,
`README.md`, `docs/`, `MODIFICHE.md`.

**Verifica:** 131 test automatici passati, inclusi rendering MuJoCo. Smoke test
end-to-end `fusion_oracle` e `fusion_learned` completati con invarianti validi
e report JSON. La singola scena di smoke aveva recall target 0: conferma che il
collegamento funziona, non costituisce validazione statistica del detector o
del LiDAR e non viene presentata come tale.

## 2026-09-18 16:09:41 — Un solo percorso per camera, bundle e ricostruzione

**Chi:** Codex (GPT-6), su richiesta di Matteo Paolini.

**Cosa:** La camera simulata configura la depth alla costruzione e alimenta
anche le osservazioni stereo di Gymnasium. `BundleBuilder` e `DetectionNoise`
gestiscono ora anche il rumore sui centroidi. `StereoObserver` seleziona
esplicitamente la ricostruzione `stereo` o `depth`. Sono state eliminate le
classi e i file equivalenti `RGBDCapture`, `OracleStereoLabels`,
`StereoDegradation`, `RGBDObserver`, `AdaptiveStereoExtractor`,
`StereoCapture`, il re-export `simulated_stereo.py` e il wrapper di calibrazione
LiDAR senza logica propria.

**Perché:** Il modulo aveva due camere, due assemblatori di `SensorBundle`, due
configurazioni sovrapposte dei disturbi, un dispatcher implicito e due observer
distinti soltanto per la gestione della depth. Inoltre l'environment aggirava
la camera pubblica con una seconda implementazione di cattura. Queste
duplicazioni rendevano ambiguo quale percorso fosse quello effettivo.

**File:** `physical_ai_mujoco/sensors/`, `physical_ai_mujoco/observe/`,
`physical_ai_mujoco/envs/target_extraction.py`,
`physical_ai_mujoco/evaluation/observe_benchmark.py`, `tests/`, `README.md`,
`docs/SENSORS.md`, `docs/OBSERVE_TEST.md`.

**Verifica:** compilazione Python completata; 59 test mirati e tutti gli 86
test della suite passati. Restano due warning Gymnasium preesistenti sui limiti
infiniti dello spazio `Box`.

## 2026-09-18 15:34:43 — Test OSSERVA avviabili direttamente dalla loro cartella

**Chi:** Codex (GPT-6), su richiesta di Matteo Paolini.

**Cosa:** I tre test `test_sensors_extended.py`, `test_simulated_stereo.py` e
`test_observe_pipeline.py` inseriscono la radice del repository nel percorso
Python quando sono lanciati come script e avviano pytest; il README del modulo
documenta questa modalita.

**Perché:** Il comando diretto dell'utente importava `physical_ai_mujoco` da
`physical_ai_mujoco_internship copia` invece che da `mujoco_deploy`, causando
`ImportError` per `TaskContext` e `DegradedObserver`. Inoltre, senza un
`pytest.main`, un file di test avviato con Python non eseguirebbe i test.

**File:** `tests/test_sensors_extended.py`, `tests/test_simulated_stereo.py`,
`tests/test_observe_pipeline.py`, `physical_ai_mujoco/sensors/README.md`.

**Verifica:** Ripetuti i tre avvii diretti con Python: rispettivamente 13,
8 e 11 test passati nel working copy; segue verifica nella cartella Desktop.

## 2026-09-18 15:21:55 — Sensori stereo, RGB-D e LiDAR con calibrazione verificabile

**Chi:** Codex (GPT-6), su richiesta di Matteo Paolini.

**Cosa:** Aggiunti rig geometrico indipendente, acquisizione depth, detector
sostituibile e oracle isolato, disturbi riproducibili, nuvola di punti e stima
geometrica, bundle con provenienza, LiDAR 2D/3D con test autonomo, perturbazione
e stima extrinsics, confronto LiDAR-depth e curva di calibrazione. OSSERVA
accetta un bundle RGB-D e il benchmark espone la nuova modalita `rgbd`,
dimensioni e forma, mantenendo il percorso stereo senza depth.

**Perché:** Sul report di 3 scene da 10 oggetti la stereo disturbata trovava
solo 4/10 oggetti in media e non produceva relazioni, perché non forniva size;
l'errore di acquisizione non era separabile da quello di ricostruzione.
Servivano inoltre una sorgente LiDAR e prove di calibrazione richieste dal
progetto, senza introdurre import MuJoCo fuori da `simulation/`.

**File:** `physical_ai_mujoco/sensors/` (rig, capture, detector, noise,
cloud, bundle, lidar, calibration, test visivi),
`physical_ai_mujoco/simulation/simulator.py`,
`physical_ai_mujoco/contracts/observation.py`,
`physical_ai_mujoco/observe/{core,pipeline,__init__}.py`,
`physical_ai_mujoco/evaluation/{observe_benchmark,sensor_calibration}.py`,
`configs/observe_tests/`, `tests/test_sensors_extended.py`,
`tests/test_architecture.py`, `README.md`, `docs/{SENSORS,OBSERVE_TEST}.md`,
`physical_ai_mujoco/sensors/README.md` (spiegazione dall'origine del modulo).

**Verifica:** Suite precedente e test nuovi passati nel working copy;
benchmark RGB-D pulito su 3 scene da 10 oggetti: recall 0,967, errore posizione
0,019 m, F1 supporti 0,391 (quindi il nuovo canale non corregge da solo il
relation estimator). Verificate anteprime PNG, curva di calibrazione e
confronto LiDAR-depth. La conta finale dei test viene riportata a consegna.

## 2026-09-18 12:38:59 — Confine privilegiato e invarianti OSSERVA verificabili

**Chi:** Codex (GPT-6), su richiesta di Matteo Paolini.

**Cosa:** La validazione dell'Observation e una funzione pura condivisa con il
builder. Il benchmark registra per scena le violazioni senza ricostruire
l'Observation, esclude gli output invalidi dalle metriche, continua con le
altre scene, esporta un grafo diagnostico e termina con codice 1 se esistono
violazioni. I test binari controllano il confine dei dati privilegiati e le
invarianti strutturali.

**Perché:** Prima `invariants_ok` era sempre `True` e il benchmark richiamava
il builder sull'output gia costruito: un grafo malformato interrompeva il
report alla prima scena, mentre il confine tra Observation e stato privilegiato
non aveva un controllo esplicito.

**File:** `physical_ai_mujoco/contracts/observation.py`,
`physical_ai_mujoco/contracts/__init__.py`,
`physical_ai_mujoco/observe/pipeline.py`,
`physical_ai_mujoco/evaluation/observe_benchmark.py`,
`tests/test_architecture.py`, `tests/test_observe_pipeline.py`,
`tests/test_simulated_stereo.py`, `docs/OBSERVE_TEST.md`.

**Verifica:** 71 test passati; prova end-to-end headless del benchmark con
Exact, Degraded e Stereo e report JSON/DOT/SVG generati.

## 2026-09-18 12:28:52 — Incertezza per oggetto visibile nel report OSSERVA

**Chi:** Codex (GPT-6), su richiesta di Matteo Paolini.

**Cosa:** Ogni osservazione del report salva per oggetto ID, visibilita,
stato ricordato e qualita di detection/posa; il riepilogo conta le scene con
`unknown_space`. La revisione visiva stampa queste informazioni. Il grafo di
relazioni non e piu dichiarato disponibile quando OSSERVA non ha osservato
alcun oggetto.

**Perché:** I contratti di OSSERVA gia contenevano le informazioni di
incertezza, ma il banco mostrava solo `unknown_space` e gli ID non visibili:
non si poteva controllare la qualita per oggetto. Inoltre, nel test con due
oggetti e disturbo stereo fisso, la stereo non ne ha ricostruito nessuno e il
report ha mostrato erroneamente `F1=0` perche `all(...)` su una scena vuota
restituiva `True`.

**File:** `physical_ai_mujoco/evaluation/observe_benchmark.py`,
`physical_ai_mujoco/observe/pipeline.py`, `tests/test_simulated_stereo.py`,
`docs/OBSERVE_TEST.md`.

**Verifica:** 63 test passati nella suite completa; test con drop totale
controlla che le relazioni non siano disponibili. Report e riapertura visiva
di una scena con due oggetti hanno mostrato i dettagli di incertezza.

## 2026-09-18 12:28:51 — Disturbo stereo e numero di oggetti selezionabili nel banco

**Chi:** Codex (GPT-6), su richiesta di Matteo Paolini.

**Cosa:** Il menu del test OSSERVA permette di scegliere il profilo delle
rilevazioni stereo indipendentemente da Degraded e un numero fisso o casuale
di oggetti per scena. La CLI offre `--stereo-profile` e
`--objects N|random` con intervallo minimo/massimo. Il test autonomo della
camera accetta rumore dei centroidi e probabilita di drop. Il report salva il
numero effettivo di oggetti e i parametri campionati per ogni scena; il
riesame ricostruisce ogni scena con il conteggio salvato.

**Perché:** Nel menu precedente il profilo clean/fixed/random controllava
insieme Degraded e Stereo, quindi non si poteva confrontare una sorgente
Degraded casuale con Stereo fissa. `object_count` restava nascosto nei JSON a
3 oggetti. I flag del test camera non esponevano il disturbo gia presente in
`StereoDegradation`. Le prove sim2real sui soli errori di rilevazione hanno
effetto sul percorso OSSERVA attuale; il degrado dei pixel rimane da valutare
quando esistera un detector che li usa.

**File:** `physical_ai_mujoco/evaluation/observe_benchmark.py`,
`physical_ai_mujoco/sensors/test_stereo_camera.py`,
`tests/test_simulated_stereo.py`, `README.md`, `docs/OBSERVE_TEST.md`.

**Verifica:** 63 test passati; il nuovo test verifica intervallo, condivisione
della scena tra sorgenti e riproducibilita. Menu provato con Degraded random,
Stereo fixed e 2-3 oggetti casuali; camera avviata con rumore 1 px e drop 0,1;
report con conteggio variabile riaperto nel viewer. Restano due warning
Gymnasium preesistenti sui limiti infiniti dell'observation space.

## 2026-09-18 12:03:23 — Apertura automatica del viewer dopo il report OSSERVA

**Chi:** Codex (GPT-6), su richiesta di Matteo Paolini.

**Cosa:** Un run interattivo del test OSSERVA apre immediatamente la prima
scena nel viewer MuJoCo/Gymnasium e le immagini stereo dopo avere stampato il
report. Dopo la chiusura con `q` o Esc propone le altre scene. `--headless`
continua a saltare la revisione.

**Perché:** Nel run del 20260918_120028 il programma era fermo al prompt
`Scena da rivedere` e non aveva aperto alcuna finestra: la scelta preventiva
della scena nascondeva la revisione visiva appena richiesta. I due report
20260918_114103 e 20260918_120028 hanno le prime due scene identiche; la
variazione delle medie deriva dalle due scene aggiunte, non da una regressione
dell'acquisizione.

**File:** `physical_ai_mujoco/evaluation/observe_benchmark.py`,
`docs/OBSERVE_TEST.md`.

**Verifica:** Run interattivo con una scena: viewer e immagini stereo aperti
automaticamente prima del prompt successivo; uscita con `0` riuscita.
Dieci test mirati OSSERVA/stereo passati.

## 2026-09-18 11:54:45 — Acquisizione stereo separata dalle etichette ideali

**Chi:** Codex (GPT-6), su richiesta di Matteo Paolini.

**Cosa:** `SimulatedStereoCamera.capture()` restituisce `StereoFrame` con sole
immagini e calibrazione. `OracleStereoLabels` aggiunge separatamente maschere,
ID e disturbo e produce il `SensorBundle` consumato da OSSERVA. Il benchmark
dichiara anche che l'ID del target arriva dal `TaskContext`, non da un
classificatore visivo. Nessun riconoscitore appreso e stato aggiunto.

**Perché:** La precedente camera leggeva contemporaneamente pixel, maschere
di segmentazione e tipi ideali MuJoCo; questo nascondeva nel sensore
l'informazione privilegiata e faceva sembrare che riconoscesse il target.
Nel report del 20260918_114103, il target era nelle maschere di entrambe le
viste ma non nell'output stereo: gli scarti verticali di 2,05 e 2,79 px
superavano la soglia attuale di 2 px. Separare i passaggi rende diagnosticabile
dove l'oggetto si perde.

**File:** `physical_ai_mujoco/sensors/simulated_stereo.py`,
`physical_ai_mujoco/sensors/oracle_stereo.py`,
`physical_ai_mujoco/sensors/__init__.py`,
`physical_ai_mujoco/sensors/test_stereo_camera.py`,
`physical_ai_mujoco/evaluation/observe_benchmark.py`,
`tests/test_simulated_stereo.py`, `docs/OBSERVE_TEST.md`.

**Verifica:** 62 test passati; il test verifica esplicitamente che il frame
grezzo non abbia maschere ne tipi. Avvio visivo della camera, benchmark e
riesame di un report esistente riusciti dopo la separazione. Restano i due
warning Gymnasium preesistenti sui limiti infiniti dell'observation space.

## 2026-09-18 11:48:33 — Riesame visivo dei report OSSERVA

**Chi:** Codex (GPT-6), su richiesta di Matteo Paolini.

**Cosa:** Dopo un benchmark interattivo si puo scegliere una scena del report
e mantenerne aperti il viewer MuJoCo/Gymnasium e le viste stereo fino a `q` o
Esc. `--review-report` riapre un JSON precedente, `--scene` seleziona la scena
e `--review-seconds` limita la durata; ogni riesame salva un PNG accanto ai
grafi. ID e `[T]` nell'anteprima sono marcati come etichette ideali.

**Perché:** Il viewer precedente si chiudeva dopo due secondi per scena,
prima che l'utente potesse leggere il report terminale. Il report random
20260918_114103 mostra recall stereo 0,5 e nessun target rilevato in due
scene: serviva vedere immagini e scena corrispondenti per capire se la perdita
nascesse da occlusione, maschere o triangolazione.

**File:** `physical_ai_mujoco/evaluation/observe_benchmark.py`, `README.md`,
`docs/OBSERVE_TEST.md`.

**Verifica:** Menu interattivo provato con una scena, scelta e uscita; report
preesistente riaperto con viewer e stereo; PNG diagnostico controllato;
62 test passati nella suite completa.

## 2026-09-18 11:38:32 — Avvio diretto dei due test dal percorso del file

**Chi:** Codex (GPT-6), su richiesta di Matteo Paolini.

**Cosa:** I due entry point in `observe/` e `sensors/` aggiungono la radice del
progetto al percorso degli import quando sono eseguiti come file Python. Il
test stereo usa l'import del package anche in questa modalita.

**Perché:** L'avvio `python /Users/matteopaolini/Desktop/mujoco_deploy/physical_ai_mujoco/observe/main_test_osserva.py`
falliva con `ModuleNotFoundError: physical_ai_mujoco.evaluation.observe_benchmark`:
Python inseriva `observe/` nel percorso degli import, mentre la prova precedente
aveva verificato soltanto `python -m`. Lo stesso problema avrebbe riguardato
l'import relativo del test stereo tramite percorso diretto.

**File:** `physical_ai_mujoco/observe/main_test_osserva.py`,
`physical_ai_mujoco/sensors/test_stereo_camera.py`, `docs/OBSERVE_TEST.md`.

**Verifica:** Entrambi i file avviati con percorso diretto e l'interprete
`mujoco-tirocinio`: OSSERVA ha prodotto report e due grafi su una scena;
stereo ha aperto il viewer e salvato l'anteprima dopo un secondo. La suite
completa di 62 test era passata prima di questa modifica agli entry point.

## 2026-09-18 11:33:31 — Test OSSERVA e stereocamera separati, viewer e baseline configurabile

**Chi:** Codex (GPT-6), su richiesta di Matteo Paolini.

**Cosa:** Il main del banco OSSERVA vive in `physical_ai_mujoco/observe/` e
quello visivo della stereocamera in `physical_ai_mujoco/sensors/`; si avviano
indipendentemente e aprono il viewer Gymnasium. Il sensore stereo consegna
immagini monocromatiche (tre canali uguali per il contratto esistente) e usa
`STEREO_BASELINE_M` per costruire fisicamente le due camere con la stessa
calibrazione. Il banco OSSERVA mantiene la scelta interattiva dei profili e
l'esportazione di report e grafi; il test camera mostra le due viste e salva
un'anteprima PNG.

**Perché:** L'avvio OSSERVA precedente offriva anche l'anteprima della camera,
quindi i due test risultavano accoppiati; inoltre il viewer non si apriva nei
normali run del benchmark. La distanza stereo era nel file di configurazione
dell'environment e il render del sensore restava RGB: non consentivano la
prova visiva autonoma e monocromatica richiesta. La voce dell'11:07:33 descrive
il comportamento precedente, ora superato da questi ingressi separati.

**File:** `physical_ai_mujoco/observe/main_test_osserva.py`,
`physical_ai_mujoco/sensors/test_stereo_camera.py`,
`physical_ai_mujoco/sensors/simulated_stereo.py`,
`physical_ai_mujoco/evaluation/observe_benchmark.py`,
`physical_ai_mujoco/envs/target_extraction.py`,
`physical_ai_mujoco/simulation/session.py`, `tests/test_simulated_stereo.py`,
`README.md`, `docs/OBSERVE_TEST.md`.

**Verifica:** 62 test passati nella suite completa; due warning Gymnasium
preesistenti sui limiti infiniti dell'observation space. Entrambi i nuovi
ingressi avviati separatamente con viewer per una scena; il test camera ha
salvato il PNG e il benchmark ha esportato JSON e sei file DOT/SVG. Un test
controlla che cambiare baseline sposti le camere MuJoCo e aggiorni la
calibrazione; un altro controlla l'uguaglianza dei tre canali immagine.

## 2026-09-18 11:07:33 — Sorgente stereo simulata e banco di prova autonomo OSSERVA

**Chi:** Codex (GPT-6), su richiesta di Matteo Paolini.

**Cosa:** La simulazione rende maschere di istanza visibili e calibrazione
stereo; `SimulatedStereoCamera` costruisce `SensorBundle` con disturbo di
centroide e omissione riproducibili. `main_test_osserva.py` confronta Exact,
Degraded e Stereo su scene identiche, scegliendo profili di disturbo fisso o
casuale; produce report JSON, grafi DOT/SVG e riepilogo a schermo. Una prova
visiva apre il viewer Gymnasium e mostra RGB e maschere affiancate.

**Perché:** Prima il ramo stereo di OSSERVA accettava soltanto maschere esterne
e un test sintetico di triangolazione: non esisteva una sorgente che partisse
dalle camere MuJoCo ne un percorso autonomo per misurare l'errore di OSSERVA.
Le maschere da segmentazione mantengono sagome e occlusioni reali della scena;
restano etichette ideali, quindi non confondono questa prova con un detector RGB.

**File:** simulation/simulator.py, sensors/, contracts/observation.py,
evaluation/observe_benchmark.py, main_test_osserva.py,
configs/observe_tests/, tests/test_simulated_stereo.py, README.md,
docs/OBSERVE_IMPLEMENTATION.md, docs/OBSERVE_TEST.md.

**Verifica:** 61 test passati nella suite completa, con i due warning Gymnasium
preesistenti sugli estremi infiniti dell'observation space; tre test mirati
passati dopo la rifinitura delle metriche. Banco provato su due scene con
disturbo casuale e 12 file DOT/SVG esportati. Prova visiva aperta con viewer
Gymnasium e finestre stereo per un secondo; PNG delle maschere controllato.

## 2026-09-17 18:17:38 — Contratto OSSERVA strutturato e sorgenti alternative

**Chi:** Codex (GPT-5), su richiesta di Matteo Paolini.

**Cosa:** `Observation` ora compone scena, relazioni candidate e incertezza,
allineate per ID, frame e timestamp. La pipeline OSSERVA attraversa estrazione,
comprensione della scena, relazioni, incertezza e builder. Sono disponibili una
sorgente esatta, una sorgente MuJoCo degradata con seed e un ingresso stereo
calibrato con maschere fornite da un detector. Il vettore a 17 valori per
oggetto resta nel ramo `PrivilegedState` del PPO teacher 1A.

**Perché:** Nel codice precedente `Observation` copiava 17 campi per oggetto
dal simulatore, inclusi massa, attrito e velocità; DECIDE poteva quindi usare
informazioni non ricavabili dalla stereo. `StereoCapture` produceva solo due
immagini e `decision_observation()` continuava a leggere lo stato esatto.
Questo impediva di sostituire la sorgente senza cambiare il contratto di
DECIDE e di misurare separatamente errore percettivo, relazionale e
incertezza. La baseline relazionale è dichiarata candidata geometrica; non
viene presentata come il modello appreso di Li né l'incertezza iniziale come
il belief CNABU di Marques.

**File:** physical_ai_mujoco/contracts/, observe/, decide/core.py,
envs/target_extraction.py, infrastructure/builder.py,
infrastructure/policy_adapter.py, evaluation/evaluator.py,
tests/test_architecture.py, tests/test_observe_pipeline.py, README.md,
docs/OBSERVE_IMPLEMENTATION.md.

**Verifica:** 58 test passati nella suite completa con accesso al display,
inclusi quelli stereo MuJoCo, i contratti, la triangolazione su maschere
sintetiche, il degrado riproducibile e il ramo PPO teacher. Due warning
preesistenti di Gymnasium sui limiti infiniti dell'observation space.

## 2026-09-17 16:19:10 — Prima migrazione modulare a oggetti

**Chi:** Codex (GPT-6), su richiesta di Matteo Paolini.

**Cosa:** Separati contracts, observe, decide, execute e task; costruzione scene
in SceneSession e visualizzazione in SimulationViewer. Environment Gymnasium
ridotto al coordinamento. Menu in ui, script mantenuti come ingressi compatibili,
training e valutazione nei rispettivi package. Voce 7 per selezionare profili
in configurazione, con fasi future esplicitamente non disponibili.

**Perché:** TargetExtractionEnv aveva 944 righe e mescolava simulazione, task,
percezione e viewer. Le policy leggevano il simulatore e campi privati. Matteo
richiede componenti a oggetti sostituibili durante la de-idealizzazione.

**File:** physical_ai_mujoco/, scripts/, main.py, configs/experiments/, docs/, tests/.

**Verifica:** 60 episodi comparati prima/dopo identici e tre scene 0A identiche;
52 test completi con grafica nell'ambiente mujoco-tirocinio; otto prove fisiche;
viewer con due episodi, GIF, modello PPO esistente e breve training completo.
Dettagli e versioni in docs/RESTRUCTURING.md e docs/validation/.

## 2026-09-17 16:19:10 — Ripristino completo di visibilita e contabilita del task

**Chi:** Codex (GPT-6), su richiesta di Matteo Paolini.

**Cosa:** Snapshot MuJoCo include RGBA. Lo snapshot pubblico dell'episodio
comprende riferimento del disturbo, totale e memoria dei crolli. La ricerca
esaustiva usa questa API e conserva separatamente il contatore TimeLimit.

**Perché:** Rimuovere un oggetto azzerava alpha, che il ripristino non recuperava.
La ricerca ripristinava pose e disturbo ma ometteva ever_collapsed, permettendo
che un ramo influenzasse il successivo. Sono difetti del ripristino, non nuove
regole fisiche o nuovi criteri di ricompensa.

**File:** simulation/simulator.py, task/, envs/target_extraction.py,
evaluation/exhaustive.py, tests/test_architecture.py.

**Verifica:** Test dedicati su visibilita dopo ripristino e pool, memoria dei
crolli dopo rollback e ricerca esaustiva operativa su due scene a tre oggetti.

## 2026-09-17 16:19:10 — Installazione e provenienza degli esperimenti

**Chi:** Codex (GPT-6), su richiesta di Matteo Paolini.

**Cosa:** Extra train/video in pyproject, requirements e start_project allineati;
metadati del training con input, versioni, commit/hash, seed e risultati. PPO
segnala dimensioni incompatibili. README aggiornato e precedenti guide archiviate.

**Perché:** Stable-Baselines3 mancava dalle dipendenze, il modello non conservava
la configurazione completa e la documentazione non corrispondeva al menu o alla
terminazione attuale. La dichiarazione di test su scene mai viste non era
supportata da una separazione esplicita dei seed.

**Verifica:** Breve training, valutazione e salvataggio artefatti nell'ambiente
mujoco-tirocinio; caricamento del modello preesistente; controllo dimensionale
PPO e apertura del menu. Lo split scientifico train/test resta da progettare.


## 2026-09-17 08:23:59 — Si puo' guardare la policy allenata mentre agisce

**Chi:** Claude (claude-opus-5), su richiesta di Matteo Paolini

**Cosa:**

- Nuovo `scripts/politiche.py`: `casuale`, `alto` e `carica(percorso)` per un
  modello allenato, tutte con la stessa firma
  `scegli(env, info, rng, osservazione)`. `carica` pretende il file delle
  statistiche di normalizzazione accanto al modello e **fallisce con un
  messaggio** se manca.
- `run_episode` accetta una policy; senza, resta la casuale mascherata di
  prima.
- Voce 1 del menu: dopo «come vuoi guardarli» chiede **chi decide le mosse** —
  a caso, l'euristica, o un modello allenato scelto da un elenco. La domanda
  compare solo dove ha effetto (finestra interattiva e senza finestra: le
  altre due modalita' girano in un altro processo).
- Il modello mostra su quanti oggetti e' stato allenato, perche' usarlo con un
  numero diverso non da' errore ma nemmeno senso.
- `allena.py` riusa le politiche condivise e **non cita piu' «l'ottimo vale
  +0,886»**: era misurato su 6 oggetti, e comparire in coda a un allenamento a
  12 e' un confronto falso. Ora rimanda a `quanto_margine.py` sulla
  configurazione in uso.

**Perché:**

Matteo, dopo il primo allenamento dal menu: *«non ho visto il risultato di
questa implementazione, non c'e' la simulazione»*. Aveva ragione e mancava
proprio: si poteva allenare un modello e poi **non c'era modo di vederlo
agire**. Il numero finale diceva se aveva imparato, ma non si poteva guardare
cosa facesse.

Il suo allenamento e' andato male per tre motivi che nessun messaggio gli
diceva: **1000 passi** (due soli aggiornamenti della rete), **12 oggetti**
invece di 6, e **magazzino spento**, che a 12 oggetti rende ogni episodio
lentissimo. Il ritorno −2,769 e' quello di una rete ancora casuale. La riga
finale che citava +0,886 come tetto peggiorava le cose, perche' quel numero
viene da una configurazione diversa.

**File:** `scripts/politiche.py` (nuovo), `scripts/run_phase_0b.py`,
`scripts/allena.py`, `main.py`

**Verifica:** 43 test passati. Provato dal menu: voce 1 → senza finestra → «un
modello allenato» → `ppo_6oggetti_25000passi` su 6 oggetti. Il modello prende
il target in una o due mosse (`object_002 (TARGET)`, `object_005 →
object_003 (TARGET)`), cioe' si comporta come i numeri dicevano. Con un modello
allenato su 3 oggetti compare la nota sul numero.

---


## 2026-09-17 08:01:19 — L'allenamento e il margine entrano nel menu

**Chi:** Claude (claude-opus-5), su richiesta di Matteo Paolini

**Cosa:**

- Nuova voce **3, «Allena la policy che sceglie l'ordine»**: chiede oggetti,
  passi, ambienti in parallelo e dimensione del magazzino, poi lancia
  `scripts/allena.py`. Le voci successive scalano di uno.
- La voce **Verifica** offre, dopo i test e le prove di fisica, anche
  `scripts/quanto_margine.py` col traguardo al target.

**Perché:**

Matteo: *«ora sul main ho una nuova opzione di scelta?»* — no, e non andava
bene. `allena.py` e `quanto_margine.py` erano due comandi da ricordare a
memoria, esattamente com'era successo a `verifica_fisica.py` prima di finire
nella voce Verifica. Uno strumento che non si raggiunge dal menu, in pratica,
non esiste.

Il margine sta dentro la verifica e non in una voce sua perche' e' un
controllo, non un esperimento: risponde a «il compito ha ancora qualcosa da
insegnare?», che e' una domanda da farsi insieme a «la fisica funziona?».

**File:** `main.py`

**Verifica:** 43 test passati. Provate da riga di comando la voce 5 (test +
prove di fisica + margine) e la voce 3 (allena 3 oggetti, 1000 passi,
magazzino 5): entrambe partono con i parametri giusti.

---


## 2026-09-16 23:01:27 — La valutazione dava l'osservazione grezza a un modello allenato su osservazioni normalizzate

**Chi:** Claude (claude-opus-5), su richiesta di Matteo Paolini

**Cosa:** in `scripts/allena.py` la policy appresa riceve
`vettore.normalize_obs(osservazione)` invece dell'osservazione grezza, e il
vettore si chiude dopo il confronto invece che prima.

**Perché:**

Primo allenamento vero, 25 000 passi. Durante l'allenamento `ep_rew_mean`
saliva regolarmente fino a **+0,713**; il confronto finale dava **−0,587 ±
1,953**. Due misure della stessa policy a un ordine di grandezza di distanza.

La causa: il modello e' allenato dentro `VecNormalize`, quindi ha visto solo
osservazioni normalizzate. Il confronto costruiva un environment pulito e
passava a `predict` i numeri grezzi. **Nessun errore viene sollevato** — la
forma del vettore e' identica, sono i valori a essere su un'altra scala — e la
policy sceglie sostanzialmente a caso senza che niente lo segnali.

Corretto e rimisurato sullo **stesso modello salvato**, 20 scene costruite da
zero:

| | ritorno |
|---|---|
| casuale | +0,728 ± 0,270 |
| alto (l'euristica) | +0,805 ± 0,070 |
| **PPO, osservazione grezza (sbagliato)** | **−0,587 ± 1,953** |
| **PPO, normalizzata (giusto)** | **+0,849 ± 0,050** |
| ottimo per ricerca esaustiva | +0,886 |

Con la correzione la policy **batte l'euristica** e copre il **54 % della
distanza** fra `alto` e l'ottimo, con 25 000 passi e trentacinque minuti su due
core. Senza la correzione la conclusione sarebbe stata «PPO non impara»: e'
esattamente il tipo di errore che rende inutile una campagna di misura, e si e'
visto solo perche' c'erano i riferimenti con cui confrontare.

**File:** `scripts/allena.py`

**Verifica:** la rimisura qui sopra, sullo stesso file di modello, cambiando
soltanto la normalizzazione dell'osservazione.

---


## 2026-09-16 22:24:54 — Magazzino di scene: l'allenamento smette di passare l'84% del tempo a far cadere oggetti

**Chi:** Claude (claude-opus-5), su richiesta di Matteo Paolini

**Cosa:**

- `Simulator.snapshot()` / `Simulator.restore()`: fotografia e ripristino di
  pose, velocita', tempo e rimozioni. `restore` **rifiuta** una fotografia che
  viene da un'altra scena.
- `TargetExtractionEnv`: due parametri nuovi, `scene_pool_size` (0 = spento,
  cioe' il comportamento di prima) e `fresh_scene_probability` (0,1). Col
  magazzino acceso, `reset` ripristina una scena gia' assestata invece di
  ricostruirla. `info["from_pool"]` dice quale delle due e' successa.
- Finche' il magazzino non e' pieno si costruisce **sempre**; a magazzino pieno
  una scena nuova ne **sostituisce** una a caso.
- `scripts/allena.py` usa il magazzino (200 scene per ambiente di default), ma
  **il confronto finale gira su scene costruite da zero**.
- Quattro test nuovi sul magazzino.

**Perché:**

Profilando l'allenamento: `reset` 0,731 s, `step` 0,045 s, 3,4 passi per
episodio — **l'84% del tempo se ne andava a costruire scene**. Prova reale:
5 passi al secondo, cioe' ventiquattro ore per 200 000 passi.

Il costo sta nel far **cadere** gli oggetti uno alla volta, non nel compilare
il modello: ricompilare costa millisecondi. Quindi la scena assestata si puo'
fotografare e riusare. A regime: **reset da 688 ms a 20,5 ms**, e
l'allenamento da 5 a **12 passi al secondo** su due core.

**Due cose sono state sbagliate e corrette prima di arrivarci**, e vale la pena
scriverle perche' erano entrambe silenziose:

1. *Il magazzino si riempiva al ritmo delle scene fresche.* Con il 10% di scene
   nuove, dopo 40 episodi il magazzino conteneva **4 scene** e la policy
   rigiocava sempre quelle. Misurato: «scene distinte 4 su 40 episodi».
   Risolto costruendo sempre finche' il magazzino non e' pieno.
2. *A magazzino pieno la varieta' si sarebbe congelata.* Una scena nuova adesso
   ne sostituisce una vecchia, cosi' l'insieme continua a rinnovarsi per tutto
   l'allenamento — che e' il motivo per cui Matteo ha scelto «meta' magazzino,
   meta' fresche» invece di un magazzino fisso.

**Il rischio residuo e' che la policy impari le scene invece del compito.** Per
questo la valutazione finale usa scene costruite da zero: se imparasse a
memoria, il confronto lo mostrerebbe.

**File:** `physical_ai_mujoco/simulation/simulator.py`,
`physical_ai_mujoco/envs/target_extraction.py`, `scripts/allena.py`,
`tests/test_phase_0b.py`

**Verifica:** **43 test passati** (39 + 4 nuovi). I test nuovi coprono cio' che
puo' rompersi in silenzio: che una scena ripresa dal magazzino sia **ferma**
come una appena costruita; che le rimozioni di un episodio **non sopravvivano**
a quello dopo (senza ripristinare `contype`/`conaffinity` la scena si
svuoterebbe da sola); che `restore` **rifiuti** la fotografia di un'altra
scena, che MuJoCo accetterebbe in silenzio producendo una scena plausibile ma
sbagliata.

---


## 2026-09-16 21:57:35 — L'episodio finisce al target, e c'e' l'allenamento PPO

**Chi:** Claude (claude-opus-5), su richiesta di Matteo Paolini

**Cosa:**

- `configs/phase_0b/env.json`: **`terminate_on_target` passa a `true`**.
  L'episodio finisce quando il target esce, invece di svuotare la scena.
- Nuovo `scripts/allena.py`: allenamento PPO (stable-baselines3) della policy
  di rimozione, sull'osservazione di stato, **senza maschera delle azioni**,
  con ambienti in parallelo e normalizzazione delle sole osservazioni. Alla
  fine confronta il modello con `casuale` e `alto` su scene mai viste.
- Nuova dipendenza: `stable-baselines3`.

**Perché:**

**Il traguardo al target.** Misurato con `quanto_margine.py` (voce precedente):
svuotando la scena, l'ottimo assoluto batte «togli il piu' in alto» di 0,015,
cioe' meno del rumore. Col traguardo al target il margine sale a 0,078, quattro
volte il rumore, e l'ottimo arriva al target in 1,70 mosse contro le 3,30 di
`alto`. E' la differenza fra un compito gia' risolto e un compito che ha una
risposta da imparare.

**Senza maschera**, per scelta di Matteo, e la ragione tiene: la maschera dice
quali oggetti sono ancora in scena, cioe' informazione privilegiata che nel
mondo vero va dedotta da cio' che si vede. Una policy allenata con la maschera
non impara a dedurla e non si trasferisce allo student, che avra' solo le
telecamere. L'environment penalizza le azioni non valide invece di rifiutarle,
quindi la cosa e' apprendibile.

**Le ricompense NON vengono normalizzate**, le osservazioni si. Normalizzare le
ricompense renderebbe i ritorni non confrontabili con quelli delle politiche di
riferimento, che sono l'unica misura che abbiamo di «ha imparato qualcosa».

**Un numero che condiziona tutto il resto.** Profilato dove va il tempo, su 8
episodi da 6 oggetti:

| | mediana | quota del tempo |
|---|---|---|
| `reset` (costruzione della scena) | **0,731 s** | **84 %** |
| `step` (una rimozione) | 0,045 s | 16 % |

Con 3,4 passi per episodio, **l'84 % del tempo di allenamento se ne va a
costruire scene**, non a simulare le mosse che la policy deve imparare. Prova
reale: 3072 passi in 8,6 minuti, cioe' 5 passi al secondo — a questo ritmo
200 000 passi sono ventiquattro ore. La costruzione sequenziale (ogni oggetto
cade e si assesta prima del successivo) e' cio' che rende le scene realistiche,
ed e' anche cio' che rende l'allenamento lento. Va risolto prima di allenare
sul serio.

**File:** `configs/phase_0b/env.json`, `scripts/allena.py` (nuovo)

**Verifica:** 39 test passati con la nuova configurazione. `allena.py` eseguito
per intero su 3000 passi: allena, salva il modello e la normalizzazione, e
confronta. Il modello a 3000 passi vale +0,154 contro +0,808 di `alto` — cioe'
non ha imparato niente, che a 3000 passi e' quello che ci si aspetta. Serviva a
verificare la conduttura, non il risultato.

---


## 2026-09-16 21:39:48 — Misurato quanto c'e' da guadagnare imparando l'ordine

**Chi:** Claude (claude-opus-5), su richiesta di Matteo Paolini

**Cosa:** nuovo `scripts/quanto_margine.py`. Confronta, sulle stesse scene,
`casuale`, `alto` (togli il piu' in alto) e **`ottimo`**: il miglior ordine
possibile, trovato provandoli tutti. La ricerca esaustiva e' praticabile perche'
lo stato del simulatore viene fotografato e ripristinato — i prefissi comuni si
simulano una volta sola e nessuna scena viene ricostruita. Con `--finisci-al-target`
confronta le due letture del compito.

**Perché:**

Matteo ha proposto di iniziare ad allenare una policy. Prima di spenderci sopra
un allenamento serviva sapere **quanto c'e' da guadagnare**, e la risposta
cambia il piano:

| configurazione | casuale | alto | ottimo | margine su 'alto' |
|---|---|---|---|---|
| **svuota la scena (attuale)** | +0,434 | +0,723 | **+0,737** | +0,015 — dentro il rumore |
| **traguardo al target** | +0,702 | +0,808 | **+0,886** | +0,078 — quattro volte il rumore |

**Con la configurazione di oggi il compito e' gia' risolto.** Se l'episodio
svuota sempre la scena, ogni politica paga tutte le rimozioni e incassa il
premio: l'unica leva e' il disturbo, e «togli quello piu' in alto» ne produce
6,8 mm contro una soglia di 50. Il meglio raggiungibile in assoluto batte
quella riga di euristica di 0,015, che con otto scene non e' nemmeno
misurabile.

Col traguardo al target il margine ricompare, e si vede anche **perche'**:
l'ottimo arriva al target in **1,70 mosse**, `alto` in **3,30**. L'ottimo scava
la meta': toglie solo cio' che grava sul target, mentre `alto` toglie il piu'
alto anche quando sta dall'altra parte del mucchio. Sulle sole 9 scene su 10
col target sepolto il margine e' +0,065.

**Un difetto trovato strada facendo, che vale la pena annotare.** La prima
versione della ricerca restituiva ritorni **sopra il tetto teorico**: +0,830
contro un massimo di +0,750. Causa: il wrapper `TimeLimit` conta i passi
globalmente, e una ricerca esaustiva ne fa migliaia senza mai un reset.
Superati i 64, ogni ramo veniva troncato a meta', e un ramo tagliato dopo due
mosse sembrava migliore di uno completo perche' aveva pagato due rimozioni
invece di cinque. Risolto fotografando anche quel contatore. Se avessimo
allenato una policy senza questo controllo, avremmo ottimizzato la stessa
misura sbagliata senza accorgercene.

**File:** `scripts/quanto_margine.py` (nuovo)

**Verifica:** 39 test passati. I ritorni dell'ottimo ora stanno sotto il tetto
teorico in entrambe le configurazioni, che e' la condizione che il difetto
violava.

---


## 2026-09-16 21:27:21 — Il riepilogo della finestra con pause diceva il falso

**Chi:** Claude (claude-opus-5), su richiesta di Matteo Paolini

**Cosa:**

- `EnvRunner.last_return`: il ritorno dell'**ultimo episodio concluso**, fissato
  quando l'episodio finisce. Il riepilogo stampa questo invece di
  `total_reward`.
- `EnvRunner.done`: un riquadro che ha finito gli episodi richiesti viene
  contato una volta sola.
- Il riepilogo dice «1 episodio completato» invece di «1 episodi completati», e
  «nessun episodio concluso» quando non ce n'e' stato nessuno.

**Perché:**

Matteo ha eseguito la finestra con pause chiedendo 3 episodi su 2 riquadri, e
il riepilogo ha risposto: *«scena 0: 1 episodi completati, ultimo ritorno
+0.000»*. Entrambi i numeri erano sbagliati, per due motivi diversi.

**Lo zero.** `total_reward` viene azzerato da `start_episode`. Chiudendo la
finestra mentre l'episodio successivo era appena partito, il riepilogo leggeva
un contatore gia' resettato — riportava zero al posto del risultato vero.

**Il conteggio.** Un riquadro che aveva finito i suoi episodi non veniva
marcato: a ogni fotogramma rientrava nel ramo `if runner.outcome`, si
incrementava `episode` di nuovo e `finished` di nuovo. Con due riquadri, il
primo che finiva bastava da solo a portare `finished` a 2 e a chiudere la
sessione — **fermando l'altro riquadro a meta' lavoro**. E `episode` finiva
sovrastimato di uno.

**File:** `scripts/watch_phase_0b.py`

**Verifica:** 39 test passati. Eseguito `--parallel 2 --episodes 3`: il
riepilogo ora dice «3 episodi completati» per entrambi i riquadri, con ritorni
+0,675 e +0,763. Prima si fermava prima.

---

## 2026-09-16 21:27:21 — Rimossi i file doppioni dal progetto sul Mac

**Chi:** Claude (claude-opus-5), su richiesta di Matteo Paolini

**Cosa:** spostati in `_to_delete/doppioni/` sei file con il suffisso `-1`
(`test_phase_0b-1.py`, `scene_description-1.py`, `scene_generator-1.py`,
`mujoco_builder-1.py`, `target_extraction-1.py`, `README_oa-1.md`) e tutte le
cartelle `__pycache__`. Nessuna modifica al codice vero.

**Perché:**

La suite riportava **1 test fallito su 72**, con l'errore
`'SimulationDescription' object has no attribute 'linear_velocity_threshold'`.
Il progetto ha 39 test, non 72: gli altri 33 venivano da
`tests/test_phase_0b-1.py`, una copia del 16 settembre alle 15:49 finita nel
repository durante un trasferimento. Pytest la raccoglieva insieme a quella
vera, e testava il codice di stamattina contro l'interfaccia di adesso.

Erano copie morte da ore che nessuno guardava. Il rosso non veniva dal
progetto, veniva da loro.

**File:** nessuno nel progetto. Solo file spostati fuori.

**Verifica:** 39 test passati, nessun fallimento.

---


## 2026-09-16 21:18:19 — Il menu passa da otto voci a cinque

**Chi:** Claude (claude-opus-5), su richiesta di Matteo Paolini

**Cosa:**

- `main.py` espone cinque voci invece di otto: **esegui episodi**, **ispeziona
  una scena**, **raccogli dati stereo**, **verifica**, **avanzate**.
- Le vecchie voci 1-4 sono diventate una sola. Il «come guardarli» —
  finestra interattiva, finestra con pause, filmato, niente — e' la prima
  domanda dopo la scelta, non quattro voci di menu.
- Nuova voce **Verifica**: esegue `pytest` e poi, chiedendo, anche
  `scripts/verifica_fisica.py` (con `--scene N` opzionale). Prima lo script
  delle prove di fisica esisteva ma dal menu non si raggiungeva.
- Fase 0A finisce sotto **Avanzate**: e' congelata e non riguarda l'uso
  quotidiano.
- `ask()` e `ask_yes_no()` gestiscono `Ctrl-D`: escono in silenzio invece di
  stampare una traccia di errore.

**Perché:**

Matteo, davanti al menu: *«ho troppe cose tra cui scegliere, e non capisco
cosa sto scegliendo nella realtà»*.

Aveva ragione, e il motivo e' strutturale: **le voci 1, 2, 3 e 4 facevano la
stessa identica cosa** — eseguire episodi con una politica — e differivano
solo nel rendering. Quattro voci di pari livello suggerivano quattro
esperimenti diversi quando l'esperimento era uno solo. La 5 era la 4 piu' il
salvataggio delle immagini. Il menu descriveva l'implementazione invece dello
scopo.

**File:** `main.py`

**Verifica:** 39 test passati. Provati da riga di comando i percorsi voce 1 →
modalita' 4 (due episodi, `Successi: 2/2`) e voce 4 (pytest + prove di fisica).
Il percorso con finestra va provato sul Mac: qui non c'e' un display.

---


## 2026-09-16 21:02:59 — Il viewer non si congela piu' mentre si sceglie la mossa

**Chi:** Claude (claude-opus-5), su richiesta di Matteo Paolini

**Cosa:**

- Nella politica `manual` di `inspect_episode`, l'attesa della tastiera non
  blocca piu' il processo: `_chiedi()` aspetta a intervalli di 30 ms e in
  ognuno richiama `env.render()`. Senza viewer, o quando l'ingresso non e' un
  terminale, legge e basta.
- `Ctrl-C` e `Ctrl-D` durante la scelta chiudono l'episodio con una riga
  («interrotto») invece di una traccia di errore.
- Il prompt stampa `[0, 1, 2]` invece di
  `[np.int64(0), np.int64(1), np.int64(2)]`.

**Perché:**

Matteo ha aperto il viewer con la politica `manual` e macOS ha dichiarato
**«python (non risponde)»**: finestra congelata, uscita forzata, episodio
perso. Due volte di fila.

La causa è `input()`. Blocca il processo finché non si preme Invio, e con il
processo fermo nessuno pompa il ciclo di eventi della finestra GLFW: dopo
qualche secondo il sistema operativo la considera bloccata. Non era un
problema di fisica né di prestazioni — era che nel momento in cui il
programma aspetta l'utente, smette anche di disegnare.

**File:** `scripts/inspect_episode.py`

**Verifica:** 39 test passati. Con ingresso da pipe il comportamento è
invariato (legge e basta); con `Ctrl-D` a metà episodio stampa «interrotto» e
chiude pulito. Il caso interattivo con finestra va provato sul Mac: qui non
c'è un display.

---

## 2026-09-16 21:02:59 — Attrito volvente piu' alto: ipotesi provata e scartata

**Chi:** Claude (claude-opus-5), su richiesta di Matteo Paolini

**Cosa:** nessuna modifica al codice. Si annota il risultato di una misura.

**Perché:**

Nelle scene restano assestamenti lunghi — sul Mac di Matteo 1 su 120 arriva al
timeout di 20 s. Tracciandoli: in entrambi i casi oltre i 4 secondi **c'è un
oggetto che percorre 23–25 cm**. Non è la strisciata della voce precedente: è
un oggetto che viene urtato, scivola giù dalla pila e viaggia sul pavimento
infinito, dove niente lo ferma. È fisica giusta, e con un pavimento senza
bordi è un esito legittimo.

L'ipotesi era che alzare l'attrito volvente li fermasse prima. **Misurata su
90 assestamenti per valore, è falsa:**

| attrito volvente | falliti | assest. mediano | viaggio massimo | ribaltamento (atteso 14,04°) |
|---|---|---|---|---|
| **0,005 (attuale)** | 0/90 | 1,12 s | 62 cm | **13,13°** |
| 0,010 | 0/90 | 1,02 s | 121 cm | 15,41° |
| 0,020 | 0/90 | 0,93 s | 114 cm | 19,07° |
| 0,050 | 0/90 | 0,91 s | 83 cm | 27,58° |

Il viaggio massimo **non diminuisce**, e l'angolo di ribaltamento peggiora
fino a sbagliare di 13 gradi e mezzo. Il valore attuale resta il migliore.

**Nota sulla riproducibilità, che va tenuta a mente.** Gli stessi semi con lo
stesso codice danno risultati **diversi su macchine diverse**: sul Mac
(MuJoCo 3.13) 1 assestamento fallito su 120 e deriva peggiore 5,0 mm; qui, con
MuJoCo 3.13, 0 falliti e deriva peggiore 2,1 mm; con MuJoCo 3.12, 0 falliti e
0,9 mm. Le scene sono identiche (compenetrazione 0,650 mm in tutte e due le
versioni): a divergere sono solo i casi al limite, dove un oggetto in
equilibrio marginale si decide in un senso o nell'altro. MuJoCo è
deterministico a parità di binario, non fra architetture diverse. Va scritto
in tesi, e riguarda la riga «stesso seed, stessa traiettoria» del banco di
prova.

**File:** nessuno.

**Verifica:** la misura è la voce stessa. Ripetibile con
`python scripts/verifica_fisica.py --scene 20`.

---


## 2026-09-16 20:22:44 — La quiete si misura sullo spostamento, e il contatto smette di strisciare

**Chi:** Claude (claude-opus-5), su richiesta di Matteo Paolini

**Cosa:**

- **Criterio di quiete rifatto.** `step_until_settled` non guarda più la
  velocità istantanea ma **quanto gli oggetti si sono spostati** nella finestra
  `stable_duration`. `SimulationDescription` sostituisce
  `linear_velocity_threshold` / `angular_velocity_threshold` con
  `linear_settle_tolerance` / `angular_settle_tolerance` (1 mm e 1° in Fase 0B,
  2 mm e 2° in Fase 0A). Nelle configurazioni le chiavi `settling` diventano
  `linear_tolerance` e `angular_tolerance`.
- **Parametri di contatto della Fase 0B:** cono `elliptic`, `impratio` **10**,
  `noslip_iterations` **10**, costante di tempo del contatto **0,01 s**
  (`solref`, dichiarata una volta in `<default>`).
- **Nuovo script `scripts/verifica_fisica.py`**: otto prove con risposta
  analitica nota (caduta libera da tre quote, angolo di scivolamento, angolo di
  ribaltamento, inerzia dei tre primitivi) più, con `--scene N`, le prove sulle
  pile vere (tempi di assestamento, oggetti sospesi, compenetrazione, deriva
  residua). Legge i parametri dalla configurazione, così misura ciò che il
  progetto usa davvero.

**Perché:**

Matteo: *«i pezzi continuano a non rispettare la fisica quando vanno a trovarsi
in situazioni di non equilibrio»*. Con i parametri della voce precedente
9 assestamenti su 36 scadevano in timeout.

**Primo risultato, ed è il più importante: quegli assestamenti non erano
falliti.** Tracciando `object_003` del seme 2 per venti secondi, dal secondo 3,9
in poi si sposta di **2 mm in x e 1 mm in z — in diciannove secondi**. È fermo.
A far ripartire il conteggio erano **picchi di velocità angolare** da 0,1–0,4
rad/s che durano pochi passi e non lo muovono. Misurato su 120 assestamenti: su
quelli dichiarati falliti gli oggetti si spostavano **0,14 mm al secondo**
(caso peggiore 0,29 mm). Una soglia sulla velocità istantanea non può
distinguere quei picchi da una rotazione vera — 0,4 rad/s sono 23 °/s, e
alzarla fin lì accetterebbe un oggetto che sta davvero ruotando. Solo mediando
nel tempo si separano, e mediare nel tempo vuol dire guardare lo spostamento.

**Secondo risultato: il contatto strisciava davvero, ed è l'altra metà della
gelatina.** Misurando la deriva residua **dopo** che l'assestamento è stato
dichiarato, su 15 scene:

| configurazione | assest. falliti | mediana | 95° | compenetraz. | deriva mediana | deriva peggiore |
|---|---|---|---|---|---|---|
| piramidale, impratio 1 (prima) | 4/90 | 1,47 s | 19,70 s | 0,11 mm | **1,465 mm/s** | **38,5 mm/s** |
| ellittico, impratio 10 | 0/90 | 1,15 s | 2,68 s | 0,56 mm | 0,436 mm/s | 1,67 mm/s |
| + noslip 10 | 0/90 | 1,11 s | 2,01 s | 0,62 mm | 0,039 mm/s | 0,265 mm/s |
| + contatto a 0,01 s | 0/120 | 1,01 s | 2,31 s | **0,21 mm** | **0,018 mm/s** | 0,87 mm/s |

38 mm/s di deriva su un assestamento da venti secondi sono i **24 cm** che
`object_010` aveva percorso "da fermo": è la strisciata che si vedeva.

`noslip_iterations` è il rimedio documentato: una passata PGS aggiuntiva sulle
sole direzioni d'attrito, a vincoli rigidi, dopo il solutore principale —
*«this suppresses the contact slip that is inherent to soft-constraint
models»*. La stessa documentazione avverte che può destabilizzare i contatti
multipli, che è esattamente il nostro caso: per questo è stato **misurato**, non
adottato sulla parola. Non destabilizza nulla, e costa il 10% di tempo di
calcolo.

Irrigidire il contatto da 0,02 s a 0,01 s (cinque passi di integrazione, contro
i due che la documentazione indica come minimo) recupera la compenetrazione
persa con il cono ellittico: **da 0,62 a 0,21 mm**, e migliora anche tutto il
resto.

**Ipotesi smentite lungo la strada**, annotate perché non vengano ritentate:
alzare le iterazioni del solutore a 500 con tolleranza 10⁻¹² non cambia
**nulla** (9 falliti su 36, identico) — il solutore converge già, il problema
non era la convergenza; dimezzare il passo di integrazione a 0,001 s dimezza
solo la velocità di calcolo.

**File:** `physical_ai_mujoco/scene/scene_description.py`,
`physical_ai_mujoco/scene/scene_generator.py`,
`physical_ai_mujoco/scene/dataset_loader.py`,
`physical_ai_mujoco/simulation/simulator.py`,
`physical_ai_mujoco/simulation/mujoco_builder.py`,
`configs/phase_0a/simulation.json`, `configs/phase_0b/simulation.json`,
`scripts/verifica_fisica.py`, `tests/test_phase_0b.py`

**Verifica:**

- **39 test passati** (la suite scende da 71 s a 43 s, perché non ci sono più
  assestamenti che aspettano il timeout).
- `scripts/verifica_fisica.py`: **otto prove su otto entro tolleranza**. Caduta
  libera entro 0,7 ms su tre quote; scivolamento **30,23°** contro 30,96°
  teorici; ribaltamento **13,13°** contro 14,04°; inerzia dei tre primitivi
  identica a quella del compilatore.
- `--scene 20`, 120 assestamenti: **0 non assestati** (erano 31 su 120),
  **0 oggetti sospesi**, compenetrazione peggiore 0,63 mm, deriva mediana
  0,018 mm/s.
- Le tre politiche di riferimento restano separate su 20 episodi: `top`
  **+0,662 ± 0,056**, `random` **+0,289 ± 0,679**, `target` **+0,029 ± 0,800**
  (prima +0,684 / +0,313 / +0,081). Il target resta sepolto in 16 scene su 20,
  contro 18 su 20: differenza dentro il rumore a questo numero di campioni, da
  ricontrollare quando si farà la campagna di misura vera.

---


## 2026-09-16 17:46:43 — Fotogrammi saltati invece di riproduzione lenta, e centro di massa randomizzato

**Chi:** Claude (claude-opus-5), su richiesta di Matteo Paolini

**Cosa:**

- `_live_render` confronta il tempo simulato con il tempo reale trascorso: se è
  in ritardo **salta il fotogramma**; se è in anticipo dorme la differenza.
- `ObjectDescription.center_of_mass`: scostamento del centro di massa dal centro
  geometrico. `principal_inertia()` e `half_extents()` in `scene_description`.
- Nuova regola di scena `randomisation.center_of_mass_jitter`, frazione della
  semi-dimensione per asse. Fase 0B: **0,25**. Default 0, cioè disattivata.
- Quando il centro di massa è spostato, l'MJCF dichiara `<inertial>` con posa,
  massa e inerzia esplicite.
- Il catalogo mostra lo scostamento in percentuale della semi-dimensione.

**Perché:**

**Il salto dei fotogrammi.** La voce precedente affermava che dormire il tempo
mancante risolvesse la lentezza. **Era sbagliato, e va detto.** Dai numeri di
Matteo: 471 disegni in 30,5 s, cioè **65 ms per fotogramma** contro un budget
di 32 ms. Disegnare costa il doppio del tempo simulato che quel fotogramma
rappresenta: tolta l'attesa di troppo restano comunque 65 ms, e la riproduzione
resta a 0,49×. L'unico modo di stare al passo è disegnarne di meno. Verificato
con un disegno artificiale da 60 ms: **1,00× a velocità piena e 0,51× col
rallentatore**, contro 0,49× in entrambi i casi prima.

**Il centro di massa.** Matteo ha chiesto se avessimo il momento d'inerzia e ha
proposto di randomizzare il centro di massa in vista del sim2real. Sul primo
punto: l'inerzia c'era già, calcolata dal compilatore MuJoCo dalla geometria e
dalla massa — verificato contro la formula analitica, scarto **6·10⁻¹¹**. Ma il
centro di massa era `[0, 0, 0]` per ogni oggetto, cioè ogni solido perfettamente
uniforme.

È un'idealizzazione che pesa proprio su questo compito: a decidere se un oggetto
si ribalta quando gli si toglie il vicino non è la massa, è **dove** sta. Ed è un
parametro che nel reale **non si misura**, quindi randomizzarlo è domain
randomization difendibile — al contrario di attriti e masse, che si potrebbero
identificare dai dati (cfr. DROPO in `upgrade/scenari futuri.md`).

Lo scostamento è espresso come frazione della semi-dimensione, non in
millimetri, così la stessa regola vale per una lastra da 1,9 cm e per una
scatola da 9 cm senza spostare la massa fuori dall'oggetto.

Effetto misurato su 15 scene da 6 oggetti, policy casuale:

| jitter | mossa peggiore (media) | dev. std |
|---|---|---|
| 0 (centrato) | 0,0348 m | 0,0352 m |
| ±25% | 0,0443 m | 0,0318 m |
| ±40% | 0,0458 m | 0,0445 m |

Le scene diventano **più difficili** (+27% di disturbo medio a ±25%), che è
esattamente lo scopo: una policy addestrata solo su solidi uniformi imparerebbe
una regola che nel reale non vale.

**File:** `physical_ai_mujoco/envs/target_extraction.py`,
`physical_ai_mujoco/scene/{scene_description,scene_generator}.py`,
`physical_ai_mujoco/simulation/mujoco_builder.py`,
`configs/phase_0b/scene_rules.json`, `scripts/inspect_episode.py`,
`tests/test_phase_0b.py`

**Verifica:** 39 test passati, cinque nuovi. Fra questi: le nostre formule
d'inerzia confrontate con quelle che il compilatore MuJoCo deriva da solo, per
tutte e tre le forme — serve perché dichiarando `<inertial>` la fonte
dell'inerzia diventiamo noi, e una formula sbagliata cambierebbe la fisica senza
che niente protesti. E un test che con jitter a 0 il centro di massa resta
esattamente al centro, cioè che la randomizzazione è un'aggiunta e non una
riscrittura silenziosa.

---

## 2026-09-16 17:31:29 — Riproduzione a velocità reale e soglie di quiete più strette

**Chi:** Claude (claude-opus-5), su richiesta di Matteo Paolini

**Cosa:**

- `_live_render` dorme il tempo che **manca** al budget del passo, non
  l'intervallo intero sommato al costo del disegno.
- Soglie di quiete di Fase 0B: lineare da 0,01 a **0,005 m/s**, angolare da
  0,05 a **0,02 rad/s**. `stable_duration` resta 0,5 s.
- Il rapporto col tempo reale si stampa con due cifre.

**Perché:**

Matteo ha segnalato che la fisica sembrava «sulla luna, o in una gelatina», e
che alcuni oggetti restavano in equilibrio sullo spigolo. Tre cause distinte,
separate misurando:

1. **La lentezza era di riproduzione, non di fisica.** Il viewer dormiva il
   budget completo del passo (32 ms) *dopo* aver già speso decine di
   millisecondi a disegnare. Sul suo Mac: 15,07 s simulati in 30,5 s reali,
   cioè **0,49× il tempo reale**. La fisica era corretta; era il filmato a
   scorrere al rallentatore.
2. **Dichiaravamo la quiete troppo presto.** Tracciando le velocità durante un
   assestamento, `object_005` risultava fermo per 1,5 s e poi a t=1,80
   schizzava a **0,43 rad/s**: era in equilibrio metastabile su uno spigolo e
   creeping a 0,012 rad/s — sotto la soglia angolare di 0,05, quindi
   classificato «fermo» mentre stava andando a ribaltarsi. Quel moto veniva poi
   addebitato alla **mossa successiva**. Misurata la deriva nei 2 s dopo la
   quiete dichiarata: **8,4 mm con le soglie vecchie, 3,0 mm con le nuove**, su
   una soglia di disturbo di 50 mm. Con le soglie strette i ribaltamenti
   ritardati passano da presenti a **0 su 6 scene**.
3. **L'attrito volvente non c'entrava.** Sweep da 0,0001 (default MuJoCo) a
   0,05: il tempo di assestamento mediano resta 1,3–1,6 s in tutti i casi.
   L'ipotesi iniziale era sbagliata e va registrata come tale.

Testato anche l'ammorbidimento del contatto (`solref` 0,05 e 0,10 invece del
default 0,02): **peggiora**, i ribaltamenti ritardati risalgono a 2/6 e 1/6.
Lasciato il contatto rigido.

Sulla domanda «abbiamo momento d'inerzia, centro di massa, attrito?»: sì. Il
compilatore MuJoCo calcola tensore d'inerzia e centro di massa dalla geometria
e dalla massa; `balanceinertia` è lasciato spento di proposito in
`mujoco_builder`, così un'inerzia non fisica emerge come errore di compilazione
invece di essere corretta in silenzio. L'attrito è per oggetto nel dataset.
Non mancava nessuna grandezza: erano sbagliati i valori delle soglie e la
cadenza del disegno.

**Costo:** l'assestamento mediano passa da 1,27 s a ~3 s simulati, e la suite
di test da 55 s a 112 s. È il prezzo di aspettare la quiete vera.

**File:** `physical_ai_mujoco/envs/target_extraction.py`,
`configs/phase_0b/simulation.json`, `scripts/inspect_episode.py`

**Verifica:** 34 test passati. Su 20 episodi con 6 oggetti (seed 0):
`top` 20/20 successi e mossa peggiore media 0,0030 m · `random` 16/20 e
0,0349 m · `target` 13/20 e 0,0498 m. Target sepolto in 18 scene su 20.
Costruzione della scena: 22,61 s simulati in 1168 ms reali, **19,4× il tempo
reale** senza viewer.

---

## 2026-09-16 15:10:49 — Registro di costruzione della scena, oggetti impilabili, contatti nel simulatore

**Chi:** Claude (claude-opus-5), su richiesta di Matteo Paolini

**Cosa:**

- `info["release_log"]`: una riga per ogni oggetto rilasciato, con massa, quota
  di rilascio, caduta effettiva del baricentro, secondi simulati e millisecondi
  reali di assestamento, se la quiete è stata raggiunta, su cosa è finito
  appoggiato, e **quanto quella caduta ha spostato ciò che c'era già**.
  `inspect_episode` e `run_phase_0b` lo stampano come tabella.
- Tre nuovi tipi nel dataset di Fase 0B — `slab`, `plank`, `disc`: faccia piana
  larga, baricentro basso. La scena di Fase 0B ora è a maggioranza di lastre.
- `Simulator.contact_pairs()`, `support_graph()`, `supported_by()`: la lettura
  di `data.contact` si è spostata dentro `simulation/`.
- Corretto il premio del target: si paga sulla **mossa** che lo estrae, non
  finché il target risulta estratto.
- `is_success` e `collapsed_ever` riguardano l'**episodio**, non l'ultimo passo.
- Il test sulla disparità stereo ha una scena dedicata (`tests/data/`).

**Perché:**

- Matteo ha chiesto di cronometrare l'assestamento di ogni oggetto e di
  stampare più dati possibili per capire se la fisica funziona. Senza quei
  numeri non c'è modo di distinguere «la pila regge» da «ogni aggiunta
  rimescola tutto».
- Con sfere e cilindri alti il target restava scoperto nel 37% delle scene:
  quelle forme rotolano e si ribaltano, e chiedere loro di impilarsi è chiedere
  alla fisica una cosa che non fanno. Con le lastre: **target sepolto dal 63% al
  80%**, e fino a 3 oggetti che gravano sul target invece di 1. Sono anche più
  rappresentative — un terreno da sminare è fatto di detriti e lamiere.
- `inspect_episode` importava `mujoco` direttamente: era l'unico modulo fuori da
  `simulation/` a farlo, contro la regola che tiene separati simulatore e resto
  del progetto.
- Il premio del target veniva incassato **a ogni passo** dopo l'estrazione. Con
  la terminazione al target non si vedeva; tolta quella, la ricompensa smetteva
  di dire qualcosa sull'ordine.
- `is_success` guardava solo il `collapsed` dell'ultimo passo: un episodio in cui
  la pila era crollata a metà strada risultava «successo».
- Il test stereo cercava la sfera verde, che in Fase 0B non esiste più.
  Agganciato al disco falliva comunque, **60 px contro 70 attesi**: la sfera è
  l'unica forma il cui centroide di sagoma coincide con il centro del corpo da
  qualunque punto di vista. Legare quel test alla composizione di produzione lo
  fa rompere a ogni cambio di scena.

**File:** `physical_ai_mujoco/envs/target_extraction.py`,
`physical_ai_mujoco/simulation/simulator.py`,
`datasets/object_dataset/geometric_objects_pile.json`,
`configs/phase_0b/scene_rules.json`, `scripts/inspect_episode.py`,
`scripts/run_phase_0b.py`, `tests/test_phase_0b.py`,
`tests/data/stereo_scene_rules.json`, `README_oa.md`

**Verifica:** 34 test passati. Confronto su 25 episodi con 6 oggetti (seed 0):
`top` 24/25 successi e mossa peggiore 0,0080 m · `random` 21/25 e 0,0345 m ·
`target` 18/25 e 0,0431 m. Target sepolto in 20 scene su 25.

---

## 2026-09-16 14:46:03 — Pavimento infinito, assestamento a quiete, modalità di misura, rilascio sequenziale

**Chi:** Claude (claude-opus-5), su richiesta di Matteo Paolini

**Cosa:**

- Terreno `floor_infinite`, un `plane` MuJoCo: infinito nelle collisioni.
  `flat_standard` (il tavolo) resta per Fase 0A.
- L'assestamento va **a quiete**, non a tempo fisso: `Simulator.step_until_settled`,
  con le soglie della scena. `settling.py` di Fase 0A ora ci delega.
- Modalità di **misura**: con `terminate_on_target` e `terminate_on_collapse` a
  `false` l'episodio prosegue finché la scena non è vuota. Entrambi a `true`
  ridanno il task. Nessun modulo sa in quale modalità si trova.
- Il disturbo è **per mossa** (`disturbance_step`), più il cumulato
  (`disturbance_total`): il riferimento si riregistra dopo ogni assestamento.
- **Rilascio sequenziale**: gli oggetti cadono uno alla volta da pochi
  centimetri sopra la cima della pila, e quelli dopo il target mirano al target.
- Fase 0B ha dataset e configurazione di simulazione propri: attrito volvente
  0,05 invece di 0,005, timeout di assestamento 20 s invece di 5.

**Perché:**

- `settle_duration` era **0,4 s fissi**. Con 10 oggetti la quota di rilascio
  arrivava a 1,3 m, e una caduta da 1,3 m dura 0,515 s: le pose di riferimento
  venivano registrate **a mezz'aria**. Il «disturbo» misurato al primo passo —
  0,28, 0,62, 0,43 m — era distanza di caduta, non effetto della rimozione, e
  ogni episodio falliva al passo 1 qualunque oggetto si scegliesse.
- Il tavolo da 1×1 m aveva due difetti: un oggetto che ne superava il bordo
  cadeva nel vuoto, producendo spostamenti enormi che inquinavano la misura; e
  la sua dimensione entrava nel calcolo dell'area di rilascio, legando insieme
  due cose indipendenti.
- Con la terminazione al primo crollo si vedeva **una sola rimozione per
  episodio**: impossibile conoscere la distribuzione dei disturbi, e quindi
  impossibile scegliere una soglia con un criterio invece che a occhio.
- Il disturbo cumulato cresce e satura: non dice più quale mossa abbia fatto
  danno.
- Rilasciando tutto insieme, il controllo di non sovrapposizione obbliga a
  partire distanti: o si allarga la scena — e allora gli oggetti non si toccano
  — o si alza la colonna — e allora cadono da metri e l'impatto li disperde.
- Attrito volvente 0,005: un cilindro che atterra su uno spigolo rotola via per
  25 cm, misurato tracciando ogni rilascio. Su 10 scene, target sepolto **0/10**
  con 0,005 contro **6/10** con 0,05. Con timeout 5 s, **9 scene su 25**
  restavano in movimento — cioè misurate male; con 20 s, 1 su 25.

**File:** `datasets/ground_dataset/basic_grounds.json`,
`datasets/object_dataset/geometric_objects_pile.json` (nuovo),
`configs/phase_0b/{env,scene_rules,simulation}.json`,
`physical_ai_mujoco/scene/{scene_description,scene_generator,dataset_loader}.py`,
`physical_ai_mujoco/simulation/{simulator,settling,mujoco_builder}.py`,
`physical_ai_mujoco/envs/target_extraction.py`, `scripts/*.py`,
`tests/test_phase_0b.py`, `README_oa.md`

**Verifica:** 34 test passati, eseguiti in ambiente con MuJoCo 3.12 e Gymnasium
1.3. Due bug trovati proprio dai test: il rilascio leggeva le pose dalla
descrizione della scena invece che dal simulatore (con `resample_shapes=False`
ogni episodio ripartiva identico), e il premio del target era pagato a ogni
passo.

**Nota:** Fase 0A non è stata toccata e continua a produrre scene
byte-identiche a parità di seed.

---

## 2026-09-16 14:11:14 — Integrazione dell'ispezione nel flusso normale

**Chi:** Claude (claude-opus-5), su richiesta di Matteo Paolini

**Cosa:** voce 6 del menu di `main.py` (Fase 0A diventa 7, i test 8).
`run_phase_0b` stampa catalogo e appoggi **prima** che la policy scelga, indica
quale oggetto è ogni indice, e a fine episodio stampa **l'ordine di rimozione
effettivo**. Opzione `--no-catalogue`.

**Perché:** Matteo ha chiesto che lo strumento vivesse dentro il flusso del
progetto, non a fianco. Una sequenza di indici senza il catalogo non dice
niente: non si sa cosa fosse l'indice 2, né se stava sopra o sotto il target.

**File:** `main.py`, `scripts/run_phase_0b.py`, `README_oa.md`

**Verifica:** eseguito da Matteo sul suo Mac; ha esposto due bug di reporting,
corretti nella voce successiva.

---

## 2026-09-16 14:05 — Strumento di ispezione di un episodio

**Chi:** Claude (claude-opus-5), su richiesta di Matteo Paolini

**Cosa:** nuovo `scripts/inspect_episode.py`. Stampa il catalogo degli oggetti,
il grafo degli appoggi letto dai contatti veri di MuJoCo, e le azioni passo per
passo. Quattro politiche di riferimento: `manual`, `random`, `top`, `target`.

**Perché:** senza un modo di vedere cosa c'è in scena e chi poggia su chi, non
si può giudicare se un ordine di rimozione sia sensato — né accorgersi che il
target è già in cima e che quella scena non pone alcun problema.

**File:** `scripts/inspect_episode.py` (nuovo)

**Verifica:** eseguito su episodi singoli; ha reso visibile che il target era
scoperto in una frazione notevole delle scene.

---

*Le date delle prime voci sono i momenti in cui i file sono stati scritti sul
Mac, ricavati dai timestamp dei file. Da qui in avanti ogni voce porta l'ora
esatta al secondo.*
