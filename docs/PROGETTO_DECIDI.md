# Progetto di DECIDI: decisioni prese

Documento di progetto del decisore, scritto il 25/09/2026 alla fine della
sessione di progettazione con Matteo. Raccoglie **cosa è deciso**, **cosa resta
aperto** e **in che ordine si implementa**. Le figure sono in
[diagrammi/architettura_progetto.drawio](diagrammi/architettura_progetto.drawio)
(nove pagine; si apre in VS Code con l'estensione "Draw.io Integration").
Lo stato dei passi è in [PIANO_LAVORO.md](PIANO_LAVORO.md).

Le specifiche canoniche sono in `software engineering/data/software_architecture/`:
le decisioni sono registrate come **D47** (DECIDE per la 1A) e **D48** (TASK e
reward) in `02_decisioni.md` v22, e riportate in `decide.md` v6, `esegue.md` v6,
`environment.md` v7 e `interfaces.md` v10. Questo documento ne è il dettaglio
operativo per il codice.

Nelle regole di architettura l'uscita di DECIDI si chiama `ObjectPlan` (lista
ordinata); nella 1A il codice usa `ObjectDecision`, cioè l'`ObjectPlan` di
lunghezza uno previsto da D40.

---

## 1. Confini dei moduli

| Modulo | Riceve | Produce | Non fa |
|---|---|---|---|
| ENVIRONMENT | la scena | l'ordine dei passi, il `DecisionInput` | non sceglie oggetti, non calcola la reward |
| OSSERVA | la scena | `Observation`; in simulazione anche `PrivilegedState` | non decide |
| DECIDI | `DecisionInput` (il teacher anche `PrivilegedState`) | `ObjectDecision(object_id)`: un solo oggetto per passo, target compreso | non sceglie come togliere, non vede la reward |
| ESEGUI | `ObjectDecision` | fatti: `FeasibilityAssessment` prima del tentativo, `ExecutionOutcome` dopo | non giudica se la scelta era buona |
| TASK | posizioni vere dopo il passo, esito | reward, terminated, disturbo, crollo, successo | non osserva, non sceglie, non esegue |
| TRAINER / EVALUATION | transizioni e metriche | policy aggiornata, confronti | esiste solo in addestramento |

- **DECIDI è l'agente**: sta fuori dall'environment, che gli manda il
  `DecisionInput` e riceve l'`ObjectDecision`.
- **TASK è il modulo di tutte le funzioni di reward.** Oggi contiene quella di
  DECIDI (`TargetExtractionTask`); quella delle skill di ESEGUI sarà una classe
  separata nello stesso modulo, usata dal proprio ciclo di training.
- **Chi giudica una scelta**: ESEGUI dice cosa è successo, TASK quanto è
  costato, il trainer cosa imparare. Il teacher può dire "avrei scelto X"
  (supervisione nel training), ESEGUI no.

## 2. Il ciclo di un passo

1. Solo al reset: OSSERVA produce `Observation_0` (e `PrivilegedState_0`).
2. L'environment compone il `DecisionInput` e lo passa a DECIDI.
3. DECIDI restituisce `ObjectDecision(object_id)`.
4. ESEGUI valuta la fattibilità e tenta. In futuro avrà un ciclo interno con
   feedback Fₜ,ₖ: quei feedback **non** richiamano DECIDI e non sostituiscono
   l'osservazione successiva.
5. L'environment attende l'assestamento e fa osservare di nuovo: `Observation_t+1`.
6. TASK calcola la reward sulle posizioni vere; ricorda da sé le posizioni
   "prima" (`reference_positions`).
7. Fine episodio se `terminated` (TASK) **o** `truncated` (64 passi, wrapper
   `TimeLimit` di Gymnasium). Altrimenti `Observation_t+1` diventa l'input del
   passo dopo e il `DecisionContext` si aggiorna.
8. Solo in training la transizione va al trainer.

## 3. Contratti

- **`DecisionInput`** (previsto) = `observation` + `context`. Un solo oggetto,
  così quando arriverà `RobotState` (fase 2) non cambia la firma dei decisori.
  Firme: student `decide(decision_input)`, teacher `decide(decision_input,
  privileged)`. `PrivilegedState` resta un argomento a parte, mai un campo
  opzionale che lo student potrebbe leggere per sbaglio.
- **`DecisionContext`** (previsto): per ogni oggetto tentativi, richieste non
  eseguibili, tentativi falliti, ultima azione; `step_index`. **Niente
  `remaining_steps`**: il limite di 64 passi è una convenzione del training,
  sul robot non esiste. Serve a evitare il ciclo "A non eseguibile → B → di
  nuovo A". Limite noto: i contatori sono per `object_id`, quindi con la
  percezione reale dipendono dalla stabilità del tracker.
- **`FeasibilityAssessment`** (previsto): executable, failure_reason, skill.
- **`ExecutionOutcome`**: oggi object_id, removed, failure_reason; previsti
  attempted, collision, durata, costo.
- Tutti i contratti nuovi vanno in `contracts/`.

## 4. Dentro DECIDI

Encoder → policy → decoder.

- **Encoder**: trasforma il `DecisionInput` in vettore **e maschera**. Sta in
  `decide/encoding.py` (spostato da `observe/` il 24/09).
- **Slot**: la rete ha **K = 12** slot, fissati nel profilo dell'esperimento e
  non dalla singola scena (oggi sono `2 × object_count`, da cambiare). Una scena
  con meno oggetti lascia slot vuoti. Con più di K oggetti: prima il target, poi
  i più alti, a parità l'`object_id`. Un oggetto tiene lo stesso slot per tutto
  l'episodio.
- **Maschera**: la calcola l'encoder, la usa la policy (`MaskablePPO`, che
  sceglie solo fra slot validi). Blocca **solo** slot vuoti e oggetti già
  rimossi, **mai** la fattibilità: la rimovibilità DECIDI la impara dagli esiti
  (D40).
- **Decoder**: slot scelto → `object_id`, con la tabella degli slot dell'encoder.
- **Stato per episodio**: l'encoder ricorda gli slot, quindi il decisore
  student ha un `reset()` chiamato all'inizio di ogni episodio.

## 5. Reward (oggi)

Da `task/core.py` e `configs/phase_0b/env.json`:

- disturbo = massimo spostamento degli oggetti **non target** fra prima e dopo;
- crollo = disturbo > 0,05 m;
- reward = −0,05 per rimozione − 4 × disturbo + 1 se il target è appena
  estratto − 1 se crollo;
- azione su un oggetto già rimosso: −0,05, nessuna misura;
- `is_success` = target estratto e nessun crollo nell'episodio;
- la reward usa sempre le posizioni vere, anche quando l'osservazione sarà
  rumorosa (1C).

## 6. Training e valutazione

- Due rami: **PPO** (transizioni dello student) e **imitazione del teacher**
  (`Observation` + scelta del teacher → behaviour cloning).
- Algoritmo: `MaskablePPO` (sb3-contrib) al posto del PPO normale.
- Valutazione su scene identiche (stessi semi) per `random`, `highest`,
  `immediate_target`, student e teacher; semi di training separati da quelli di
  valutazione. Metriche per episodio: successo, decisioni, rimozioni, tentativi
  falliti, scelte non eseguibili, disturbo, crolli.

## 7. Ancora da decidere

| Tema | Domanda |
|---|---|
| ESEGUI | Quando l'esecutore ideale risponde "non eseguibile" (candidato: target coperto)? |
| ESEGUI | Chi fornisce Fₜ,ₖ: OSSERVA (nuova interfaccia, da scrivere in `interfaces.md`) o i sensori del robot? |
| ESEGUI | L'`ExecutionOutcome` arriva prima o dopo l'assestamento? |
| TASK | Crollo: fallimento che chiude l'episodio o solo un costo? Pesi della reward; costo di una richiesta non eseguibile. |
| DECIDI | PPO a slot oppure GNN; pianificatore sul grafo come baseline; teacher T1 o T2. |
| Valutazione | Numero di scene e insiemi di semi. |
| OSSERVA-oracolo | Correzione del grafo (13% di archi invertiti) prima dei risultati di tesi. |

## 8. Ordine di implementazione

1. Modalità `observation` dell'environment con l'oracolo: l'environment
   restituisce l'`Observation` codificata, l'azione è lo slot; `reset()` del
   decisore. K = 12 dal profilo.
2. Decisore student: encoder → policy → decoder, dietro l'interfaccia `Decider`.
3. Training con `MaskablePPO` sulla nuova modalità.
4. Confronto con le baseline sulle stesse scene.
5. Contratti `DecisionInput`, `DecisionContext`, `FeasibilityAssessment` quando
   ESEGUI potrà rifiutare.
