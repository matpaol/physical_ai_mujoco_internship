# Architettura software

## Obiettivo

Il progetto evolve per de-idealizzazione progressiva: parte da un ciclo completo
ma semplificato e sostituisce una idealizzazione alla volta. La struttura del
software resta organizzata per responsabilita stabili, mentre le fasi scelgono
implementazioni diverse di quelle responsabilita.

Principio guida:

> Confini stabili, interfacce semplici, implementazioni sostituibili e nessuna
> complessita anticipata.

## Stile architetturale

Il progetto resta un monolite modulare Python. I moduli comunicano in-process
attraverso oggetti e contratti espliciti. ROS 2 e Docker non sono meccanismi di
comunicazione interna fra i moduli.

Si preferiscono:

- oggetti per componenti con stato, configurazione o ciclo di vita;
- classi astratte piccole per i ruoli sostituibili;
- dataclass per i dati scambiati;
- composizione all'ereditarieta profonda;
- funzioni per calcoli puri e trasformazioni locali.

## Responsabilita stabili

### SCENE

Definisce e genera la configurazione iniziale: oggetti, terreno, target, pose,
proprieta fisiche e randomizzazione.

### SIMULATION

Costruisce ed esegue il mondo MuJoCo. Mantiene privati `MjModel`, `MjData`,
contatti, snapshot e dettagli del solver. Espone soltanto operazioni necessarie
agli altri moduli.

### OBSERVE

Produce cio che il sistema decisionale puo conoscere. Le implementazioni
previste sono, in ordine di de-idealizzazione: stato esatto, stereo simulata,
osservazioni degradate e stereo reale.

### DECIDE

Sceglie il prossimo oggetto da rimuovere. Opera su una Observation e produce
una ObjectDecision. Non dipende da MuJoCo, ROS 2 o dettagli percettivi interni.

### EXECUTE

Decide come realizzare la rimozione richiesta e restituisce un
ExecutionOutcome. Inizia con una rimozione ideale; in seguito conterra la
policy di manipolazione dell'UR5 simulato e poi reale.

### TASK

Definisce il problema scientifico: target, disturbo, reward, crollo, successo,
terminazione e metriche di transizione. Non implementa la fisica.

### ENVIRONMENT

Adatta il ciclo al protocollo Gymnasium. Riceve un'azione, delega l'esecuzione,
fa evolvere la simulazione, valuta il task e produce la nuova osservazione.
DECIDE resta esterno all'Environment.

### EXPERIMENTS ed EVALUATION

Gli esperimenti gestiscono training, raccolta dati, seed, configurazioni e
artefatti. La valutazione confronta policy e baseline sulle stesse scene e
produce metriche riproducibili.

### VISION_TRAINING

Produce il detector usato da OBSERVE: genera dataset da SIMULATION con
randomizzazione visiva, addestra il segmenter e ne misura il riconoscimento
del target per % di sagoma visibile. Non e codice deployable: consegna solo
pesi e una scheda. Non importa MuJoCo; cambia l'aspetto della scena solo con
`Simulator.apply_visual_conditions()`. Dettaglio in
`physical_ai_mujoco/vision_training/README.md`.

## Contratti iniziali

I contratti vengono stabilizzati, testati e versionati; non sono considerati
immutabili prima che gli esperimenti ne dimostrino l'adeguatezza.

- `Observation`: informazione disponibile alla policy operativa.
- `PrivilegedState`: informazione esatta disponibile soltanto in simulazione,
  training e valutazione.
- `ObjectDecision`: prossimo oggetto che DECIDE richiede di rimuovere.
- `ExecutionOutcome`: esito dell'esecuzione della richiesta.
- `TaskOutcome`: reward, terminazione e metriche della transizione.

Lo stato completo interno di MuJoCo non e un contratto pubblico. Rimane dentro
SIMULATION e viene trasformato in PrivilegedState o Observation quando serve.

## Ciclo operativo

```text
Runner/Agent
    -> DECIDE(Observation)
    -> ObjectDecision
    -> Environment.step(...)
        -> EXECUTE
        -> SIMULATION
        -> TASK
        -> OBSERVE
    -> nuova Observation
```

Si prende una decisione per volta e si riosserva dopo ogni esecuzione.

## Sim2real

Il ponte sim2real e costituito dai contratti di OBSERVE, DECIDE ed EXECUTE:

```text
SimulatedStereoObserver --\
                          -> Observation -> DECIDE -> ObjectDecision
RealStereoObserver ------/

ObjectDecision -> SimulatedUR5Executor
ObjectDecision -> RealUR5Executor
```

MuJoCo resta interno al ramo simulato. ROS 2 entra attraverso adapter usati
dalle implementazioni reali di OBSERVE ed EXECUTE. Nessun contratto pubblico
deve contenere tipi MuJoCo o messaggi ROS 2.

## Regole di dipendenza

- DECIDE non importa MuJoCo, Gymnasium, ROS 2 o detector interni.
- TASK non importa ROS 2 e non modifica direttamente il simulatore.
- SCENE non conosce policy, reward o robot.
- OBSERVE non importa algoritmi di training.
- `main.py` legge la configurazione, costruisce i componenti e avvia il runner.
- Le fasi della roadmap non diventano package `phase_*`.

## Regola anti-overengineering

Una nuova astrazione deve risolvere un problema attuale, rappresentare un
confine scientifico reale, proteggere una dipendenza esterna oppure avere una
seconda implementazione prevista nella fase immediatamente successiva.

Non si progettano ora belief, grasp, traiettorie, messaggi ROS 2 o gerarchie
per DLO. Verranno introdotti quando la rispettiva idealizzazione sara rimossa.



## Implementazione della prima migrazione

La descrizione concreta, i percorsi pubblici, le compatibilita e le verifiche
sono in [RESTRUCTURING.md](RESTRUCTURING.md). `envs` e il package dell'Environment.
La voce 7 del menu sceglie i profili `configs/experiments/`: i numeri di fase
rimangono nei dati e nella presentazione. `ExactObserver` restituisce Observation,
mentre `privileged_state` fornisce separatamente PrivilegedState per supervisione.
La cattura stereo grezza non viene presentata come ricostruzione percettiva.
