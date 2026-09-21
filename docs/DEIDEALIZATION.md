# Roadmap di de-idealizzazione

## Metodo

Ogni passaggio sostituisce una idealizzazione, mantenendo invariati gli altri
componenti. Prima e dopo la sostituzione si usano le stesse scene, gli stessi
seed e metriche confrontabili.

Per ogni fase si registrano:

1. idealizzazione rimossa;
2. implementazione sostituita;
3. componenti mantenuti invariati;
4. test del modulo e dei contratti;
5. misura della perdita o del guadagno;
6. gate per procedere.

## Mappa delle idealizzazioni

| Idealizzazione | Cosa nasconde | Rimossa o verificata | Misura principale |
|---|---|---|---|
| Scelta casuale | Valore della strategia | 1A | Guadagno di DECIDE rispetto alle baseline |
| Stato esatto del simulatore | Errori e lacune percettive | 1B | Errore stereo rispetto a MuJoCo |
| Un solo modello di errore sensoriale | Variabilita possibile nel reale | 1C | Robustezza su osservazioni degradate e stereo |
| Rimozione ideale | Limiti e fallimenti del robot | 2 | Costo e successo dell'esecuzione con UR5 simulato |
| Stereo provata senza braccio | Occlusioni causate dall'UR5 | Gate 2 | Scarto percettivo con e senza braccio |
| Stima puntuale | Incertezza percettiva | Futuro 4 | Calibrazione del belief |
| Scene geometriche semplici | Varieta di forme e proprieta fisiche | 5 | Generalizzazione a scene rappresentative |
| Sensori e robot simulati | Divario sim2real | 6 | Prestazione reale rispetto alla simulazione |

## Fasi

### 0A - mondo simulato

Dataset e configurazione alimentano il generatore, la descrizione della scena,
il builder MJCF e MuJoCo. Si validano riproducibilita, fisica e assestamento.

### 0B - primo ciclo completo

Si aggiungono runner, Environment Gymnasium, task e reward, osservazione
esatta, decisione casuale, esecuzione ideale e log. Lo scopo e ottenere un
ciclo completo e misurabile, non una strategia efficace.

### 1A - strategia

Una policy sostituisce la scelta casuale. L'osservazione resta esatta e
l'esecuzione resta ideale. Si aggiungono training e valutazione contro casuale,
target immediato, altezza e ordine ottimo.

### 1B - prima percezione stereo

OBSERVE acquisisce camera B/N e LiDAR, segmenta, stabilizza gli ID, localizza
in 3D e completa la PFM-1 con il CAD quando possibile. La pipeline produce una
`Observation` codificata a slot fissi per PPO. Il
PrivilegedState MuJoCo rimane disponibile soltanto come riferimento per
misurare l'errore percettivo.

### 1C - robustezza sensoriale

La stessa `SimulatedSensorSource` applica a runtime disturbi fotometrici,
dropout ed errori LiDAR riproducibili. DECIDE puo quindi addestrarsi su errori
controllati senza cambiare contratto.

### 2 - manipolatore simulato

IdealRemovalExecutor viene sostituito da SimulatedUR5Executor. ESEGUI contiene
la policy che sceglie primitiva, contatto, forza e traiettoria. Entrano stato
del robot ed esito motorio.

### Gate 2 - stereo con manipolatore

Si ripete la validazione percettiva con l'UR5 presente e in movimento. Si
misurano occlusioni, oggetti persi e perdita decisionale rispetto alla scena
senza braccio.

### 3

Non e una fase autonoma. I compiti precedentemente previsti sono distribuiti
fra 1B e 2.

### Futuro 4 - belief calibrato

Si introdurra soltanto dopo aver misurato errori percettivi reali e simulati.

### 5 - scene rappresentative

Si estendono asset e variazioni fisiche controllate nella stessa pipeline. Una
quota crescente di varieta puo essere introdotta prima; la fase 5 contiene la
validazione sistematica sulla distribuzione rappresentativa.

### 6 - sistema reale

Adapter ROS 2 collegano RealStereoObserver e RealUR5Executor a sensori e robot
reali. DECIDE e i contratti centrali non vengono riscritti.

## Matrice delle implementazioni

| Fase | OBSERVE | DECIDE | EXECUTE |
|---|---|---|---|
| 0B | ExactObserver | RandomDecider | IdealRemovalExecutor |
| 1A | ExactObserver | TeacherDecider | IdealRemovalExecutor |
| 1B | SensorObserver + encoder | PPO sensoriale | IdealRemovalExecutor |
| 1C | SensorObserver disturbato + encoder | PPO sensoriale robusto | IdealRemovalExecutor |
| 2 | SimulatedStereoObserver | RobustDecider | SimulatedUR5Executor |
| 6 | RealStereoObserver | policy trasferita/adattata | RealUR5Executor |

## Gate comuni

Una fase puo essere dichiarata completata soltanto se:

- il componente nuovo rispetta il contratto pubblico;
- i test delle fasi precedenti continuano a passare;
- la valutazione usa scene e seed separati dal training;
- configurazione, seed, versione del codice e artefatti sono registrati;
- la perdita o il guadagno rispetto alla fase precedente e quantificato;
- e possibile attribuire il cambiamento all'idealizzazione rimossa.
