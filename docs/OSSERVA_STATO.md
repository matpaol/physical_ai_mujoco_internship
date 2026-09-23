# OSSERVA: stato al congelamento (`osserva-v1`)

Dal 23/09/2026 la percezione reale e' **congelata** per decisione presa con il
tutor: DECIDI ed ESEGUI si sviluppano sull'osservazione sintetica
dell'**oracolo MuJoCo** (`OracleObserver`, profilo `configs/experiments/oracolo.json`).
Questo file dice cosa c'e', quanto funziona e da dove ripartire. Il tag git
`osserva-v1` segna il codice di questo stato.

---

## 1. Cosa c'e'

| Osservatore | Profilo / componente | Ingresso | Stato |
|---|---|---|---|
| `ExactObserver` | `"observer": "exact"` (0B, 1A) | stato MuJoCo | pose esatte, supporti **stimati** con regola geometrica |
| `OracleObserver` | `"observer": "oracle"` (`oracolo.json`) | stato MuJoCo + `PrivilegedState` | pose esatte, supporti **dai contatti** (`OracleRelationEstimator`). Solo simulazione: banco di prova di DECIDI/ESEGUI |
| `DegradedObserver` | `"observer": "degraded"` | stato MuJoCo + rumore | per robustezza di DECIDI |
| `SensorObserver` | `"observer": "sensor_learned"` (1B, 1C) | stereo B/N + LiDAR | pipeline deployable: YOLO-seg → LiDAR → geometria → CAD PFM-1 → tracking |

Tutti producono la stessa `Observation` (scena + grafo dei supporti +
incertezza): DECIDI non sa quale osservatore c'e' a monte. Si cambia
osservatore cambiando **una riga del profilo**, non il codice.

## 2. Quanto funziona (numeri misurati)

**Detector `sim_dr_v1_rtx3050`** (22/09, validation di `sim_dr_v1`): maschere
P 0,90, R 0,78, mAP50 0,84, mAP50-95 0,64. Scheda in
`outputs/detector_weights/sim_dr_v1_rtx3050.json`, run completo sull'archivio
Drive (`04_visore/`).

**Benchmark OSSERVA** (`outputs/observe_tests/20260923_135955`, 4 scene, 4-10
oggetti, disturbi casuali, maschere *oracle*, nessun detector):

| Sorgente | Oggetti trovati | Errore posizione | Target | Precisione supporti |
|---|---|---|---|---|
| exact | 100% | 0 | 4/4 | 0,43 |
| degraded | 94% | 2,4 cm | 4/4 | 0,29 |
| rgbd | 97% | 2,9 cm | 4/4 | 0,40 |
| fusion_oracle (camera + LiDAR) | 91% | 4,2 cm | 4/4 | 0,22 |
| stereo sola | 26% | 6,7 cm | 2/4 | n.d. |

Quattro scene sono poche: numeri indicativi.

La precisione dei supporti bassa **anche con pose esatte** (0,43) viene dalla
regola geometrica (`GeometricRelationEstimator`): su tre scene di prova trova
14, 12 e 6 candidati contro 5, 3 e 3 appoggi veri. E' il motivo per cui
DECIDI parte dall'oracolo, che legge i contatti.

## 3. Cosa resta aperto (per quando si riprende)

1. **Valutazione per fascia di sagoma visibile** del detector: mai eseguita
   (manca `observability` nella scheda). Comando:
   `python physical_ai_mujoco/observe/main_test_osserva.py evaluate-detector outputs/detector_weights/sim_dr_v1_rtx3050.pt datasets/generated/sim_dr_v1 --split test`.
2. **Benchmark con il detector appreso** (`--mode fusion_learned --weights ...`)
   sulle stesse scene di `fusion_oracle` (stesso `--seed`), almeno 20 scene.
3. **Scelta del modello**: senza `--weights` il menu e il builder prendono
   `pfm_1_seg.pt` o il `pfm_1_seg*.pt` piu' recente, non `sim_dr_v1_rtx3050.pt`.
   Nei profili 1B/1C si puo' fissare con `"detector_weights"` dentro
   `sensor_observation` (il builder lo legge gia').
4. **Supporti**: sostituire la regola geometrica con uno stimatore migliore e
   misurarlo contro l'oracolo (che ora esiste come riferimento).
5. **Stereo sola** non basta (26% degli oggetti): la strada e' la fusione con
   il LiDAR.

## 4. L'oracolo, in breve

Segue `observa.md`, Blocco 3: `OracleRelationEstimator` e' l'implementazione
"simulation-only" prevista accanto a `GeometricBaseline` e `LiRelationEstimator`,
e legge la verita' dal ramo `PrivilegedState`, non dai prodotti deployable.

```text
Simulator ─> ExactObserver.privileged_state() ─> PrivilegedState.contact_supports
                                                          │
SceneState (pose esatte) ────────────────> OracleRelationEstimator ─> PhysicalRelationState
                                                          │            (support, score 1.0,
                                                          │             "mujoco_contacts_oracle")
SceneState + PhysicalRelationState + UncertaintyState ─> Observation Builder ─> Observation ─> DECIDE
```

- `PrivilegedState.contact_supports`: coppie `(sotto, sopra)` fra oggetti
  presenti, lette da `simulator.support_graph()`; pavimento e oggetti rimossi
  esclusi. `SensorEvidence` non contiene dati privilegiati.
- **E' il grafo degli appoggi da contatto, non la verita' causale delle
  dipendenze** (che per `observa.md` richiede interventi/rollout simulati).
  Limiti: vale su scena assestata; fra due corpi a contatto "sostiene" quello
  col centro piu' basso, anche se il contatto e' laterale.
- `OracleObserver` legge di proposito il ramo privilegiato: non va mai usato
  come osservatore deployable. Scena e incertezza sono identiche a `ExactObserver`.
