# Struttura di `Observation` e `PrivilegedState`

Cosa riceve DECIDI a ogni passo, campo per campo, con un esempio reale e una
valutazione di quanto la struttura regga per decidere. Vale per l'osservatore
**oracolo** (profilo `configs/experiments/oracolo.json`, branch
`osserva-oracolo` / `decidi`); gli osservatori reali (stereo, LiDAR) producono
la stessa struttura con valori meno ideali.

Definizioni nel codice: `physical_ai_mujoco/contracts/observation.py`
(`Observation`) e `physical_ai_mujoco/contracts/core.py` (`PrivilegedState`).

---

## 1. Due rami, due destinatari

| | `Observation` | `PrivilegedState` |
|---|---|---|
| Chi la riceve | ogni decisore (student e teacher) | solo il teacher, gli oracoli, la valutazione |
| Esiste nel mondo reale? | sì: è ciò che OSSERVA può stimare dai sensori | no: solo in simulazione |
| Chi la produce | `Observer.observe(source, context)` | `ExactObserver.privileged_state(simulator, target_id)` |
| Interfaccia di DECIDI | `Decider.decide(observation)` | `TeacherDecider.decide(observation, privileged)` |

Nel ciclo (`experiments/episode.py`, `run_episode`) entrambe vengono ricalcolate
a ogni passo, dopo che ESEGUI ha tolto l'oggetto precedente e la scena si è
assestata.

**Convenzioni comuni**

- frame `"world"`: z verso l'alto, origine sul pavimento; lunghezze in metri;
- `quaternion` in ordine MuJoCo `(w, x, y, z)`;
- `size` = ingombro **pieno** lungo gli assi del corpo: box `(x, y, z)`,
  cilindro `(2r, 2r, h)`, mesh = bounding box;
- `timestamp` = tempo di simulazione in secondi.

---

## 2. `Observation` — l'unico input dello student

Tre blocchi con gli stessi ID, lo stesso frame e lo stesso timestamp
(controllati da `validate_observation`).

### 2.1 `scene: SceneState` — cosa c'è e dove

| Campo | Tipo | Significato | Oracolo |
|---|---|---|---|
| `objects` | tuple di `SceneObject` | oggetti **presenti**; quelli già rimossi spariscono | tutti, esatti |
| `target_id` | `str \| None` | ID del PFM-1 | noto |
| `ground_height` | `float` | quota del pavimento | `0.0` |
| `bounds` | `(x0, x1, y0, y1) \| None` | area di lavoro | `None` |
| `frame`, `timestamp` | `str`, `float` | | `"world"`, t sim |
| `protected_zone`, `target_safe_zone`, `obstacle_drop_zone` | tuple \| `None` | zone del compito | `None` (non usate ancora) |

`SceneObject`:

| Campo | Tipo | Significato | Oracolo |
|---|---|---|---|
| `object_id` | `str` | `object_000`, `object_001`, … stabile nell'episodio | |
| `role` | `"target" \| "obstacle"` | | |
| `type_id` | `str \| None` | `slab`, `disc`, `plank`, `pfm_1_target`, … | noto |
| `position` | `(x, y, z) \| None` | centro del corpo [m] | esatto |
| `quaternion` | `(w, x, y, z) \| None` | orientamento | esatto |
| `shape` | `str \| None` | `box`, `cylinder`, `sphere`, `mesh` | |
| `size` | `(x, y, z) \| None` | ingombro pieno [m] | |
| `detected` | `bool` | l'oggetto è stato rilevato | sempre `True` |
| `perception_quality` | `float \| None` | fiducia del rilevamento, 0-1 | `1.0` |

### 2.2 `relations: PhysicalRelationState` — chi poggia su chi

| Campo | Tipo | Significato | Oracolo |
|---|---|---|---|
| `object_ids` | tuple di `str` | stesso ordine di `scene.objects` | |
| `relations` | tuple di `PhysicalRelation` | archi del grafo | un arco per coppia a contatto |
| `estimator` | `str` | chi ha stimato il grafo | `"mujoco_contacts_oracle"` |
| `available` | `bool` | il grafo esiste | `True` |

`PhysicalRelation(source_id, target_id, relation_type, relation_score)`:

- **verso**: `source_id` **sostiene** `target_id` (sotto → sopra);
- `relation_type`: `"support"`;
- `relation_score`: nell'oracolo sempre `1.0` (arco sì/no).

La proprietà `dependency_graph` restituisce, per ogni oggetto, gli oggetti che
gli stanno sopra. Gli oggetti che bloccano il target sono tutti quelli
raggiungibili dal target seguendo gli archi.

**Come l'oracolo costruisce gli archi** (`Simulator.support_graph`): per ogni
coppia di corpi con un contatto MuJoCo attivo, quello con il **centro più
basso** è preso come sostegno. Il contatto con il pavimento viene scartato.

### 2.3 `uncertainty: UncertaintyState` — quanto fidarsi

| Campo | Tipo | Significato | Oracolo |
|---|---|---|---|
| `objects` | tuple di `ObjectUncertainty(object_id, visible, detection_quality, pose_quality)` | per oggetto | tutti visibili, qualità `1.0` |
| `unknown_space` | `bool` | esistono zone non osservate | `False` |
| `provider` | `str` | | `"exact"` |

---

## 3. `PrivilegedState` — la verità del simulatore

| Campo | Tipo | Significato |
|---|---|---|
| `objects` | tuple di `ObjectObservation` | **tutti** gli oggetti dell'episodio, anche quelli rimossi (`present=False`) |
| `contact_supports` | tuple di `(lower_id, upper_id)` | gli stessi archi dell'oracolo, solo fra oggetti presenti |

`ObjectObservation`:

| Campo | Tipo | Unità | Note |
|---|---|---|---|
| `object_id` | `str` | | |
| `position`, `quaternion` | come sopra | m | |
| `linear_velocity`, `angular_velocity` | `(x, y, z)` | m/s, rad/s | su scena assestata ~1e-5 |
| `mass` | `float` | kg | |
| `friction` | `float` | | attrito radente |
| `present`, `is_target` | `bool` | | |
| `type_id`, `shape`, `size` | come sopra | m | |
| `center_of_mass` | `(x, y, z)` | m | scostamento dal centro geometrico, frame del corpo |
| `torsional_friction`, `rolling_friction` | `float` | | |

`as_vector()` produce il vettore storico del teacher 1A: 17 valori per oggetto
(posizione 3, quaternione 4, velocità 3+3, massa, attrito radente, presente,
target). I campi aggiunti dopo (forma, dimensioni, centro di massa, attriti
torsionale e di rotolamento) **non** vi entrano.

### Chi vede cosa

| Informazione | `Observation` | `PrivilegedState` |
|---|---|---|
| posa, forma, dimensioni, tipo, ruolo | sì | sì |
| grafo dei sostegni | sì (`relations`) | sì (`contact_supports`) |
| qualità della percezione | sì | — |
| oggetti già rimossi | no (spariscono) | sì (`present=False`) |
| velocità, massa, attriti, centro di massa | no | sì |

---

## 4. Esempio: `docs/esempi/oracolo_seed7/`

Generato con:

```
python physical_ai_mujoco/observe/main_test_oracle.py --seed 7 --output docs/esempi/oracolo_seed7
```

(i file finiscono in `scene_001/`; qui sono stati spostati nella cartella.)
Scena: seed `1390851128`, scene seed `2001665126`, 5 oggetti, target
`object_002`.

| File | Contenuto |
|---|---|
| `observation.json` | l'`Observation` completa |
| `privileged_state.json` | il `PrivilegedState` completo |
| `support_graph.dot`, `.svg` | il grafo dei sostegni (sotto → sopra) |

| Oggetto | Tipo | Forma | z centro [m] | Massa [kg] |
|---|---|---|---|---|
| `object_000` | slab | box | 0,0157 | 0,202 |
| `object_001` | disc | cylinder | 0,0110 | 0,176 |
| `object_002` | **pfm_1_target** | mesh | 0,0100 | 0,075 |
| `object_003` | plank | box | 0,0147 | 0,160 |
| `object_004` | slab | box | 0,0307 | 0,373 |

Archi dell'oracolo: `000 → 004`, `001 → 004`, `002 → 000`. Letto così, il
target ha sopra `000`, che ha sopra `004`: per liberarlo servirebbero due
rimozioni (`004`, poi `000`).

---

## 5. La struttura ha senso? Cosa regge e cosa no

### Cosa regge

- Separazione pulita student/teacher: nessun dato fisico riservato entra
  nell'`Observation`.
- ID, frame e timestamp allineati e controllati a ogni passo.
- Stessa forma per oracolo e osservatori reali: DECIDI si può sviluppare
  sull'oracolo e poi passare ai sensori senza cambiare interfaccia.

### Problema 1 — il verso dell'arco è sbagliato in un caso su otto

La regola "il centro più basso sostiene" non guarda le forze. Misura sulla
scena d'esempio, con `mj_contactForce` (forza verticale ricevuta da ogni
oggetto; le somme coincidono con i pesi, `m·g`):

| Coppia | Normale del contatto `\|n_z\|` | Forza verticale | Chi regge chi |
|---|---|---|---|
| 000 – 004 | 0,96 | 1,10 N su 004 | 000 regge 004 (30% del suo peso) ✔ |
| 001 – 004 | **0,00** (laterale) | 0,75 N su 004, per attrito | 001 regge 004 (20% del suo peso) ✔ |
| 002 – 000 | 0,63 | **0,20 N sul target** | **000 regge il target** (27% del suo peso) ✘ |

L'oracolo dice `002 → 000` (il target regge 000). In realtà il target è
inclinato e appoggia un bordo su `000`: **sul target non preme nulla**, si
potrebbe estrarre subito. Il grafo gli attribuisce due ostacoli che non ha.

Su 30 scene casuali (4-10 oggetti, `random.Random(123)`):

| Misura | Valore |
|---|---|
| archi dell'oracolo | 147 |
| archi con il verso confermato dalla forza verticale | 128 (87%) |
| archi con il verso **invertito** | **19 (13%)** |
| archi senza carico (< 0,01 N) | 0 |
| archi con contatto laterale (`\|n_z\|` < 0,5) | 10 (7%) |
| scene in cui l'oracolo sbaglia se il target è coperto | **6 su 30** (3 lo dà coperto quando è libero, 3 libero quando è coperto) |

Metodo: per ogni contatto attivo, forza con `mujoco.mj_contactForce` riportata
nel frame mondo; si somma la componente verticale ricevuta da ciascun corpo
da ciascun altro. Un arco `lower → upper` è confermato se `upper` riceve da
`lower` più di 0,01 N verso l'alto, invertito se ne riceve più di 0,01 N
verso il basso. Lo script di questa analisi non è ancora nel repository.

Un contatto laterale non è di per sé un errore: nella scena d'esempio
`001 – 004` è laterale ma porta il 20% del peso di 004 per attrito. Il criterio
giusto è la **forza verticale**, non la normale né l'altezza del centro.

### Problema 2 — il pavimento sparisce

Nell'esempio tutti e cinque gli oggetti toccano il pavimento, anche 000 e 004.
Il grafo tiene solo i contatti fra oggetti, quindi non dice che 004 poggia per
metà (1,81 N su 3,66 N) sul pavimento: un arco vale "sostiene tutto" come
"sostiene un po'".

### Problema 3 — archi binari

`relation_score` è sempre 1,0. La quota di peso portata (27%, 30%, 20%
nell'esempio) sarebbe l'informazione più utile per stimare quanto disturbo
provoca una rimozione.

### Limiti noti, non errori

- È un grafo di **contatti a riposo**, non la verità causale ("se tolgo X, Y
  si muove?"). Quella si ottiene solo simulando la rimozione.
- Il numero di oggetti cambia a ogni passo (quelli rimossi spariscono):
  va bene per un pianificatore o una GNN, per una rete a vettore fisso serve
  un encoder a slot.
- Incertezza, zone e `bounds` sono banali o vuoti con l'oracolo: pronti per
  `oracle_degraded` e per i sensori reali.
- L'`Observation` non contiene la storia (rimozioni fallite, disturbi
  precedenti).
- Le velocità del `PrivilegedState` su una scena assestata sono ~1e-5: per il
  teacher contano poco.

---

## 6. Cosa vuole DECIDI — opzioni da decidere insieme

| Opzione | Input che usa | Cosa serve in più | Pro / contro |
|---|---|---|---|
| **Pianificatore sul grafo** (regole) | `Observation` così com'è: `dependency_graph` + pose | un grafo con il verso giusto | trasparente, baseline per la tesi; debole se il grafo sbaglia |
| **PPO a slot fissi** | vettore da `ObservationEncoder` (22 valori per slot + adiacenza N×N) | spostare l'encoder da `observe/` a `decide/`; eventualmente la forza negli archi | già integrato con Gym; perde la struttura a grafo |
| **GNN** | nodi = oggetti, archi = relazioni, direttamente | un encoder a grafo in `decide/` | gestisce N variabile e il grafo nativamente; più lavoro |
| **Teacher** (una delle tre, con privilegi) | anche `PrivilegedState` | nulla | tetto di prestazione per lo student |

Domande aperte:

1. Correggere prima l'oracolo (verso dalla forza verticale, peso portato come
   `relation_score`, contatto col pavimento)? È una modifica di OSSERVA-oracolo,
   non di DECIDI.
2. Quale decisore per primo: il pianificatore sul grafo come baseline?
3. Cosa misura il successo: numero di rimozioni, disturbo totale, crolli?
