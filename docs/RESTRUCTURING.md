# Prima ristrutturazione

La baseline e il commit `30e5667` sul branch `refactor/progressive-deidealization`.
La migrazione conserva il task di rimozione ideale. La fase di de-idealizzazione
successiva richiede prima la prova interattiva dell'utente.

## Cosa e stato separato

| Prima | Ora |
|---|---|
| TargetExtractionEnv legge lo stato | ExactObserver -> Observation; PrivilegedState distinto |
| Policy negli script leggono MuJoCo | Decider a oggetti, input dati e output ObjectDecision |
| Env rimuove direttamente il corpo | IdealRemovalExecutor -> ExecutionOutcome |
| Env misura disturbo e calcola premio | TargetExtractionTask -> TaskOutcome |
| Env costruisce scene, pool e cadute | SceneSession nel backend simulation |
| Env gestisce la finestra | SimulationViewer nel backend simulation |
| main.py contiene il menu completo | main.py avvia ui/menu.py |
| Script contengono applicazione e confronti | experiments ed evaluation; script compatibili |

`envs/` conserva il nome e la registrazione `TargetExtraction-v0`. Non c'e una
cartella parallela `environment/` che duplichi questa responsabilita.

## Selezione della fase

La voce 7 del menu carica un profilo in `configs/experiments/`. Il profilo
stabilisce operazioni disponibili, componenti e decisore preferito. Il menu
consente confronti espliciti con altre baseline. I componenti non esaminano il
numero della fase. La variabile di processo `PHYSICAL_AI_EXPERIMENT` trasmette
il percorso della configurazione ai programmi figli e viene letta dal builder
all'avvio. Si puo impostarla anche per un comando non interattivo.

0A offre generazione e verifica; 0B il ciclo ideale; 1A anche il training PPO.
1B/1C/2/4/5/6 sono visibili ma non attivabili. La fase 3 e assorbita.
I vecchi percorsi `configs/phase_0a` e `configs/phase_0b` rimangono per compatibilita.

## Verifiche

- 60 episodi prima/dopo: cinque configurazioni (normale, misura completa,
  arresto al crollo, forme fisse, pool), tre policy e quattro seed ciascuna.
  Incluse azioni ripetute non valide. Tracce identiche: osservazioni, indici,
  premi, disturbo, maschere, terminazione e log di rilascio.
- Scene 0A con seed 0, 42 e 67 identiche. Sorgenti di generazione e relativi
  dataset/configurazioni conservati.
- **52 test superati** nel progetto installato, con due avvisi preesistenti
  sui limiti infiniti dello spazio osservazioni. Inclusi rendering e stereo e
  test dei contratti,
  sostituibilita, soglia del task, snapshot, configurazioni e normalizzazione.
- Otto prove analitiche della fisica.
- Avvio del menu e selezione 1A, viewer con due reset a forme fisse, GIF.
- Caricamento del modello esistente a 12 oggetti, normalizzazione e azione.
- Training breve a due oggetti: 256 passi effettivi, salvataggio modello,
  normalizzazione e metadati, confronto su due scene. E un controllo operativo,
  non una validazione dell'apprendimento.

Le impronte delle tracce e le versioni della prova di equivalenza sono in
`validation/baseline_comparison.json`. Si possono produrre nuove tracce con:

```bash
python tools/capture_baseline.py . outputs/checks/traces.json
```

Confrontare versioni prima/dopo con lo stesso interprete e le stesse librerie.
Il confronto esclude tempo reale e frame; la grafica e verificata separatamente.

## Correzioni circoscritte

1. Snapshot della simulazione: conserva e ripristina anche RGBA, evitando oggetti
   ancora invisibili quando si riprende una scena gia giocata.
2. Ricerca esaustiva: usa lo snapshot pubblico dell'episodio, che comprende anche
   la memoria dei crolli. Prima quel flag poteva propagarsi fra rami differenti.
3. Caricamento PPO: errore esplicito per numero di oggetti incompatibile;
   statistiche di normalizzazione obbligatorie come prima.
4. Dipendenze: unica fonte pyproject, extra train e video; requirements e avvio
   Conda richiamano gli stessi extra.
5. Training: metadati e risultati sono salvati insieme al modello. I messaggi
   non presentano piu i seed correnti come prova di scene sicuramente mai viste.

## Limiti e prossima revisione

La ricostruzione stereo, la gestione di oggetti mancanti, l'identificazione del
target tramite sensori e i campi non osservabili (massa/attrito) vanno definiti
in 1B. Il vettore legacy e conservato per compatibilita con il teacher.
Scene e fisica non sono state rese piu difficili. La validazione scientifica di
1A, inclusa la separazione training/test e il confronto con l'ottimo, resta aperta.

I controlli fisici dedicati in evaluation importano MuJoCo intenzionalmente per
costruire esperimenti analitici indipendenti dal generatore. Il core decisionale
ed i contratti non lo importano. Docker/ROS2/robot rimangono sviluppi futuri.

L'installazione supportata e quella editable dal repository; un wheel autonomo
richiedera di distribuire anche dataset e configurazioni come risorse.

Verificato anche il training con due processi: 512 passi effettivi, salvataggio
e valutazione completati. `main.py` ha 6 righe; l’adapter Gymnasium 251.
