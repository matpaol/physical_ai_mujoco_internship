# Physical AI con MuJoCo

Banco di prova sim2real per scegliere l'ordine di rimozione di oggetti da un
mucchio, raggiungere un target e misurare il disturbo provocato agli altri.
Il progetto cresce per de-idealizzazione: prima decisioni con stato esatto e
rimozione ideale, poi percezione stereo, robustezza, UR5 simulato e sistema reale.

## Avvio

Dalla cartella del progetto, nell'ambiente Conda gia usato:

```bash
conda activate mujoco-tirocinio
python main.py
```

La voce **7** seleziona la fase/configurazione: 0A, 0B e il prototipo 1A sono
utilizzabili; le fasi successive sono visibili come "da sviluppare". Il menu
mostra le operazioni del profilo selezionato. Per allenare PPO, scegli 1A e
poi la voce 3. Il profilo iniziale e 0B.

I profili vivono in `configs/experiments/`: selezionano i componenti nel builder.
Anche i processi avviati dal menu ereditano la selezione. Le librerie scientifiche
non leggono nomi o numeri delle fasi. La fase 3 e assorbita da 1B e 2.
`main.py` si puo ancora avviare dal pulsante Run di VS Code.

Per una nuova installazione (Python >= 3.10):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test,train,video]"
python main.py
```

Anche `python -m pip install -r requirements.txt` usa le stesse dipendenze.
`train` installa Stable-Baselines3; `video` aggiunge il supporto MP4.
L'installazione supportata e editable dalla copia del repository: configurazioni
e dataset rimangono nella radice, fuori dal pacchetto Python.

## Comandi disponibili

```bash
python -m scripts.run_phase_0a --seed 42
python -m scripts.run_phase_0b --objects 6 --headless --episodes 3
python -m scripts.inspect_episode --objects 6 --policy top --seed 0 --episodes 20 --quiet
```

Per allenare usare il menu oppure:

```bash
python scripts/allena.py --oggetti 6 --passi 25000 --paralleli 2
python scripts/quanto_margine.py --oggetti 3 --scene 2 --finisci-al-target
python scripts/verifica_fisica.py
python -m scripts.record_phase_0b --objects 3 --format gif
```

I precedenti comandi `scripts/*` sono mantenuti come punti di ingresso; il codice
operativo risiede nel package. I modelli esistenti in `outputs/modelli/` e i loro
file `_normalizzazione.pkl` restano utilizzabili con lo stesso numero di oggetti.
Un numero diverso viene ora segnalato con un errore esplicito.

## Organizzazione

| Package | Responsabilita |
|---|---|
| `contracts` | Observation, PrivilegedState, decisione ed esiti |
| `scene` | Dataset, descrizione e campionamento della scena |
| `simulation` | MuJoCo, sessione della scena, assestamento, pool, viewer |
| `observe` | Lettura esatta e acquisizione stereo grezza |
| `decide` | Classi decisionali casuale, altezza, target e PPO |
| `execute` | Rimozione ideale tramite API del simulatore |
| `task` | Disturbo, reward, crollo, successo e fine episodio |
| `envs` | Adapter Gymnasium; percorso storico mantenuto |
| `infrastructure` | Builder e adapter dei callback esistenti |
| `experiments` | Rollout, training, registrazione, monitor e metadati |
| `evaluation` | Ispezione, ricerca esaustiva, confronto e verifiche fisiche |
| `ui` | Menu e interazione con l'utente |

Ogni ruolo pubblica la sua API tramite `__init__.py`. DECIDE riceve dati, produce
un `ObjectDecision` e non legge il simulatore. Il runner passa la decisione
all'ambiente, che coordina gli altri componenti.

## Stato attuale e limiti

- Fase 0A e ciclo 0B implementati; esiste un prototipo PPO per 1A.
- `state` conserva i 17 valori originali per oggetto; `stereo` restituisce due
  immagini RGB; `both` restituisce stato e immagini sincronizzati.
- Le immagini sono acquisizione grezza: non c'e ancora ricostruzione stereo,
  student percettivo o belief calibrato.
- L'osservazione esatta include massa, attrito, presenza e target. Come ottenere
  o sostituire questi campi nella percezione e una decisione della fase 1B.
- La rimozione resta ideale: il robot non e ancora presente.
- Con la configurazione attuale l'episodio termina al target, prosegue dopo un
  crollo e ha limite di 64 azioni. `is_success` richiede target rimosso senza
  crolli precedenti. Il disturbo considera la traslazione finale massima degli
  oggetti rimasti, escludendo il target; non misura il percorso o la rotazione.
- La validazione scientifica di 1A deve ancora definire split e gate. I seed
  della valutazione del trainer sono registrati, ma non dimostrano un test set
  indipendente dal training.

## Test e riproducibilita

```bash
python -m pytest -q
python scripts/verifica_fisica.py
```

I test grafici richiedono accesso al display macOS. Su Linux senza display il
backend offscreen va configurato in base alla macchina (EGL oppure OSMesa).
Un errore CoreGraphics di accesso al display non certifica un errore della stereo.

Ogni nuovo training salva anche `<modello>_run.json`: parametri, seed di
valutazione, versioni, commit, hash del codice, configurazioni, dataset e risultati.
La migrazione e documentata in [docs/RESTRUCTURING.md](docs/RESTRUCTURING.md).

## Documenti guida

- [Architettura](docs/ARCHITECTURE.md)
- [De-idealizzazione](docs/DEIDEALIZATION.md)
- [Registro delle modifiche](MODIFICHE.md)
- [Documentazione precedente, storica](docs/history/)

Gli output e le vecchie copie in `_to_delete/` rimangono locali, esclusi da Git.
