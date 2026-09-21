# Physical AI con MuJoCo

Repository del progetto di tirocinio dedicato a **MuJoCo**, simulazione fisica, reinforcement learning e studio dei DLO (*Deformable Linear Objects*).

Questo README contiene soprattutto un promemoria pratico per sincronizzare il progetto tra:

- Mac con Visual Studio Code;
- GitHub;
- PC Linux del tirocinio;
- eventualmente Google Colab.

## Regola fondamentale

Prima di iniziare a lavorare:

```bash
git pull
```

Dopo aver modificato i file:

```bash
git add .
git commit -m "Descrizione breve delle modifiche"
git push
```

Il flusso è quindi:

```text
git pull → modifica i file → git add → git commit → git push
```

## Struttura prevista

```text
physical-ai-mujoco/
├── README.md
├── environment.yml
├── requirements.txt
├── test_1_mujoco_base/
├── test_2_pallina_bicchieri/
├── test_3_dyndlo/
├── notebooks/
└── docs/
```

L'ambiente Conda, Miniforge, cache, password, token e file molto pesanti non devono essere caricati nel repository.

## 1. Prima configurazione sul Mac

Questi passaggi si eseguono una volta sola.

### Verificare Git

Aprire il terminale del Mac e digitare:

```bash
git --version
```

### Configurare nome ed email

```bash
git config --global user.name "Matteo"
git config --global user.email "LA-TUA-EMAIL-GITHUB"
```

### Scaricare il repository

Sostituire `TUO-USERNAME` con il proprio username GitHub:

```bash
cd ~/Documents
git clone https://github.com/TUO-USERNAME/physical-ai-mujoco.git
cd physical-ai-mujoco
```

### Aprire il progetto in VS Code

```bash
code .
```

Se il comando `code` non è disponibile, aprire VS Code e selezionare **File → Open Folder**, quindi scegliere la cartella `physical-ai-mujoco`.

## 2. Prima configurazione sul PC Linux

Questi passaggi si eseguono una volta sola.

### Verificare Git

```bash
git --version
```

### Configurare nome ed email

```bash
git config --global user.name "Matteo"
git config --global user.email "LA-TUA-EMAIL-GITHUB"
```

### Scaricare il repository

```bash
cd /home/matteo
git clone https://github.com/TUO-USERNAME/physical-ai-mujoco.git
cd physical-ai-mujoco
```

Se esiste già una vecchia cartella locale con gli esperimenti, non copiarla alla cieca sopra il repository: spostare soltanto i singoli file necessari dopo aver eseguito il clone.

## 3. Routine quotidiana in VS Code

Nel terminale integrato di VS Code, verificare innanzitutto di trovarsi nella cartella del progetto:

```bash
pwd
git status
```

### Prima di modificare i file

```bash
git pull
```

### Dopo le modifiche

Controllare cosa è cambiato:

```bash
git status
git diff
```

Preparare i file per il commit:

```bash
git add .
```

Creare il commit:

```bash
git commit -m "Aggiunge test di caduta della pallina"
```

Caricare il commit su GitHub:

```bash
git push
```

### Controllare la cronologia

```bash
git log --oneline --max-count=10
```

## 4. Passare dal Mac al PC Linux

Sul Mac, dopo aver terminato le modifiche:

```bash
git add .
git commit -m "Aggiorna esperimento MuJoCo"
git push
```

Sul PC Linux, prima di eseguire il codice:

```bash
cd /home/matteo/physical-ai-mujoco
git pull
conda activate mujoco-tirocinio
```

Esempio di esecuzione:

```bash
python test_1_mujoco_base/test_caduta.py
```

## 5. Passare dal PC Linux al Mac

Sul PC Linux, dopo aver modificato o creato file:

```bash
git status
git add .
git commit -m "Aggiorna test eseguito su Linux"
git push
```

Sul Mac:

```bash
cd ~/Documents/physical-ai-mujoco
git pull
```

Ora VS Code mostrerà i file aggiornati.

## 6. Comandi Git più utili

| Comando | Significato |
|---|---|
| `git status` | Mostra i file modificati e lo stato del repository |
| `git pull` | Scarica e integra gli ultimi aggiornamenti da GitHub |
| `git diff` | Mostra le modifiche non ancora preparate |
| `git add nome_file.py` | Prepara un solo file per il commit |
| `git add .` | Prepara tutte le modifiche nella cartella corrente |
| `git commit -m "messaggio"` | Registra localmente una versione delle modifiche |
| `git push` | Carica su GitHub i commit locali |
| `git log --oneline` | Mostra la cronologia sintetica dei commit |
| `git branch --show-current` | Mostra il branch attuale |

## 7. Se `git push` viene rifiutato

Può accadere quando GitHub contiene modifiche che il computer locale non ha ancora scaricato.

Eseguire:

```bash
git pull --rebase
git push
```

Se Git segnala un conflitto, non forzare il caricamento. Aprire i file indicati da VS Code, scegliere quali modifiche mantenere, quindi eseguire:

```bash
git add .
git rebase --continue
git push
```

Se non si è sicuri di come risolvere il conflitto, fermarsi e controllare `git status` prima di procedere.

## 8. Annullare modifiche non ancora salvate in un commit

Per vedere prima quali modifiche andrebbero perse:

```bash
git diff
```

Per annullare le modifiche locali di un singolo file:

```bash
git restore percorso/del/file.py
```

Questo comando elimina le modifiche non registrate di quel file, quindi va usato con attenzione.

## 9. Usare il repository in Google Colab

Per un repository pubblico:

```python
!git clone https://github.com/TUO-USERNAME/physical-ai-mujoco.git
%cd physical-ai-mujoco
!pip install -q mujoco
```

Il runtime Colab è temporaneo. Prima di chiuderlo, scaricare i file importanti oppure eseguire commit e push solo dopo aver configurato un metodo di autenticazione sicuro. Non scrivere token GitHub direttamente nei notebook destinati al repository.

## 10. Buone abitudini

- Eseguire sempre `git pull` prima di iniziare.
- Fare commit piccoli, con un messaggio chiaro.
- Eseguire `git status` prima di `git add .`.
- Non lavorare contemporaneamente sullo stesso file da Mac e Linux senza sincronizzare.
- Non caricare password, token, credenziali o materiale riservato dell'ente.
- Non caricare l'ambiente Conda: usare `environment.yml` o `requirements.txt` per descrivere le dipendenze.
- Caricare codice e configurazioni; evitare video, dataset, cache e risultati molto pesanti.

## Promemoria velocissimo

All'inizio:

```bash
cd percorso/physical-ai-mujoco
git pull
```

Alla fine:

```bash
git status
git add .
git commit -m "Descrive cosa è cambiato"
git push
```

