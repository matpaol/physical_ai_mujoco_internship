# Lavorare su più PC: git e archivio Drive

Il progetto gira su più macchine: il **Mac M3** (lavoro quotidiano), il **ROG con
RTX 3050** (training pesanti), il **PC Linux del tirocinio** ed eventualmente
**Colab**. Questo file dice dove sta ogni cosa e con quali comandi si tiene
tutto allineato. Vale per le persone e per le IA che lavorano sul repository.

---

## 1. Tre posti, tre regole

| Dove | Cosa ci va | Come si sincronizza |
|---|---|---|
| **GitHub** (git) | codice, configurazioni, documentazione, **pesi consegnati** (`outputs/detector_weights/`) | `git pull` / `git push` |
| **Archivio Drive** (`physical_ai_mujoco_tesi/`) | risultati pesanti utili alla tesi: dataset generati, cartelle di training, video, benchmark, dati reali | `python -m tools.archivio_drive` |
| **Solo locale** | ambienti (`.venv`), cache, modelli base (`yolo*.pt`), esperimenti di prova | non si sincronizza |

Domanda da farsi per ogni file prodotto:

1. **Il codice lo carica per funzionare?** (pesi del detector in uso, policy
   consegnata) → git. Solo la versione consegnata, non le intermedie: ogni
   `.pt` committato resta per sempre nella storia del repository.
2. **Serve alla tesi o a ripetere un risultato?** → archivio Drive.
3. **Si rigenera con un comando o si riscarica?** → resta locale.

Cosa esclude git (`.gitignore`): tutto `outputs/` **tranne**
`outputs/detector_weights/`, `datasets/generated/`, i modelli base `/yolo*.pt`
e `/weights/`, gli ambienti virtuali.

---

## 2. Git su più PC

### Il concetto di branch, in breve

Un branch è una linea di lavoro. `main` è la versione stabile; un branch come
`observe-implementation` contiene un lavoro in corso. Si lavora sul branch, e
quando il lavoro è pronto lo si unisce (*merge*) in `main`.

```bash
git branch --show-current     # su quale branch sono
git branch -a                 # tutti i branch, locali e su GitHub
```

### Routine su qualsiasi PC

**Prima di iniziare:**

```bash
cd <cartella del progetto>
git switch observe-implementation   # o il branch su cui stai lavorando
git pull
```

**Alla fine** (o prima di cambiare PC):

```bash
git status                    # cosa è cambiato
git diff                      # le modifiche, riga per riga
git add <file>                # oppure: git add .   (dopo aver guardato git status)
git commit -m "Short English sentence saying what changes"   # messaggi di commit in inglese
git push
```

Il cambio PC è sempre: **push sul PC che lasci → pull su quello dove arrivi.**

### Primo pull di un branch su un PC che non l'ha mai visto

```bash
git fetch
git switch observe-implementation   # crea il branch locale collegato a quello su GitHub
git pull
```

### Unire il lavoro in `main`

Quando un branch è finito e i test passano:

```bash
git switch main
git pull
git merge observe-implementation
git push
git switch observe-implementation   # per continuare a lavorare sul branch
```

### Aprire un branch nuovo

```bash
git switch main
git pull
git switch -c nome-del-lavoro
git push -u origin nome-del-lavoro  # la prima volta: lo pubblica su GitHub
```

### Problemi frequenti

| Sintomo | Cosa fare |
|---|---|
| `git push` rifiutato (*rejected*, *fetch first*) | `git pull --rebase`, poi `git push` |
| Conflitto dopo pull o merge | Apri i file segnalati (VS Code evidenzia i blocchi), scegli cosa tenere, poi `git add <file>` e `git rebase --continue` (oppure `git commit` se era un merge). Se non sei sicuro: `git status` e fermati. |
| `git status` su Windows mostra quasi tutti i file modificati senza differenze vere | Sono i fine riga (CRLF/LF). Controlla con `git diff --ignore-cr-at-eol --stat`: se è vuoto, `git config core.autocrlf true` su Windows (su Mac/Linux: `input`). |
| `fatal: Unable to create '.git/index.lock': File exists` | Un altro git è in corso o si è chiuso male. Chiudi VS Code/terminali git; se persiste cancella `.git/index.lock`. |
| Hai modificato file ma devi fare pull | `git stash`, `git pull`, `git stash pop` |
| Vuoi buttare le modifiche di un file | `git restore percorso/file` (irreversibile) |

**Mai** `git push --force` su `main`. **Mai** committare password, token o
dati riservati dell'ente.

---

## 3. Archivio Drive (in breve)

I risultati pesanti vanno nella cartella `physical_ai_mujoco_tesi/` di Google
Drive, gestita dal modulo **`tools/archivio_drive/`**. Guida completa
(struttura, comandi, regole): [tools/archivio_drive/README.md](../tools/archivio_drive/README.md).

```bash
python -m tools.archivio_drive categorie                 # dove va cosa
python -m tools.archivio_drive carica SORGENTE --categoria visore/training --fase 1B --nota "..."
python -m tools.archivio_drive stato                     # cosa c'e' nell'archivio
python -m tools.archivio_drive scarica NOME_RUN          # riportarlo su questo PC
```

---

## 4. Checklist

### Dopo un training o un esperimento pesante

1. Pesi da consegnare in `outputs/detector_weights/` → `git add`, commit, push.
2. Cartella di training, dataset, video, report → `python -m tools.archivio_drive carica`
   con nota e fase.
3. Configurazioni o ricette nuove → git, con la voce in `MODIFICHE.md`.

### Arrivando su un altro PC

1. `git switch <branch>` e `git pull`.
2. Servono i risultati di un altro PC? `python -m tools.archivio_drive stato`, poi `scarica`.
   (Aspetta che l'app di Drive abbia finito di sincronizzare.)

### Primo setup di un PC nuovo

1. Clonare: `git clone https://github.com/<utente>/physical-ai-mujoco.git`.
2. Ambiente: `python -m venv .venv`, attivarlo, `pip install -r requirements.txt`
   (per il visore anche `pip install -e .[detector]`).
3. Installare Google Drive per desktop e fare l'accesso con l'account del
   progetto; controllare con `python -m tools.archivio_drive stato`.

---

## 5. Per le IA che lavorano sul repository

- Prima di chiudere una sessione che ha prodotto risultati pesanti, proponi di
  archiviarli con `python -m tools.archivio_drive carica` e scrivi tu la `--nota`
  (problema/scopo e numeri). Firma con `--autore "<IA> (<modello>) su richiesta di <persona>"`.
- Non modificare a mano `README.md`, `registro.jsonl` o le `scheda.json`
  dell'archivio.
- Non committare nulla di `outputs/` fuori da `outputs/detector_weights/`, né
  modelli base o dataset.
- Le operazioni git che scrivono (commit, push, merge) le decide la persona:
  prepara i file e proponi i comandi.
