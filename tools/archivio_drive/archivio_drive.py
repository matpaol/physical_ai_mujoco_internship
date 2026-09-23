"""Archivio su Google Drive dei risultati pesanti del progetto.

Git tiene codice, configurazioni e pesi consegnati. Tutto il resto che serve
alla tesi (dataset generati, cartelle di training, video, benchmark, dati
reali) va nell'archivio Drive, catalogato per modulo e tracciato da un
registro append-only. Guida completa: tools/archivio_drive/README.md.

Esempi, dalla radice del repository (su qualsiasi PC):

    python -m tools.archivio_drive carica outputs/detector_training/sim_dr_v1_rtx3050 \
        --categoria visore/training --fase 1B --nota "Perche' lo archivio e cosa contiene"
    python -m tools.archivio_drive stato
    python -m tools.archivio_drive scarica 2026-09-22_sim_dr_v1_rtx3050
    python -m tools.archivio_drive annota 2026-09-22_sim_dr_v1_rtx3050 --nota "Correzione..."
    python -m tools.archivio_drive categorie

Drive viene trovato da solo (G:\\Il mio Drive su Windows, ~/Library/CloudStorage
su Mac, /content/drive/MyDrive su Colab). Altrimenti: --drive PERCORSO oppure
la variabile d'ambiente ARCHIVIO_DRIVE.

Solo libreria standard: deve girare su ogni PC anche fuori dall'ambiente del
progetto.
"""

from __future__ import annotations

import argparse
import fnmatch
import getpass
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath

VERSIONE_STRUMENTO = "2"
ROOT = Path(__file__).resolve().parents[2]
CONFIG_PREDEFINITA = ROOT / "configs" / "archivio_drive.json"
REGISTRO = "registro.jsonl"
SCHEDA = "scheda.json"
README = "README.md"
NOMI_DRIVE = ("Il mio Drive", "My Drive")
OPERAZIONI_RUN = ("aggiunta", "nuova_versione")


class ErroreArchivio(Exception):
    """Errore da mostrare all'utente senza traceback."""


# --------------------------------------------------------------------------- #
# Configurazione e percorsi
# --------------------------------------------------------------------------- #


def adesso() -> datetime:
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("Europe/Rome")).replace(microsecond=0)
    except Exception:  # Windows senza tzdata: l'ora locale del PC e' quella di Roma
        return datetime.now().astimezone().replace(microsecond=0)


def carica_config(percorso: Path | None = None) -> dict:
    percorso = Path(percorso or CONFIG_PREDEFINITA)
    if not percorso.is_file():
        raise ErroreArchivio(f"Configurazione non trovata: {percorso}")
    config = json.loads(percorso.read_text(encoding="utf-8"))
    moduli = config.get("moduli", {})
    for nome, categoria in config.get("categorie", {}).items():
        cartella = categoria.get("cartella", "")
        if not cartella or PurePosixPath(cartella).parts[0] not in moduli:
            raise ErroreArchivio(
                f"Categoria '{nome}': la cartella '{cartella}' non sta sotto un modulo dichiarato"
            )
    return config


def candidati_drive() -> list[Path]:
    candidati: list[Path] = []
    sistema = platform.system()
    if sistema == "Windows":
        for lettera in "GHIJKLMNOPQRSTUVWXYZDEF":
            for nome in NOMI_DRIVE:
                candidati.append(Path(f"{lettera}:/") / nome)
    elif sistema == "Darwin":
        cloud = Path.home() / "Library" / "CloudStorage"
        if cloud.is_dir():
            for account in sorted(cloud.glob("GoogleDrive-*")):
                for nome in NOMI_DRIVE:
                    candidati.append(account / nome)
    candidati.append(Path("/content/drive/MyDrive"))  # Google Colab
    return candidati


def trova_drive(esplicito: str | None = None) -> Path:
    scelto = esplicito or os.environ.get("ARCHIVIO_DRIVE")
    if scelto:
        percorso = Path(scelto).expanduser()
        if not percorso.is_dir():
            raise ErroreArchivio(f"La cartella di Drive indicata non esiste: {percorso}")
        return percorso
    for candidato in candidati_drive():
        try:
            if candidato.is_dir():
                return candidato
        except OSError:
            continue
    raise ErroreArchivio(
        "Google Drive non trovato. Avvia l'app Google Drive per desktop, oppure indica "
        "la cartella con --drive PERCORSO o con la variabile ARCHIVIO_DRIVE."
    )


def fasi_note() -> dict[str, str]:
    cartella = ROOT / "configs" / "experiments"
    if not cartella.is_dir():
        return {}
    return {p.stem.lower(): p.stem.upper() for p in cartella.glob("*.json")}


def normalizza_fase(fase: str | None) -> str | None:
    if not fase:
        return None
    note = fasi_note()
    if not note:
        return fase.upper()
    if fase.lower() not in note:
        raise ErroreArchivio(
            f"Fase '{fase}' sconosciuta. Fasi in configs/experiments/: {', '.join(sorted(note.values()))}"
        )
    return note[fase.lower()]


def percorso_da_rel(base: Path, rel: str) -> Path:
    return base.joinpath(*PurePosixPath(rel).parts)


def relativo_al_repo(percorso: Path) -> str | None:
    try:
        return percorso.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# File, impronte, git
# --------------------------------------------------------------------------- #


def escluso(rel: PurePosixPath, modelli: list[str]) -> bool:
    return any(fnmatch.fnmatch(parte, modello) for parte in rel.parts for modello in modelli)


def raccogli_file(sorgenti: list[Path], escludi: list[str]) -> list[tuple[str, Path]]:
    """Coppie (percorso relativo nell'archivio, file locale).

    Una sola cartella: il suo contenuto va alla radice del run. Altrimenti ogni
    sorgente entra con il proprio nome.
    """
    coppie: list[tuple[str, Path]] = []
    una_cartella = len(sorgenti) == 1 and sorgenti[0].is_dir()
    for sorgente in sorgenti:
        if sorgente.is_file():
            coppie.append((sorgente.name, sorgente))
            continue
        prefisso = PurePosixPath() if una_cartella else PurePosixPath(sorgente.name)
        for file in sorted(p for p in sorgente.rglob("*") if p.is_file()):
            rel = PurePosixPath(file.relative_to(sorgente).as_posix())
            if escluso(rel, escludi):
                continue
            coppie.append(((prefisso / rel).as_posix(), file))
    visti: set[str] = set()
    for rel, _ in coppie:
        if rel in visti:
            raise ErroreArchivio(f"Due sorgenti producono lo stesso file nell'archivio: {rel}")
        visti.add(rel)
    return coppie


def sha256_file(percorso: Path) -> str:
    h = hashlib.sha256()
    with percorso.open("rb") as f:
        for blocco in iter(lambda: f.read(1 << 20), b""):
            h.update(blocco)
    return h.hexdigest()


def impronta(file: list[dict]) -> str:
    righe = "".join(f"{v['percorso']}\t{v['sha256']}\n" for v in sorted(file, key=lambda v: v["percorso"]))
    return hashlib.sha256(righe.encode("utf-8")).hexdigest()


def info_git() -> dict:
    ambiente = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}  # sola lettura: non tocca l'indice

    def git(*argomenti: str) -> str | None:
        try:
            esito = subprocess.run(
                ["git", *argomenti], cwd=ROOT, env=ambiente, capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return esito.stdout.strip() if esito.returncode == 0 else None

    stato = git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "modifiche_non_committate": None if stato is None else bool(stato),
    }


def dimensione_leggibile(byte: int) -> str:
    valore = float(byte)
    for unita in ("B", "KB", "MB", "GB"):
        if valore < 1024 or unita == "GB":
            return f"{valore:.0f} {unita}" if unita == "B" else f"{valore:.1f} {unita}"
        valore /= 1024
    return f"{byte} B"


# --------------------------------------------------------------------------- #
# Registro
# --------------------------------------------------------------------------- #


def leggi_registro(radice: Path) -> list[dict]:
    percorso = radice / REGISTRO
    if not percorso.is_file():
        return []
    voci = []
    for numero, riga in enumerate(percorso.read_text(encoding="utf-8").splitlines(), 1):
        if riga.strip():
            try:
                voci.append(json.loads(riga))
            except json.JSONDecodeError as errore:
                raise ErroreArchivio(f"{percorso}, riga {numero} illeggibile: {errore}") from errore
    return voci


def aggiungi_al_registro(radice: Path, voce: dict) -> None:
    with (radice / REGISTRO).open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(voce, ensure_ascii=False) + "\n")


def runs(registro: list[dict]) -> list[dict]:
    return [v for v in registro if v.get("operazione") in OPERAZIONI_RUN]


def trova_run(registro: list[dict], richiesta: str) -> dict:
    richiesta = richiesta.replace("\\", "/").strip("/")
    esatti = [v for v in runs(registro) if v["cartella"] == richiesta]
    if esatti:
        return esatti[-1]
    simili = [v for v in runs(registro) if v["cartella"].endswith("/" + richiesta)]
    if len(simili) == 1:
        return simili[0]
    if not simili:
        raise ErroreArchivio(f"Nessun run '{richiesta}' nel registro. Elenco: python -m tools.archivio_drive stato")
    raise ErroreArchivio(
        f"'{richiesta}' e' ambiguo: " + ", ".join(v["cartella"] for v in simili) + ". Usa il percorso completo."
    )


def chi_e_dove(argomenti) -> dict:
    return {
        "pc": argomenti.pc or os.environ.get("ARCHIVIO_PC") or platform.node(),
        "autore": argomenti.autore or os.environ.get("ARCHIVIO_AUTORE") or getpass.getuser(),
    }


# --------------------------------------------------------------------------- #
# Comandi
# --------------------------------------------------------------------------- #


def nome_run_libero(cartella_categoria: Path, base: str) -> str:
    if not (cartella_categoria / base).exists():
        return base
    n = 2
    while (cartella_categoria / f"{base}_v{n}").exists():
        n += 1
    return f"{base}_v{n}"


def destinazione_predefinita(sorgenti: list[Path]) -> str | None:
    if len(sorgenti) == 1 and sorgenti[0].is_dir():
        return relativo_al_repo(sorgenti[0])
    genitori = {s.resolve().parent for s in sorgenti}
    return relativo_al_repo(genitori.pop()) if len(genitori) == 1 else None


def comando_carica(argomenti, config: dict, radice: Path) -> int:
    categorie = config["categorie"]
    if argomenti.categoria not in categorie:
        raise ErroreArchivio(
            f"Categoria '{argomenti.categoria}' sconosciuta. Elenco: python -m tools.archivio_drive categorie"
        )
    nota = (argomenti.nota or "").strip()
    if len(nota) < 10:
        raise ErroreArchivio("--nota e' obbligatoria: scrivi cosa contiene e perche' lo archivi (almeno 10 caratteri).")
    sorgenti = [Path(s).expanduser() for s in argomenti.sorgenti]
    mancanti = [str(s) for s in sorgenti if not s.exists()]
    if mancanti:
        raise ErroreArchivio("Sorgenti inesistenti: " + ", ".join(mancanti))
    fase = normalizza_fase(argomenti.fase)
    categoria = categorie[argomenti.categoria]
    comprimi = categoria.get("comprimi", False) if argomenti.comprimi is None else argomenti.comprimi

    coppie = raccogli_file(sorgenti, config.get("escludi", []))
    if not coppie:
        raise ErroreArchivio("Nessun file da archiviare nelle sorgenti indicate.")
    print(f"Calcolo le impronte di {len(coppie)} file...", flush=True)
    file = []
    for indice, (rel, locale) in enumerate(coppie, 1):
        file.append({"percorso": rel, "byte": locale.stat().st_size, "sha256": sha256_file(locale)})
        if indice % 2000 == 0:
            print(f"  {indice}/{len(coppie)}", flush=True)
    contenuto = impronta(file)
    registro = leggi_registro(radice)
    stessa_categoria = [v for v in runs(registro) if v["categoria"] == argomenti.categoria]
    gia = [v for v in stessa_categoria if v["contenuto_sha256"] == contenuto]
    if gia and not argomenti.forza:
        print(f"Gia' archiviato, identico, in {gia[-1]['cartella']}: non carico nulla (usa --forza per duplicarlo).")
        return 0

    nome = argomenti.nome or (sorgenti[0].name if sorgenti[0].is_dir() else sorgenti[0].stem)
    precedenti = [v for v in stessa_categoria if v["nome"] == nome]
    momento = adesso()
    cartella_categoria = percorso_da_rel(radice, categoria["cartella"])
    nome_run = nome_run_libero(cartella_categoria, f"{momento.date().isoformat()}_{nome}")
    cartella_rel = f"{categoria['cartella']}/{nome_run}"
    byte_totali = sum(v["byte"] for v in file)

    voce = {
        "operazione": "nuova_versione" if precedenti else "aggiunta",
        "data": momento.isoformat(),
        **chi_e_dove(argomenti),
        "categoria": argomenti.categoria,
        "cartella": cartella_rel,
        "nome": nome,
        "fase": fase,
        "nota": nota,
        "sorgenti": [relativo_al_repo(s) or str(s.resolve()) for s in sorgenti],
        "destinazione_predefinita": destinazione_predefinita(sorgenti),
        "compresso": bool(comprimi),
        "n_file": len(file),
        "byte": byte_totali,
        "contenuto_sha256": contenuto,
        "versione_precedente": precedenti[-1]["cartella"] if precedenti else None,
        "git": info_git(),
        "strumento": f"tools/archivio_drive v{VERSIONE_STRUMENTO}",
    }

    print(
        f"{'[prova] ' if argomenti.prova else ''}{voce['operazione']}: {len(file)} file, "
        f"{dimensione_leggibile(byte_totali)} -> {radice / cartella_rel}"
        + (" (zip)" if comprimi else "")
    )
    if argomenti.prova:
        return 0

    destinazione = percorso_da_rel(radice, cartella_rel)
    destinazione.mkdir(parents=True, exist_ok=False)
    try:
        if comprimi:
            archivio = f"{nome}.zip"
            _scrivi_zip(destinazione / archivio, coppie)
            voce["archivio_zip"] = archivio
        else:
            for indice, ((rel, locale), attesa) in enumerate(zip(coppie, file), 1):
                copia = percorso_da_rel(destinazione, rel)
                copia.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(locale, copia)
                if sha256_file(copia) != attesa["sha256"]:
                    raise ErroreArchivio(f"Copia corrotta: {copia}. Run non registrato.")
                if indice % 500 == 0:
                    print(f"  copiati {indice}/{len(coppie)} file", flush=True)
        scheda = {**voce, "file": file}
        (destinazione / SCHEDA).write_text(json.dumps(scheda, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        # Interruzione (Ctrl+C) o errore: il run a meta' non deve restare nell'archivio.
        shutil.rmtree(destinazione, ignore_errors=True)
        print(f"Interrotto: rimossa la cartella incompleta {cartella_rel}. Nulla e' stato registrato.", file=sys.stderr)
        raise
    aggiungi_al_registro(radice, voce)
    scrivi_readme(radice, config)
    print(f"Archiviato e registrato: {cartella_rel}")
    return 0


def _scrivi_zip(percorso: Path, coppie: list[tuple[str, Path]]) -> None:
    parziale = percorso.with_name(percorso.name + ".parziale")
    with zipfile.ZipFile(parziale, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for indice, (rel, locale) in enumerate(coppie, 1):
            z.write(locale, rel)
            if indice % 500 == 0:
                print(f"  compressi {indice}/{len(coppie)} file", flush=True)
    with zipfile.ZipFile(parziale) as z:
        rotto = z.testzip()
        if rotto is not None or len(z.namelist()) != len(coppie):
            raise ErroreArchivio(f"Zip corrotto ({rotto}): {parziale}. Run non registrato.")
    parziale.rename(percorso)


def comando_scarica(argomenti, config: dict, radice: Path) -> int:
    voce = trova_run(leggi_registro(radice), argomenti.run)
    cartella = percorso_da_rel(radice, voce["cartella"])
    scheda_path = cartella / SCHEDA
    if not scheda_path.is_file():
        raise ErroreArchivio(f"Scheda mancante: {scheda_path}. Drive ha finito di sincronizzare?")
    scheda = json.loads(scheda_path.read_text(encoding="utf-8"))
    if argomenti.dest:
        destinazione = Path(argomenti.dest).expanduser()
    elif scheda.get("destinazione_predefinita"):
        destinazione = percorso_da_rel(ROOT, scheda["destinazione_predefinita"])
    else:
        raise ErroreArchivio("Le sorgenti erano fuori dal repository: indica dove scaricare con --dest.")

    zip_file = zipfile.ZipFile(cartella / scheda["archivio_zip"]) if scheda.get("archivio_zip") else None
    copiati = saltati = 0
    conflitti: list[str] = []
    try:
        for voce_file in scheda["file"]:
            obiettivo = percorso_da_rel(destinazione, voce_file["percorso"])
            if obiettivo.exists():
                if sha256_file(obiettivo) == voce_file["sha256"]:
                    saltati += 1
                    continue
                if not argomenti.sovrascrivi:
                    conflitti.append(str(obiettivo))
                    continue
            obiettivo.parent.mkdir(parents=True, exist_ok=True)
            if zip_file is not None:
                with zip_file.open(voce_file["percorso"]) as origine, obiettivo.open("wb") as uscita:
                    shutil.copyfileobj(origine, uscita)
            else:
                shutil.copy2(percorso_da_rel(cartella, voce_file["percorso"]), obiettivo)
            if sha256_file(obiettivo) != voce_file["sha256"]:
                raise ErroreArchivio(f"File scaricato diverso dall'originale: {obiettivo}")
            copiati += 1
    finally:
        if zip_file is not None:
            zip_file.close()

    print(f"{voce['cartella']} -> {destinazione}: {copiati} copiati, {saltati} gia' presenti e identici.")
    if conflitti:
        print(f"{len(conflitti)} file locali DIVERSI non toccati (usa --sovrascrivi per rimpiazzarli):")
        for percorso in conflitti[:20]:
            print(f"  {percorso}")
        return 1
    return 0


def comando_annota(argomenti, config: dict, radice: Path) -> int:
    nota = (argomenti.nota or "").strip()
    if len(nota) < 10:
        raise ErroreArchivio("--nota e' obbligatoria (almeno 10 caratteri).")
    voce_run = trova_run(leggi_registro(radice), argomenti.run)
    aggiungi_al_registro(
        radice,
        {
            "operazione": "annotazione",
            "data": adesso().isoformat(),
            **chi_e_dove(argomenti),
            "cartella": voce_run["cartella"],
            "nota": nota,
            "strumento": f"tools/archivio_drive v{VERSIONE_STRUMENTO}",
        },
    )
    scrivi_readme(radice, config)
    print(f"Nota aggiunta a {voce_run['cartella']}")
    return 0


def verifica_run(radice: Path, voce: dict) -> list[str]:
    cartella = percorso_da_rel(radice, voce["cartella"])
    scheda_path = cartella / SCHEDA
    if not scheda_path.is_file():
        return [f"{voce['cartella']}: scheda.json mancante"]
    scheda = json.loads(scheda_path.read_text(encoding="utf-8"))
    problemi = []
    if scheda.get("archivio_zip"):
        zip_path = cartella / scheda["archivio_zip"]
        if not zip_path.is_file():
            return [f"{voce['cartella']}: {scheda['archivio_zip']} mancante"]
        with zipfile.ZipFile(zip_path) as z:
            if z.testzip() is not None or len(z.namelist()) != len(scheda["file"]):
                problemi.append(f"{voce['cartella']}: zip corrotto o incompleto")
        return problemi
    for voce_file in scheda["file"]:
        percorso = percorso_da_rel(cartella, voce_file["percorso"])
        if not percorso.is_file():
            problemi.append(f"{voce['cartella']}: manca {voce_file['percorso']}")
        elif sha256_file(percorso) != voce_file["sha256"]:
            problemi.append(f"{voce['cartella']}: {voce_file['percorso']} modificato dopo l'archiviazione")
    return problemi


def comando_stato(argomenti, config: dict, radice: Path) -> int:
    registro = leggi_registro(radice)
    elenco = runs(registro)
    print(f"Archivio: {radice}\n{len(elenco)} run, {len(registro)} operazioni nel registro.\n")
    for voce in elenco:
        fase = voce.get("fase") or "-"
        print(f"  {voce['cartella']:<60} {fase:<4} {dimensione_leggibile(voce['byte']):>9}  {voce['nota'][:60]}")
    if not argomenti.verifica:
        return 0
    print("\nVerifica dell'integrita'...")
    problemi = [p for voce in elenco for p in verifica_run(radice, voce)]
    for problema in problemi:
        print(f"  PROBLEMA: {problema}")
    print("Tutto integro." if not problemi else f"{len(problemi)} problemi.")
    return 1 if problemi else 0


def comando_categorie(argomenti, config: dict, radice: Path | None) -> int:
    for nome, categoria in config["categorie"].items():
        zip_ = " [zip]" if categoria.get("comprimi") else ""
        print(f"  {nome:<26} -> {categoria['cartella']}{zip_}\n      {categoria['descrizione']}")
    return 0


def comando_prepara(argomenti, config: dict, radice: Path) -> int:
    for modulo in config["moduli"]:
        (radice / modulo).mkdir(parents=True, exist_ok=True)
    for categoria in config["categorie"].values():
        percorso_da_rel(radice, categoria["cartella"]).mkdir(parents=True, exist_ok=True)
    scrivi_readme(radice, config)
    print(f"Struttura pronta in {radice}")
    return 0


def comando_readme(argomenti, config: dict, radice: Path) -> int:
    scrivi_readme(radice, config)
    print(f"README rigenerato: {radice / README}")
    return 0


# --------------------------------------------------------------------------- #
# README generato
# --------------------------------------------------------------------------- #


def _cella(testo: str | None) -> str:
    return (testo or "-").replace("|", "\\|").replace("\n", " ").strip()


def genera_readme(config: dict, registro: list[dict], generato: datetime) -> str:
    elenco = runs(registro)
    annotazioni: dict[str, list[dict]] = {}
    for voce in registro:
        if voce.get("operazione") == "annotazione":
            annotazioni.setdefault(voce["cartella"], []).append(voce)

    def nota_completa(voce: dict) -> str:
        testo = voce["nota"]
        for extra in annotazioni.get(voce["cartella"], []):
            testo += f" — *Nota del {extra['data'][:10]}:* {extra['nota']}"
        return _cella(testo)

    righe = [
        "# Archivio della tesi — Physical AI con MuJoCo",
        "",
        f"> **File generato** da `tools/archivio_drive` il {generato.strftime('%Y-%m-%d %H:%M:%S')}. "
        "Non modificarlo a mano: viene riscritto a ogni operazione. Per correggere o "
        "completare una nota usa il comando `annota`.",
        "",
        config.get("descrizione", ""),
        "",
        "## Regole",
        "",
        "- Si carica **solo** con `tools/archivio_drive`: non trascinare, rinominare o cancellare file a mano, "
        "altrimenti registro e cartelle smettono di corrispondere.",
        "- Ogni caricamento e' un **run**: una cartella `AAAA-MM-GG_nome` con dentro `scheda.json` "
        "(commit git, PC, fase, hash di ogni file, nota).",
        "- `registro.jsonl` e' **append-only**: una riga per operazione, nessuna riga si modifica. "
        "Un run nuovo non sovrascrive mai il precedente: diventa `nuova_versione`.",
        "- La **nota** e' obbligatoria e dice *perche'* il run esiste e cosa contiene, con i numeri.",
        "- La fase non e' una cartella: e' un campo della scheda. Vedi la vista per fase qui sotto.",
        "",
        "## Comandi",
        "",
        "```bash",
        "python -m tools.archivio_drive categorie            # dove va cosa",
        "python -m tools.archivio_drive carica SORGENTE --categoria visore/training --fase 1B --nota \"...\"",
        "python -m tools.archivio_drive stato [--verifica]   # elenco run, controllo integrita'",
        "python -m tools.archivio_drive scarica NOME_RUN      # riporta un run nel repository locale",
        "python -m tools.archivio_drive annota NOME_RUN --nota \"...\"",
        "```",
        "",
        "## Struttura",
        "",
        "| Cartella | Contenuto | Run |",
        "|---|---|---|",
    ]
    for modulo, descrizione in config["moduli"].items():
        righe.append(f"| **`{modulo}/`** | {_cella(descrizione)} | |")
        for nome, categoria in config["categorie"].items():
            if PurePosixPath(categoria["cartella"]).parts[0] != modulo:
                continue
            conteggio = sum(1 for v in elenco if v["categoria"] == nome)
            zip_ = " (zip)" if categoria.get("comprimi") else ""
            righe.append(
                f"| &nbsp;&nbsp;`{categoria['cartella']}/` | {_cella(categoria['descrizione'])}{zip_} "
                f"| {conteggio or ''} |"
            )

    righe += ["", "## Catalogo", ""]
    if not elenco:
        righe.append("*Ancora nessun run archiviato.*")
    for modulo in config["moduli"]:
        del_modulo = [v for v in elenco if v["cartella"].split("/")[0] == modulo]
        if not del_modulo:
            continue
        righe += [f"### `{modulo}/`", "", "| Run | Data | Fase | Dimensione | Commit | PC | Nota |", "|---|---|---|---|---|---|---|"]
        for voce in sorted(del_modulo, key=lambda v: v["cartella"]):
            commit = (voce.get("git") or {}).get("commit") or ""
            sporco = " *" if (voce.get("git") or {}).get("modifiche_non_committate") else ""
            righe.append(
                f"| `{voce['cartella']}` | {voce['data'][:10]} | {voce.get('fase') or '-'} "
                f"| {dimensione_leggibile(voce['byte'])}{' zip' if voce.get('compresso') else ''} "
                f"| `{commit[:7] or '-'}`{sporco} | {_cella(voce.get('pc'))} | {nota_completa(voce)} |"
            )
        righe.append("")
    if elenco:
        righe.append("`*` dopo il commit: al momento del caricamento c'erano modifiche non committate.")

    righe += ["", "## Vista per fase", ""]
    fasi = sorted({v.get("fase") or "senza fase" for v in elenco})
    if not fasi:
        righe.append("*Nessun run.*")
    for fase in fasi:
        righe.append(f"- **{fase}**: " + ", ".join(
            f"`{v['cartella']}`" for v in elenco if (v.get("fase") or "senza fase") == fase
        ))

    righe += ["", "## Registro delle operazioni", "", "Dal piu' recente. Fonte: `registro.jsonl`.", ""]
    for voce in reversed(registro):
        righe.append(
            f"- `{voce['data']}` — **{voce['operazione']}** `{voce['cartella']}` — "
            f"{_cella(voce.get('autore'))} su {_cella(voce.get('pc'))}: {_cella(voce['nota'])}"
        )
    if not registro:
        righe.append("*Registro vuoto.*")
    return "\n".join(righe) + "\n"


def scrivi_readme(radice: Path, config: dict) -> None:
    testo = genera_readme(config, leggi_registro(radice), adesso())
    (radice / README).write_text(testo, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Interfaccia a riga di comando
# --------------------------------------------------------------------------- #


def costruisci_parser() -> argparse.ArgumentParser:
    comuni = argparse.ArgumentParser(add_help=False)
    comuni.add_argument("--drive", help="Cartella radice di Google Drive (altrimenti trovata da sola)")
    comuni.add_argument("--config", help="Configurazione dell'archivio (default configs/archivio_drive.json)")
    comuni.add_argument("--autore", help="Chi esegue l'operazione (default: utente del PC o ARCHIVIO_AUTORE)")
    comuni.add_argument("--pc", help="Nome del PC (default: nome host o ARCHIVIO_PC)")

    parser = argparse.ArgumentParser(description="Archivio Drive dei risultati pesanti del progetto")
    sotto = parser.add_subparsers(dest="comando", required=True)

    carica = sotto.add_parser("carica", parents=[comuni], help="Archivia file o cartelle come un nuovo run")
    carica.add_argument("sorgenti", nargs="+", help="File o cartelle da archiviare")
    carica.add_argument("--categoria", required=True, help="Es. visore/training (elenco: comando categorie)")
    carica.add_argument("--nota", required=True, help="Cosa contiene e perche' lo archivi")
    carica.add_argument("--fase", help="Profilo di configs/experiments/ (es. 1B)")
    carica.add_argument("--nome", help="Nome del run (default: nome della prima sorgente)")
    zip_ = carica.add_mutually_exclusive_group()
    zip_.add_argument("--zip", dest="comprimi", action="store_const", const=True, help="Forza la compressione")
    zip_.add_argument("--no-zip", dest="comprimi", action="store_const", const=False, help="Non comprimere")
    carica.add_argument("--forza", action="store_true", help="Archivia anche se un run identico esiste gia'")
    carica.add_argument("--prova", action="store_true", help="Mostra cosa farebbe senza scrivere nulla")

    scarica = sotto.add_parser("scarica", parents=[comuni], help="Riporta un run nel repository locale")
    scarica.add_argument("run", help="Nome o percorso del run (es. 2026-09-22_sim_dr_v1_rtx3050)")
    scarica.add_argument("--dest", help="Dove scaricare (default: dove stava in origine nel repository)")
    scarica.add_argument("--sovrascrivi", action="store_true", help="Rimpiazza i file locali diversi")

    stato = sotto.add_parser("stato", parents=[comuni], help="Elenca i run archiviati")
    stato.add_argument("--verifica", action="store_true", help="Controlla che file e hash corrispondano")

    annota = sotto.add_parser("annota", parents=[comuni], help="Aggiunge una nota a un run esistente")
    annota.add_argument("run")
    annota.add_argument("--nota", required=True)

    sotto.add_parser("categorie", parents=[comuni], help="Elenca le categorie dell'archivio")
    sotto.add_parser("prepara", parents=[comuni], help="Crea tutte le cartelle e il README")
    sotto.add_parser("readme", parents=[comuni], help="Rigenera il README dal registro")
    return parser


COMANDI = {
    "carica": comando_carica,
    "scarica": comando_scarica,
    "stato": comando_stato,
    "annota": comando_annota,
    "prepara": comando_prepara,
    "readme": comando_readme,
}


def main(argv: list[str] | None = None) -> int:
    for flusso in (sys.stdout, sys.stderr):
        try:
            flusso.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    argomenti = costruisci_parser().parse_args(argv)
    try:
        config = carica_config(argomenti.config)
        if argomenti.comando == "categorie":
            return comando_categorie(argomenti, config, None)
        radice = trova_drive(argomenti.drive) / config["cartella_drive"]
        radice.mkdir(exist_ok=True)
        return COMANDI[argomenti.comando](argomenti, config, radice)
    except ErroreArchivio as errore:
        print(f"Errore: {errore}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
