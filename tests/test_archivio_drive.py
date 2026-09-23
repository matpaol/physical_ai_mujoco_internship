"""Archivio Drive: una cartella temporanea fa da Google Drive."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("archivio_drive", ROOT / "tools" / "archivio_drive" / "archivio_drive.py")
archivio = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(archivio)

NOTA = "Run di prova per il test dell'archivio"


@pytest.fixture
def ambiente(tmp_path):
    drive = tmp_path / "drive"
    drive.mkdir()
    sorgente = tmp_path / "run_prova"
    (sorgente / "weights").mkdir(parents=True)
    (sorgente / "results.csv").write_text("epoch,map\n1,0.5\n", encoding="utf-8")
    (sorgente / "weights" / "best.pt").write_bytes(b"\x00pesi" * 100)
    (sorgente / "__pycache__").mkdir()
    (sorgente / "__pycache__" / "x.pyc").write_bytes(b"da escludere")
    (sorgente / "._results.csv").write_bytes(b"metadati del Mac")  # AppleDouble
    return drive, sorgente, tmp_path


def esegui(drive, *argomenti):
    return archivio.main([*argomenti, "--drive", str(drive), "--autore", "test", "--pc", "pc-test"])


def radice(drive):
    return drive / archivio.carica_config()["cartella_drive"]


def test_configurazione_coerente():
    config = archivio.carica_config()
    assert "visore/training" in config["categorie"]
    assert all(c["descrizione"] for c in config["categorie"].values())


def test_carica_copia_registra_e_scrive_readme(ambiente):
    drive, sorgente, _ = ambiente
    assert esegui(drive, "carica", str(sorgente), "--categoria", "visore/training", "--nota", NOTA) == 0

    registro = archivio.leggi_registro(radice(drive))
    assert len(registro) == 1
    voce = registro[0]
    assert voce["operazione"] == "aggiunta"
    assert voce["cartella"].startswith("04_visore/training/") and voce["cartella"].endswith("_run_prova")
    assert voce["n_file"] == 2  # __pycache__ escluso
    run = radice(drive) / voce["cartella"]
    assert (run / "weights" / "best.pt").read_bytes() == b"\x00pesi" * 100
    scheda = json.loads((run / "scheda.json").read_text(encoding="utf-8"))
    assert {f["percorso"] for f in scheda["file"]} == {"results.csv", "weights/best.pt"}
    readme = (radice(drive) / "README.md").read_text(encoding="utf-8")
    assert voce["cartella"] in readme and NOTA in readme


def test_contenuto_identico_non_viene_ricaricato(ambiente):
    drive, sorgente, _ = ambiente
    esegui(drive, "carica", str(sorgente), "--categoria", "visore/training", "--nota", NOTA)
    esegui(drive, "carica", str(sorgente), "--categoria", "visore/training", "--nota", NOTA)
    assert len(archivio.leggi_registro(radice(drive))) == 1


def test_contenuto_cambiato_diventa_nuova_versione(ambiente):
    drive, sorgente, _ = ambiente
    esegui(drive, "carica", str(sorgente), "--categoria", "visore/training", "--nota", NOTA)
    (sorgente / "results.csv").write_text("epoch,map\n1,0.5\n2,0.6\n", encoding="utf-8")
    esegui(drive, "carica", str(sorgente), "--categoria", "visore/training", "--nota", NOTA + " (seconda)")

    prima, seconda = archivio.leggi_registro(radice(drive))
    assert seconda["operazione"] == "nuova_versione"
    assert seconda["versione_precedente"] == prima["cartella"]
    assert seconda["cartella"] == prima["cartella"] + "_v2"
    assert (radice(drive) / prima["cartella"] / "results.csv").read_text(encoding="utf-8") == "epoch,map\n1,0.5\n"


def test_zip_e_scarica_ripristinano_i_file(ambiente):
    drive, sorgente, tmp_path = ambiente
    assert esegui(drive, "carica", str(sorgente), "--categoria", "visore/dataset", "--nota", NOTA) == 0
    voce = archivio.leggi_registro(radice(drive))[0]
    run = radice(drive) / voce["cartella"]
    assert (run / "run_prova.zip").is_file()
    assert not list(run.glob("*.parziale"))

    destinazione = tmp_path / "scaricato"
    assert esegui(drive, "scarica", voce["cartella"].split("/")[-1], "--dest", str(destinazione)) == 0
    assert (destinazione / "weights" / "best.pt").read_bytes() == b"\x00pesi" * 100
    assert esegui(drive, "scarica", voce["cartella"], "--dest", str(destinazione)) == 0  # gia' presenti


def test_scarica_non_sovrascrive_file_locali_diversi(ambiente):
    drive, sorgente, tmp_path = ambiente
    esegui(drive, "carica", str(sorgente), "--categoria", "visore/training", "--nota", NOTA)
    voce = archivio.leggi_registro(radice(drive))[0]
    destinazione = tmp_path / "locale"
    destinazione.mkdir()
    (destinazione / "results.csv").write_text("mio lavoro", encoding="utf-8")

    assert esegui(drive, "scarica", voce["cartella"], "--dest", str(destinazione)) == 1
    assert (destinazione / "results.csv").read_text(encoding="utf-8") == "mio lavoro"
    assert esegui(drive, "scarica", voce["cartella"], "--dest", str(destinazione), "--sovrascrivi") == 0
    assert (destinazione / "results.csv").read_text(encoding="utf-8") == "epoch,map\n1,0.5\n"


def test_piu_file_entrano_con_il_loro_nome(ambiente):
    drive, sorgente, _ = ambiente
    pesi = sorgente / "weights" / "best.pt"
    esegui(drive, "carica", str(pesi), str(sorgente / "results.csv"), "--categoria", "visore/pesi", "--nota", NOTA)
    voce = archivio.leggi_registro(radice(drive))[0]
    assert voce["nome"] == "best"
    assert (radice(drive) / voce["cartella"] / "best.pt").is_file()
    assert (radice(drive) / voce["cartella"] / "results.csv").is_file()


def test_errori_di_uso(ambiente):
    drive, sorgente, _ = ambiente
    assert esegui(drive, "carica", str(sorgente), "--categoria", "visore/training", "--nota", "corta") == 2
    assert esegui(drive, "carica", str(sorgente), "--categoria", "inesistente", "--nota", NOTA) == 2
    assert esegui(drive, "carica", str(sorgente / "manca"), "--categoria", "visore/training", "--nota", NOTA) == 2
    assert not archivio.leggi_registro(radice(drive))


def test_annota_e_verifica(ambiente):
    drive, sorgente, _ = ambiente
    esegui(drive, "carica", str(sorgente), "--categoria", "visore/training", "--nota", NOTA)
    voce = archivio.leggi_registro(radice(drive))[0]
    assert esegui(drive, "annota", voce["cartella"], "--nota", "Precisazione aggiunta dopo") == 0
    assert "Precisazione aggiunta dopo" in (radice(drive) / "README.md").read_text(encoding="utf-8")
    assert esegui(drive, "stato", "--verifica") == 0

    (radice(drive) / voce["cartella"] / "results.csv").write_text("alterato", encoding="utf-8")
    assert esegui(drive, "stato", "--verifica") == 1


def test_prepara_crea_tutte_le_cartelle(ambiente):
    drive, _, _ = ambiente
    assert esegui(drive, "prepara") == 0
    config = archivio.carica_config()
    for categoria in config["categorie"].values():
        assert (radice(drive) / categoria["cartella"]).is_dir()


def test_prova_non_scrive_nulla(ambiente):
    drive, sorgente, _ = ambiente
    esegui(drive, "carica", str(sorgente), "--categoria", "visore/training", "--nota", NOTA, "--prova")
    assert not archivio.leggi_registro(radice(drive))
    assert not (radice(drive) / "04_visore").exists()


def test_interruzione_non_lascia_run_a_meta(ambiente, monkeypatch):
    drive, sorgente, _ = ambiente

    def copia_che_si_interrompe(*_argomenti, **_opzioni):
        raise KeyboardInterrupt

    monkeypatch.setattr(archivio.shutil, "copy2", copia_che_si_interrompe)
    with pytest.raises(KeyboardInterrupt):
        esegui(drive, "carica", str(sorgente), "--categoria", "visore/training", "--nota", NOTA)
    assert not archivio.leggi_registro(radice(drive))
    assert not any((radice(drive) / "04_visore" / "training").iterdir())
