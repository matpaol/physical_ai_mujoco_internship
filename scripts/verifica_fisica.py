"""Prove con risposta analitica nota, per decidere se la fisica e' accettabile.

Non e' il banco di prova completo descritto in `upgrade/banco di prova della
fisica.md` — quello verra' dopo, separato e randomizzato. Questo e' il minimo
indispensabile: le poche prove di cui conosciamo la risposta esatta, scritte
una volta sola invece che riscritte a mano a ogni modifica dei parametri di
contatto.

Si usa cosi':

    python3 scripts/verifica_fisica.py
    python3 scripts/verifica_fisica.py --scene 20   # anche le prove sulle pile

Ogni prova stampa il valore misurato, quello atteso e lo scarto. Le prove
statiche costruiscono modelli MJCF minimi scritti qui: devono misurare la
FISICA, non il nostro generatore di scene. Le prove sulle pile usano invece
l'environment vero, perche' li' l'oggetto della misura e' proprio la scena che
produciamo.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco  # noqa: E402

CONTATTO = json.loads(
    (PROJECT_ROOT / "configs/phase_0b/simulation.json").read_text(encoding="utf-8")
)


def _opzioni() -> str:
    """Le opzioni di `<option>` della Fase 0B, lette dalla configurazione.

    Scriverle a mano qui vorrebbe dire misurare parametri diversi da quelli
    che il progetto usa davvero — cioe' misurare un'altra cosa.
    """
    contatto = CONTATTO.get("contact", {})
    attributi = {
        "timestep": CONTATTO["timestep"],
        "integrator": CONTATTO["integrator"],
        "gravity": " ".join(str(value) for value in CONTATTO["gravity"]),
        "impratio": contatto.get("impratio", 1.0),
        "cone": contatto.get("cone", "pyramidal"),
        "noslip_iterations": int(contatto.get("noslip_iterations", 0)),
    }
    return " ".join(f'{key}="{value}"' for key, value in attributi.items())


def _default() -> str:
    """Il blocco `<default>` con la rigidezza di contatto della Fase 0B."""
    timeconst = CONTATTO.get("contact", {}).get("timeconst", 0.02)
    return f'<default><geom solref="{timeconst} 1"/></default>'


def _modello_piano_inclinato(
    inclinazione: float,
    semi: tuple[float, float, float],
    attrito: tuple[float, float, float],
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Un blocco appoggiato su un piano inclinato di `inclinazione` radianti.

    Il blocco e' generato gia' in appoggio e con l'asse verticale ruotato come
    il piano: cosi' la prova misura l'inizio del moto, non l'atterraggio.
    """
    x, y, z = semi
    mezzo = inclinazione / 2.0
    # Rotazione attorno a y: quaternione (cos, 0, sin, 0).
    quaternione = f"{math.cos(mezzo)} 0 {math.sin(mezzo)} 0"
    altezza = z + 1e-4
    centro = (-altezza * math.sin(inclinazione), 0.0, altezza * math.cos(inclinazione))
    xml = f"""
    <mujoco>
      <compiler angle="radian" autolimits="true"/>
      <option {_opzioni()}/>
      {_default()}
      <worldbody>
        <body name="piano" quat="{quaternione}">
          <geom type="plane" size="5 5 0.1" condim="6"
                friction="{attrito[0]} {attrito[1]} {attrito[2]}"/>
        </body>
        <body name="blocco" pos="{centro[0]} {centro[1]} {centro[2]}"
              quat="{quaternione}">
          <freejoint/>
          <geom type="box" size="{x} {y} {z}" mass="1.0" condim="6"
                friction="{attrito[0]} {attrito[1]} {attrito[2]}"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _scivola_o_si_ribalta(
    inclinazione: float,
    semi: tuple[float, float, float],
    attrito: tuple[float, float, float],
    durata: float = 1.5,
    assestamento: float = 0.5,
) -> tuple[float, float]:
    """Simula e restituisce (scivolamento lungo il piano, rotazione) in 1,5 s.

    I primi `assestamento` secondi non vengono misurati: il blocco nasce a un
    decimo di millimetro dal piano e il primo contatto e' un urto. Quell'urto
    sposta il blocco di frazioni di millimetro anche quando l'attrito lo
    terrebbe fermo, e misurando da zero verrebbe scambiato per scivolamento.
    """
    model, data = _modello_piano_inclinato(inclinazione, semi, attrito)
    # La rotazione si misura dall'ORIENTAMENTO INIZIALE, lo scivolamento dalla
    # posizione dopo l'assestamento. Un blocco che si ribalta lo fa nel primo
    # mezzo secondo: misurandone la rotazione a partire da li' risulterebbe
    # gia' caduto e quindi immobile.
    quaternione = data.qpos[3:7].copy()
    for _ in range(int(assestamento / model.opt.timestep)):
        mujoco.mj_step(model, data)

    partenza = data.qpos[:3].copy()
    for _ in range(int(durata / model.opt.timestep)):
        mujoco.mj_step(model, data)
    spostamento = float(np.linalg.norm(data.qpos[:3] - partenza))
    coseno = abs(float(np.dot(quaternione, data.qpos[3:7])))
    return spostamento, 2.0 * math.acos(min(1.0, coseno))


def _angolo_critico(prova, massimo: float = math.radians(60.0)) -> float:
    """Il piu' piccolo angolo, al decimo di grado, per cui `prova` e' vera."""
    basso, alto = 0.0, massimo
    for _ in range(40):
        mezzo = (basso + alto) / 2.0
        if prova(mezzo):
            alto = mezzo
        else:
            basso = mezzo
    return alto


def prova_scivolamento() -> None:
    """Un blocco largo scivola quando tan(theta) supera il coefficiente."""
    mu = 0.6
    semi = (0.05, 0.05, 0.01)  # largo e basso: scivola prima di ribaltarsi
    atteso = math.atan(mu)
    misurato = _angolo_critico(
        lambda angolo: _scivola_o_si_ribalta(angolo, semi, (mu, 0.005, 0.005))[0] > 0.01
    )
    _riga(
        "scivolamento su piano inclinato",
        math.degrees(misurato),
        math.degrees(atteso),
        "gradi",
        tolleranza=1.0,
    )


def prova_ribaltamento() -> None:
    """Un blocco alto si ribalta quando tan(theta) supera base/altezza."""
    semi = (0.02, 0.05, 0.08)  # stretto e alto: si ribalta prima di scivolare
    atteso = math.atan(semi[0] / semi[2])
    misurato = _angolo_critico(
        lambda angolo: _scivola_o_si_ribalta(angolo, semi, (2.0, 0.005, 0.005))[1]
        > math.radians(20.0)
    )
    _riga(
        "ribaltamento su piano inclinato",
        math.degrees(misurato),
        math.degrees(atteso),
        "gradi",
        tolleranza=2.0,
    )


def prova_caduta_libera() -> None:
    """Il tempo di caduta da quota h deve valere sqrt(2h/g)."""
    g = abs(CONTATTO["gravity"][2])
    for quota in (0.2, 0.5, 1.0):
        xml = f"""
        <mujoco>
          <compiler angle="radian" autolimits="true"/>
          <option {_opzioni()}/>
          {_default()}
          <worldbody>
            <geom type="plane" size="5 5 0.1"/>
            <body name="palla" pos="0 0 {quota + 0.01}">
              <freejoint/>
              <geom type="sphere" size="0.01" mass="1.0"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        while data.qpos[2] > 0.01 and data.time < 5.0:
            mujoco.mj_step(model, data)
        _riga(
            f"caduta libera da {quota:.1f} m",
            float(data.time),
            math.sqrt(2.0 * quota / g),
            "s",
            tolleranza=0.01,
        )


def prova_inerzia() -> None:
    """L'inerzia che calcoliamo noi deve coincidere con quella del compilatore."""
    from physical_ai_mujoco.scene.scene_description import principal_inertia

    casi = (
        ("box", {"x": 0.08, "y": 0.05, "z": 0.03}, 'type="box" size="0.04 0.025 0.015"'),
        ("cylinder", {"radius": 0.03, "height": 0.09}, 'type="cylinder" size="0.03 0.045"'),
        ("sphere", {"radius": 0.04}, 'type="sphere" size="0.04"'),
    )
    for forma, dimensioni, geom in casi:
        xml = f"""
        <mujoco>
          <worldbody>
            <body name="corpo"><freejoint/>
              <geom {geom} mass="0.7"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        nostra = np.array(principal_inertia(forma, dimensioni, 0.7))
        sua = np.array(model.body_inertia[1])
        _riga(
            f"inerzia {forma}",
            float(np.max(np.abs(nostra - sua))),
            0.0,
            "kg m^2 (scarto)",
            tolleranza=1e-9,
        )


def prove_sulle_pile(scene: int) -> None:
    """Sulle scene vere: assestamento, oggetti che fluttuano, compenetrazione."""
    import gymnasium as gym

    import physical_ai_mujoco.envs  # noqa: F401  (registra l'environment)

    durate: list[float] = []
    falliti = 0
    fluttuanti = 0
    penetrazione = 0.0
    deriva: list[float] = []

    for seed in range(scene):
        env = gym.make("TargetExtraction-v0", disable_env_checker=True, obs_mode="state")
        _, info = env.reset(seed=seed)
        simulatore = env.unwrapped.simulator

        for riga in info["release_log"]:
            durate.append(riga["settling_seconds"])
            if not riga["settled"]:
                falliti += 1

        for instance_id in simulatore.get_present_object_states():
            if not simulatore.supported_by(instance_id):
                fluttuanti += 1

        for indice in range(simulatore.data.ncon):
            penetrazione = max(penetrazione, -float(simulatore.data.contact[indice].dist))

        prima = {
            i: np.array(s.position)
            for i, s in simulatore.get_present_object_states().items()
        }
        simulatore.step_for(1.0)
        dopo = simulatore.get_present_object_states()
        deriva.append(
            max(
                float(np.linalg.norm(np.array(dopo[i].position) - prima[i]))
                for i in prima
            )
        )
        env.close()

    durate.sort()
    deriva.sort()
    print()
    print(f"  pile su {scene} scene, {len(durate)} assestamenti")
    print(
        f"    assestamento      mediana {durate[len(durate) // 2]:5.2f} s   "
        f"95mo {durate[int(0.95 * len(durate))]:5.2f} s   max {durate[-1]:5.2f} s"
    )
    print(f"    non assestati     {falliti} su {len(durate)}")
    print(f"    oggetti sospesi   {fluttuanti}   (atteso 0)")
    print(f"    compenetrazione   {penetrazione * 1000:.3f} mm")
    print(
        f"    deriva in 1 s     mediana {deriva[len(deriva) // 2] * 1000:.3f} mm   "
        f"peggiore {deriva[-1] * 1000:.3f} mm"
    )


def _riga(nome: str, misurato: float, atteso: float, unita: str, tolleranza: float) -> None:
    scarto = abs(misurato - atteso)
    esito = "ok  " if scarto <= tolleranza else "FUORI"
    print(
        f"  {esito} {nome:36s} misurato {misurato:10.5g}  "
        f"atteso {atteso:10.5g}  scarto {scarto:9.3g} {unita}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        type=int,
        default=0,
        help="numero di scene su cui eseguire anche le prove sulle pile (0 = nessuna)",
    )
    arguments = parser.parse_args()

    contatto = CONTATTO.get("contact", {})
    print("Parametri di contatto in prova:")
    print(
        f"  cono {contatto.get('cone', 'pyramidal')}, "
        f"impratio {contatto.get('impratio', 1.0)}, "
        f"noslip {int(contatto.get('noslip_iterations', 0))}, "
        f"passo {CONTATTO['timestep']} s, integratore {CONTATTO['integrator']}"
    )
    print()
    prova_caduta_libera()
    prova_scivolamento()
    prova_ribaltamento()
    prova_inerzia()
    if arguments.scene:
        prove_sulle_pile(arguments.scene)


if __name__ == "__main__":
    main()
