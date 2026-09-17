"""Viewer e temporizzazione della presentazione, separati dal task."""

import time
import numpy as np
from physical_ai_mujoco.simulation import compat  # noqa: F401 (compatibilita viewer)

WINDOW_TITLE = "Gymnasium viewer - TargetExtraction-v0"


class SimulationViewer:
    def __init__(self, render_mode=None, realtime_factor=1.0):
        self.render_mode = render_mode
        self.realtime_factor = realtime_factor
        self.simulator = None
        self._mujoco_renderer = None
        self._window_named = False
        self._pacing_origin = None
        self.step_callback = None

    @property
    def renderer(self):
        return self._mujoco_renderer

    def render(self):
        if self._mujoco_renderer is not None:
            return self._mujoco_renderer.render("human")

    def begin_settle(self):
        self._pacing_origin = None

    def attach(self, simulator):
        self.simulator = simulator
        self._reset_viewer()

    def close(self):
        if self._mujoco_renderer is not None:
            self._mujoco_renderer.close()
            self._mujoco_renderer = None
        self.simulator = None

    def on_step(self) -> None:
        """Aggiorna la visualizzazione DURANTE l'assestamento.

        Senza questo si vedrebbe un aggiornamento per step, cioe' uno per
        rimozione: dei salti fra configurazioni statiche invece degli oggetti
        che si muovono.

        Due destinatari possibili:
        - il viewer di Gymnasium, quando render_mode == "human";
        - `step_callback`, un aggancio per chi disegna per conto proprio.
        """
        if self.step_callback is None and (
            self.render_mode != "human" or self._mujoco_renderer is None
        ):
            return

        # Alla massima velocita' non c'e' niente da sincronizzare: si disegna
        # ogni volta che capita e si va avanti.
        if self.realtime_factor <= 0:
            self._draw()
            return

        now = time.perf_counter()
        if self._pacing_origin is None:
            self._pacing_origin = (now, self.simulator.time)
        wall_origin, sim_origin = self._pacing_origin

        # Il tempo simulato che a quest'ora dovremmo aver raggiunto, se la
        # riproduzione andasse a `realtime_factor` volte il tempo reale.
        due = sim_origin + (now - wall_origin) * self.realtime_factor
        ahead = self.simulator.time - due

        if ahead < 0:
            # Siamo in ritardo: si SALTA il fotogramma e si lascia correre la
            # fisica. E' cio' che distingue una riproduzione a velocita' reale
            # da una al rallentatore — disegnare costa decine di millisecondi,
            # molto piu' del tempo simulato che quel fotogramma rappresenta, e
            # disegnarli tutti significa per forza andare piu' piano del reale.
            # Meno fotogrammi, stessa durata: e' quello che fa qualsiasi
            # riproduzione video quando la macchina non ce la fa.
            return

        self._draw()
        # Si dorme cio' che resta DOPO aver disegnato, non il budget intero.
        remaining = ahead - (time.perf_counter() - now)
        if remaining > 0:
            time.sleep(remaining)

    def _draw(self) -> None:
        if self.step_callback is not None:
            self.step_callback()
        if self.render_mode == "human" and self._mujoco_renderer is not None:
            self._mujoco_renderer.render("human")
            self._name_the_window()

    def _name_the_window(self) -> None:
        """Rinomina la finestra del viewer di Gymnasium.

        Gymnasium crea la sua finestra GLFW con il titolo letterale "mujoco"
        (gymnasium/envs/mujoco/mujoco_rendering.py, glfw.create_window). Qui
        le si da' un nome che dice cosa si sta guardando.
        """
        if self._window_named:
            return
        viewer = getattr(self._mujoco_renderer, "viewer", None)
        window = getattr(viewer, "window", None)
        if window is None:
            return
        try:
            import glfw

            glfw.set_window_title(window, WINDOW_TITLE)
            self._window_named = True
        except Exception:  # noqa: BLE001  (il titolo e' un dettaglio estetico)
            self._window_named = True

    def _reset_viewer(self) -> None:
        """(Ri)crea il viewer interattivo di Gymnasium.

        Il modello MuJoCo viene ricompilato a ogni reset (le dimensioni degli
        oggetti sono campionate per scena), quindi anche il viewer va
        ricreato: con render_mode="human" la finestra si richiude e riapre a
        ogni episodio. E' accettabile per ispezionare, non per allenare.
        """
        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        if self._mujoco_renderer is not None:
            self._mujoco_renderer.close()
        self._window_named = False
        self._mujoco_renderer = MujocoRenderer(
            self.simulator.model,
            self.simulator.data,
            default_cam_config={
                "distance": 2.0,
                "azimuth": 110.0,
                "elevation": -25.0,
                "lookat": np.array([0.0, 0.0, 0.1]),
            },
        )
