from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("mujoco")

import physical_ai_mujoco.envs  # noqa: E402,F401  (registra l'environment)
from physical_ai_mujoco.envs.target_extraction import (  # noqa: E402
    STATE_FEATURES_PER_OBJECT,
)


def make_env(**kwargs):
    return gym.make("TargetExtraction-v0", disable_env_checker=True, **kwargs)


@pytest.fixture
def env():
    # Profilo di misura: deve osservare tutte le rimozioni anche quando il
    # target fisso PFM-1 viene scelto presto dalla policy casuale.
    environment = make_env(obs_mode="state", terminate_on_target=False)
    yield environment
    environment.close()


def random_rollout(environment, seed: int):
    """Rollout con policy casuale mascherata. Ritorna la traccia degli step."""
    _, info = environment.reset(seed=seed)
    generator = np.random.default_rng(seed)
    trace = []
    done = False

    while not done:
        action = int(generator.choice(np.flatnonzero(info["action_mask"])))
        _, reward, terminated, truncated, info = environment.step(action)
        trace.append((action, round(reward, 9), round(info["disturbance_step"], 9)))
        done = terminated or truncated

    return trace


# --------------------------------------------------------------- spazi e API


def test_state_observation_matches_space(env):
    observation, _ = env.reset(seed=0)
    assert env.observation_space.contains(observation)
    object_count = env.unwrapped.object_count
    assert observation.shape == (object_count * STATE_FEATURES_PER_OBJECT,)


def test_action_space_matches_object_count(env):
    env.reset(seed=0)
    assert env.action_space.n == env.unwrapped.object_count


@pytest.mark.parametrize("obs_mode", ["stereo", "both"])
def test_image_observations_match_space(obs_mode):
    environment = make_env(obs_mode=obs_mode)
    try:
        observation, _ = environment.reset(seed=0)
        assert environment.observation_space.contains(observation)
        assert observation["rgb_left"].dtype == np.uint8
        for side in ("rgb_left", "rgb_right"):
            np.testing.assert_array_equal(observation[side][:, :, 0], observation[side][:, :, 1])
            np.testing.assert_array_equal(observation[side][:, :, 1], observation[side][:, :, 2])
    finally:
        environment.close()


def test_stereo_pair_has_parallax():
    """Le due camere devono vedere la scena da punti diversi.

    Se il rig fosse costruito male (baseline nulla, o camere coincidenti) le
    due immagini sarebbero identiche e la distillazione stereo perderebbe ogni
    informazione di profondita': questo test lo impedisce.
    """
    environment = make_env(obs_mode="stereo")
    try:
        observation, _ = environment.reset(seed=0)
        left = observation["rgb_left"].astype(int)
        right = observation["rgb_right"].astype(int)
        assert not np.array_equal(left, right)
        assert np.abs(left - right).mean() > 1.0
    finally:
        environment.close()


def test_stereo_disparity_matches_geometry():
    """La disparita' misurata sui pixel deve valere f * B / z.

    Verifica che il rig sia davvero rettificato e con la baseline dichiarata:
    si localizza la sfera (unico oggetto verde) nelle due immagini, si misura
    lo scostamento orizzontale e lo si confronta con la previsione geometrica
    calcolata dalla posa reale della camera nel modello. Se le camere fossero
    orientate male, o la baseline fosse sbagliata, lo student imparerebbe una
    profondita' falsa senza che nulla segnali l'errore.
    """
    import mujoco

    # Scena dedicata: una sola sfera verde. La sfera e' l'unica forma il cui
    # centroide di sagoma coincide con il centro del corpo da qualunque punto
    # di vista, quindi e' l'unica su cui questa misura ha senso. E la scena e'
    # separata da quella di produzione perche' la geometria del rig non
    # dipende da cosa si mette nel mucchio.
    environment = make_env(
        obs_mode="stereo",
        scene_rules_path=str(Path(__file__).parent / "data/stereo_scene_rules.json"),
    )
    try:
        observation, _ = environment.reset(seed=5)
        simulator = environment.unwrapped.simulator
        rig = simulator.scene.stereo_camera

        sphere = next(
            item for item in simulator.scene.objects if item.shape == "sphere"
        )
        truth = np.asarray(
            simulator.get_object_state(sphere.instance_id).position
        )

        masks_left = simulator.render_instance_masks("cam_left")
        masks_right = simulator.render_instance_masks("cam_right")
        left = _mask_centroid(masks_left[sphere.instance_id])
        right = _mask_centroid(masks_right[sphere.instance_id])
        assert left is not None and right is not None, "sfera non visibile"

        camera_id = mujoco.mj_name2id(
            simulator.model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_left"
        )
        camera_position = simulator.data.cam_xpos[camera_id].copy()
        camera_matrix = simulator.data.cam_xmat[camera_id].reshape(3, 3)
        # La camera MuJoCo guarda lungo -z del proprio frame.
        forward = -camera_matrix[:, 2]
        depth = float(np.dot(truth - camera_position, forward))

        expected = rig.focal_length_px() * rig.baseline / depth
        measured = left[0] - right[0]

        assert abs(measured - expected) < 3.0, (
            f"disparita' misurata {measured:.2f} px contro {expected:.2f} px attesi"
        )
        # Rig rettificato: nessuna disparita' verticale.
        assert abs(left[1] - right[1]) < 2.0
    finally:
        environment.close()


def _mask_centroid(mask: np.ndarray):
    if mask.sum() < 20:
        return None
    rows, columns = np.nonzero(mask)
    return columns.mean(), rows.mean()


# ------------------------------------------------------------------ dinamica


def test_removal_shrinks_action_mask(env):
    _, info = env.reset(seed=1)
    assert info["action_mask"].all()

    action = int(np.flatnonzero(info["action_mask"])[0])
    _, _, _, _, info = env.step(action)

    assert not info["action_mask"][action]
    assert info["action_mask"].sum() == env.unwrapped.object_count - 1


def test_invalid_action_is_penalised_not_raised(env):
    _, info = env.reset(seed=1)
    action = int(np.flatnonzero(info["action_mask"])[0])
    env.step(action)

    present_before = env.unwrapped.simulator.present_objects()
    _, reward, terminated, _, info = env.step(action)

    assert info["invalid_action"] is True
    assert reward < 0
    assert not terminated
    assert env.unwrapped.simulator.present_objects() == present_before


def test_removed_object_leaves_the_scene(env):
    env.reset(seed=1)
    simulator = env.unwrapped.simulator
    instance_id = simulator.present_objects()[0]

    env.step(env.unwrapped._object_ids.index(instance_id))

    assert instance_id not in simulator.present_objects()
    # Parcheggiato lontano, fermo e invisibile.
    state = simulator.get_object_state(instance_id)
    assert state.position[2] < -10.0
    assert np.allclose(state.linear_velocity, 0.0)


def test_removing_target_ends_the_episode_only_as_a_task():
    """Come TASK il target chiude l'episodio; in MISURA no.

    E' la stessa scena e la stessa rimozione: cambia solo un flag di
    configurazione. Se un giorno un modulo dovesse guardare in che modalita'
    si trova, questo test e' il posto dove accorgersene.
    """
    as_task = make_env(obs_mode="state", terminate_on_target=True)
    try:
        _, info = as_task.reset(seed=3)
        _, _, terminated, _, info = as_task.step(info["target_index"])
        assert terminated
        assert info["target_removed"] is True
    finally:
        as_task.close()

    measuring = make_env(obs_mode="state", terminate_on_target=False)
    try:
        _, info = measuring.reset(seed=3)
        _, _, terminated, _, info = measuring.step(info["target_index"])
        assert info["target_removed"] is True
        assert not terminated, "in misura si continua fino a svuotare la scena"
    finally:
        measuring.close()


def test_measurement_run_empties_the_scene(env):
    """Senza fallimento l'episodio finisce solo quando non resta niente."""
    _, info = env.reset(seed=4)
    object_count = env.unwrapped.object_count
    generator = np.random.default_rng(0)
    removals = 0
    done = False

    while not done:
        action = int(generator.choice(np.flatnonzero(info["action_mask"])))
        _, _, terminated, truncated, info = env.step(action)
        removals += 1
        done = terminated or truncated

    assert removals == object_count, "una rimozione valida per oggetto"
    assert info["scene_is_empty"] is True
    assert not info["action_mask"].any()


def test_step_disturbance_is_per_move_not_cumulative(env):
    """`disturbance_step` misura la singola mossa, `disturbance_total` la somma."""
    _, info = env.reset(seed=7)
    generator = np.random.default_rng(1)
    per_move = []
    done = False

    while not done:
        action = int(generator.choice(np.flatnonzero(info["action_mask"])))
        _, _, terminated, truncated, info = env.step(action)
        per_move.append(info["disturbance_step"])
        done = terminated or truncated

    assert info["disturbance_total"] == pytest.approx(sum(per_move))
    # Un valore cumulato non potrebbe mai scendere; uno per mossa si'.
    assert len(per_move) > 1


def test_scene_is_settled_before_the_first_measurement(env):
    """Al reset la scena deve essere ferma, non ancora in caduta.

    E' la condizione che rende sensato il disturbo: se si misurasse mentre gli
    oggetti stanno ancora atterrando, il primo passo registrerebbe la caduta
    come se fosse un effetto della rimozione.
    """
    _, info = env.reset(seed=2)
    assert info["settled"] is True, "assestamento non raggiunto entro il timeout"

    # La verifica indipendente e' la stessa che usa il criterio di quiete, ma
    # scritta qui a mano: in mezzo secondo di simulazione in piu' nessun
    # oggetto deve muoversi oltre la tolleranza dichiarata. Rileggere la
    # velocita' istantanea non direbbe niente — e' proprio la grandezza che i
    # picchi di contatto rendono inaffidabile.
    simulator = env.unwrapped.simulator
    tolerances = simulator.scene.simulation
    before = {
        instance_id: np.array(state.position)
        for instance_id, state in simulator.get_present_object_states().items()
    }
    simulator.step_for(tolerances.stable_duration)

    for instance_id, state in simulator.get_present_object_states().items():
        moved = np.linalg.norm(np.array(state.position) - before[instance_id])
        assert moved < tolerances.linear_settle_tolerance, (
            f"{instance_id} si e' spostato di {moved * 1000:.3f} mm dopo "
            "l'assestamento"
        )


def test_simulator_reset_restores_removed_objects(env):
    env.reset(seed=1)
    simulator = env.unwrapped.simulator
    everything = tuple(item.instance_id for item in simulator.scene.objects)

    simulator.remove_object(everything[0])
    assert simulator.present_objects() != everything

    simulator.reset()
    assert simulator.present_objects() == everything


# ------------------------------------------------------------ visualizzazione


def test_frame_capture_is_off_by_default(env):
    _, info = env.reset(seed=1)
    assert info["frames"] == []


def test_ground_is_a_plane_without_edges(env):
    """Il terreno di Fase 0B e' un pavimento, non un tavolo.

    Un box ha bordi, e un oggetto che li supera cade nel vuoto: lo spostamento
    che ne risulta inquinerebbe la misura del disturbo con un artefatto che
    non c'entra con la rimozione.
    """
    env.reset(seed=1)
    ground = env.unwrapped.simulator.scene.ground
    assert ground.shape == "plane"
    assert ground.is_bounded is False
    assert ground.surface_thickness == 0.0
    # La superficie di appoggio resta a z=0 come per il tavolo.
    assert ground.pose.position[2] == 0.0


def test_frame_capture_returns_frames():
    environment = make_env(
        obs_mode="state", capture_camera="cam_overview", capture_stride=16
    )
    try:
        _, info = environment.reset(seed=1)
        assert len(info["frames"]) > 1
        frame = info["frames"][0]
        assert frame.ndim == 3 and frame.shape[2] == 3
        assert frame.dtype == np.uint8
        # Frame successivi diversi fra loro: la scena si sta muovendo.
        assert not np.array_equal(info["frames"][0], info["frames"][-1])
    finally:
        environment.close()


def test_overview_camera_always_exists(env):
    import mujoco

    env.reset(seed=1)
    camera_id = mujoco.mj_name2id(
        env.unwrapped.simulator.model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_overview"
    )
    assert camera_id >= 0


# -------------------------------------------------------------- determinismo


def test_same_seed_gives_same_episode(env):
    assert random_rollout(env, seed=11) == random_rollout(env, seed=11)


def test_different_seeds_give_different_scenes(env):
    _, first = env.reset(seed=11)
    _, second = env.reset(seed=12)
    assert first["scene_seed"] != second["scene_seed"]


# ------------------------------------------------------- compatibilita' API


def test_passes_gymnasium_env_checker():
    from gymnasium.utils.env_checker import check_env

    environment = make_env(obs_mode="state")
    try:
        check_env(environment.unwrapped, skip_render_check=True)
    finally:
        environment.close()


# ------------------------------------------------- numero oggetti e finestra


@pytest.mark.parametrize("object_count", [1, 2, 4, 8])
def test_object_count_is_configurable(object_count):
    environment = make_env(obs_mode="state", object_count=object_count)
    try:
        observation, info = environment.reset(seed=0)
        assert environment.action_space.n == object_count
        assert len(info["present_objects"]) == object_count
        assert observation.shape == (
            object_count * STATE_FEATURES_PER_OBJECT,
        )
    finally:
        environment.close()


def test_fixed_shapes_keeps_the_same_compiled_model():
    """Con resample_shapes=False il modello non va ricompilato.

    E' la condizione che permette a una finestra interattiva di restare
    aperta fra un episodio e l'altro: un viewer e' legato a un modello
    compilato, e ricompilarlo lo invaliderebbe.
    """
    environment = make_env(
        obs_mode="state", object_count=3, resample_shapes=False
    )
    try:
        models = []
        shapes = []
        positions = []
        for episode in range(3):
            environment.reset(seed=episode)
            simulator = environment.unwrapped.simulator
            models.append(id(simulator.model))
            shapes.append(
                tuple(
                    round(value, 6)
                    for item in simulator.scene.objects
                    for value in item.size.values()
                )
            )
            positions.append(
                tuple(
                    round(value, 4)
                    for name in simulator.present_objects()
                    for value in simulator.get_object_state(name).position
                )
            )

        assert len(set(models)) == 1, "il modello e' stato ricompilato"
        assert len(set(shapes)) == 1, "le forme sono cambiate"
        assert len(set(positions)) == 3, "le pose non sono state rimescolate"
    finally:
        environment.close()


def test_default_resamples_shapes():
    environment = make_env(obs_mode="state", object_count=3)
    try:
        shapes = []
        for episode in range(3):
            environment.reset(seed=episode)
            shapes.append(
                tuple(
                    round(value, 6)
                    for item in environment.unwrapped.simulator.scene.objects
                    for value in item.size.values()
                )
            )
        assert len(set(shapes)) == 3
    finally:
        environment.close()


# --------------------------------------------------- aggancio per disegnare


def test_step_callback_is_called_during_settling():
    """L'hook che pilota un viewer esterno deve scattare piu' volte per step.

    E' cio' che distingue un rollout ANIMATO da una sequenza di salti: senza,
    la visualizzazione verrebbe ridisegnata una volta per rimozione.
    """
    environment = make_env(obs_mode="state", object_count=3, realtime_factor=0)
    try:
        calls = {"count": 0}
        environment.unwrapped.step_callback = lambda: calls.__setitem__(
            "count", calls["count"] + 1
        )

        _, info = environment.reset(seed=0)
        during_reset = calls["count"]
        assert during_reset > 5, "il viewer non verrebbe animato durante la caduta"

        environment.step(int(np.flatnonzero(info["action_mask"])[0]))
        assert calls["count"] - during_reset > 5
    finally:
        environment.unwrapped.step_callback = None
        environment.close()


# ------------------------------------------------ inerzia e centro di massa


@pytest.mark.parametrize("shape,size", [
    ("box", {"x": 0.12, "y": 0.08, "z": 0.03}),
    ("cylinder", {"radius": 0.06, "height": 0.04}),
    ("sphere", {"radius": 0.05}),
])
def test_our_inertia_matches_what_mujoco_derives(shape, size):
    """Le nostre formule devono dare gli stessi numeri del compilatore MuJoCo.

    Contano perche' spostare il centro di massa obbliga a dichiarare l'inerzia
    a mano: da quel momento la fonte siamo noi, e se sbagliassimo una formula
    la fisica cambierebbe senza che niente protesti.
    """
    import mujoco

    from physical_ai_mujoco.scene.scene_description import principal_inertia

    mass = 0.4
    # if/elif e non un dizionario: un dizionario valuterebbe tutte e tre le
    # espressioni, e size['radius'] non esiste per un box.
    if shape == "box":
        size_attribute = f"{size['x']/2} {size['y']/2} {size['z']/2}"
    elif shape == "cylinder":
        size_attribute = f"{size['radius']} {size['height']/2}"
    else:
        size_attribute = f"{size['radius']}"
    xml = f"""
    <mujoco>
      <worldbody>
        <body name="probe">
          <freejoint/>
          <geom type="{shape}" size="{size_attribute}" mass="{mass}"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    derived = model.body_inertia[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "probe")
    ]
    assert np.allclose(derived, principal_inertia(shape, size, mass), rtol=1e-6)


def test_center_of_mass_is_offset_and_stays_inside_the_object():
    """Il centro di massa e' spostato, ma resta dentro l'oggetto.

    Fuori dalla sagoma non corrisponderebbe a nessuna distribuzione di massa
    possibile, e il comportamento sarebbe un artefatto invece che un caso
    difficile.
    """
    import mujoco

    from physical_ai_mujoco.scene.scene_description import half_extents

    environment = make_env(obs_mode="state", object_count=4)
    try:
        environment.reset(seed=0)
        simulator = environment.unwrapped.simulator
        model = simulator.model
        jitter = float(
            environment.unwrapped._scene_rules["randomisation"][
                "center_of_mass_jitter"
            ]
        )
        assert jitter > 0.0, "Fase 0B deve randomizzare il centro di massa"

        offsets = []
        for item in simulator.scene.objects:
            # Per una mesh MuJoCo ricava centro di massa e inerzia dal volume
            # STL; questo test riguarda la randomizzazione dei solidi primitivi.
            if item.shape == "mesh":
                continue
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, item.instance_id
            )
            # Cio' che chiede la descrizione e' cio' che finisce nel modello.
            assert np.allclose(
                model.body_ipos[body_id], item.center_of_mass, atol=1e-9
            )
            for offset, extent in zip(
                item.center_of_mass, half_extents(item.shape, item.size)
            ):
                assert abs(offset) <= jitter * extent + 1e-12
                offsets.append(abs(offset) / extent)

        assert max(offsets) > 0.01, "nessuno spostamento: la randomizzazione non agisce"
    finally:
        environment.close()


def test_uniform_objects_keep_the_compiler_derived_inertia():
    """Senza jitter nulla cambia rispetto a prima.

    E' la garanzia che la randomizzazione sia una AGGIUNTA e non una
    riscrittura silenziosa della fisica di tutte le scene.
    """
    import json
    import tempfile
    from pathlib import Path as _Path

    import mujoco

    from physical_ai_mujoco.envs.target_extraction import PROJECT_ROOT

    rules = json.loads(
        (PROJECT_ROOT / "configs/phase_0b/scene_rules.json").read_text(encoding="utf-8")
    )
    rules["randomisation"]["center_of_mass_jitter"] = 0.0
    with tempfile.TemporaryDirectory() as folder:
        path = _Path(folder) / "scene_rules.json"
        path.write_text(json.dumps(rules), encoding="utf-8")
        environment = make_env(
            obs_mode="state", object_count=3, scene_rules_path=str(path)
        )
        try:
            environment.reset(seed=0)
            simulator = environment.unwrapped.simulator
            for item in simulator.scene.objects:
                assert item.center_of_mass == (0.0, 0.0, 0.0)
                if item.shape == "mesh":
                    continue
                body_id = mujoco.mj_name2id(
                    simulator.model, mujoco.mjtObj.mjOBJ_BODY, item.instance_id
                )
                assert np.allclose(simulator.model.body_ipos[body_id], 0.0)
        finally:
            environment.close()


# ------------------------------------------------------- magazzino di scene


def test_pool_reuses_scenes_without_rebuilding_them():
    """Col magazzino pieno, gli episodi ripartono da scene gia' costruite."""
    environment = make_env(obs_mode="state", object_count=3, scene_pool_size=2)
    try:
        prese_dal_magazzino = 0
        for seed in range(8):
            _, info = environment.reset(seed=seed)
            prese_dal_magazzino += int(info["from_pool"])
        # I primi due episodi riempiono il magazzino, quindi sono per forza
        # costruiti; degli altri sei, con il 10% di scene fresche, la
        # stragrande maggioranza deve venire dal magazzino.
        assert prese_dal_magazzino >= 4
    finally:
        environment.close()


def test_pool_episode_starts_from_a_settled_scene():
    """Una scena ripresa dal magazzino e' ferma come una appena costruita.

    E' la proprieta' che rende il magazzino sicuro: se le pose ripristinate
    non fossero quelle di quiete, il primo passo misurerebbe come disturbo un
    movimento che nessuna rimozione ha causato.
    """
    environment = make_env(obs_mode="state", object_count=3, scene_pool_size=1)
    try:
        environment.reset(seed=0)  # riempie il magazzino
        _, info = environment.reset(seed=1)
        assert info["from_pool"] is True

        simulatore = environment.unwrapped.simulator
        tolleranze = simulatore.scene.simulation
        prima = {
            instance_id: np.array(state.position)
            for instance_id, state in simulatore.get_present_object_states().items()
        }
        simulatore.step_for(tolleranze.stable_duration)
        for instance_id, state in simulatore.get_present_object_states().items():
            spostamento = np.linalg.norm(np.array(state.position) - prima[instance_id])
            assert spostamento < tolleranze.linear_settle_tolerance
    finally:
        environment.close()


def test_pool_does_not_leak_removals_between_episodes():
    """Le rimozioni di un episodio non sopravvivono a quello successivo.

    Il magazzino ripristina anche `contype`/`conaffinity`: senza, un oggetto
    tolto in un episodio resterebbe invisibile alle collisioni in tutti quelli
    ripresi dalla stessa scena, e la scena si svuoterebbe da sola.
    """
    environment = make_env(obs_mode="state", object_count=3, scene_pool_size=1)
    try:
        environment.reset(seed=0)
        for seed in range(1, 5):
            _, info = environment.reset(seed=seed)
            presenti = environment.unwrapped.simulator.present_objects()
            assert len(presenti) == 3, (
                f"episodio {seed}: {len(presenti)} oggetti invece di 3"
            )
            assert int(np.asarray(info["action_mask"]).sum()) == 3
            environment.step(int(np.flatnonzero(info["action_mask"])[0]))
    finally:
        environment.close()


def test_pool_snapshot_refuses_a_different_scene():
    """Ripristinare la fotografia di un'altra scena deve fallire, non passare.

    I vettori delle pose hanno la stessa forma per scene con lo stesso numero
    di oggetti: MuJoCo accetterebbe lo scambio in silenzio, producendo una
    scena plausibile ma sbagliata.
    """
    primo = make_env(obs_mode="state", object_count=3)
    secondo = make_env(obs_mode="state", object_count=3)
    try:
        primo.reset(seed=0)
        secondo.reset(seed=1)
        fotografia = primo.unwrapped.simulator.snapshot()
        with pytest.raises(ValueError, match="un'altra scena"):
            secondo.unwrapped.simulator.restore(fotografia)
    finally:
        primo.close()
        secondo.close()
