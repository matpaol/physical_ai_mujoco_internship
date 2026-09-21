"""Stima della geometria degli oggetti dai punti LiDAR associati."""

from dataclasses import replace

from physical_ai_mujoco.sensors.cloud import ObjectGeometry, estimate_geometry

from .localization import LocalizedDetection


class LidarGeometryEstimator:
    def __init__(
        self,
        minimum_points: int = 6,
        mesh_target_type_ids: tuple[str, ...] = (),
        cad_matchers: dict[str, object] | None = None,
    ):
        if minimum_points < 6:
            raise ValueError("Servono almeno sei punti per stimare la geometria")
        self.minimum_points = int(minimum_points)
        self.mesh_target_type_ids = frozenset(mesh_target_type_ids)
        self.cad_matchers = dict(cad_matchers or {})

    def estimate(
        self,
        localized: LocalizedDetection,
        target_type_id: str | None = None,
    ) -> ObjectGeometry | None:
        if len(localized.points) < self.minimum_points:
            return None
        target_types = self.mesh_target_type_ids | (
            frozenset((target_type_id,)) if target_type_id is not None else frozenset()
        )
        is_mesh_target = localized.detection.class_id in target_types
        matcher = self.cad_matchers.get(localized.detection.class_id)
        if matcher is not None:
            try:
                return matcher.match(localized.points).geometry
            except ValueError:
                # Un fit CAD cattivo non deve inventare una posa completa:
                # resta disponibile la stima conservativa della parte vista.
                pass
        # Per il target noto vogliamo gli ingombri orientati della nuvola.
        # Il classificatore geometrico generico scambierebbe gli otto vertici
        # di un parallelepipedo per una sfera, perche' sono equidistanti dal centro.
        geometry = estimate_geometry(
            localized.points,
            shape_hint="box" if is_mesh_target else None,
        )
        if is_mesh_target:
            geometry = replace(geometry, shape="mesh")
        return geometry
