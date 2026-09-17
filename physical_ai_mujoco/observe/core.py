"""API di osservazione. StereoCapture acquisisce immagini, senza ricostruzione."""

from abc import ABC, abstractmethod
from physical_ai_mujoco.contracts import Observation, PrivilegedState, ObjectObservation


class Observer(ABC):
    @abstractmethod
    def observe(self, simulator, target_id: str) -> Observation:
        """Produce l'informazione operativa; il backend e collegato dal coordinatore."""
        raise NotImplementedError


class ExactObserver(Observer):
    def privileged_state(self, simulator, target_id: str) -> PrivilegedState:
        objects = []
        for item in simulator.scene.objects:
            state = simulator.get_object_state(item.instance_id)
            objects.append(
                ObjectObservation(
                    item.instance_id,
                    tuple(state.position),
                    tuple(state.quaternion),
                    tuple(state.linear_velocity),
                    tuple(state.angular_velocity),
                    item.mass,
                    item.friction[0],
                    simulator.is_present(item.instance_id),
                    item.instance_id == target_id,
                )
            )
        return PrivilegedState(tuple(objects))

    def observe(self, simulator, target_id: str) -> Observation:
        return Observation(self.privileged_state(simulator, target_id).objects)


class StereoCapture:
    """Sensore simulato gia esistente: non e ancora la percezione della fase 1B."""

    def capture(self, simulator) -> dict:
        left, right = simulator.render_stereo()
        return {"rgb_left": left, "rgb_right": right}
