from physical_ai_mujoco.contracts import SynchronizedSensorPacket
from physical_ai_mujoco.sensors.source import SensorSource


class FakeSensorSource(SensorSource):
    def __init__(self, packet: SynchronizedSensorPacket):
        self.packet = packet
        self.capture_count = 0

    def capture(self) -> SynchronizedSensorPacket:
        self.capture_count += 1
        return self.packet
