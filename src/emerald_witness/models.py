from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MeasurementSetting:
    setting_id: int
    bases: dict[int, str]
    measured_stabilizers: list[int]

    def to_dict(self) -> dict[str, object]:
        return {
            "setting_id": self.setting_id,
            "bases": {str(index): basis for index, basis in sorted(self.bases.items())},
            "measured_stabilizers": list(self.measured_stabilizers),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "MeasurementSetting":
        return cls(
            setting_id=int(payload["setting_id"]),
            bases={int(index): str(basis) for index, basis in dict(payload["bases"]).items()},
            measured_stabilizers=[int(index) for index in list(payload["measured_stabilizers"])],
        )

    def basis_string(self, num_qubits: int) -> str:
        return "".join(self.bases[index] for index in range(num_qubits))
