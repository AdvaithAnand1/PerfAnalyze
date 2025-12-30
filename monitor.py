# monitor.py
"""
Real-time telemetry bridge using HWiNFO shared memory.

Requirements:
  - HWiNFO running
  - Sensors window open
  - Settings -> "Shared Memory Support" enabled
  - Python 3.8+ (for multiprocessing.shared_memory)
  - pip install numpy psutil
"""

import struct
from multiprocessing import shared_memory

import numpy as np
import psutil

# Name of the shared memory block created by HWiNFO
HWINF0_SHM_NAME = r"Global\HWiNFO_SENS_SM2"

# Header layout (reverse-engineered)
# uint32 magic, uint32 version, uint32 version2,
# int64 last_update,
# uint32 sensor_section_offset, uint32 sensor_element_size, uint32 sensor_element_count,
# uint32 entry_section_offset,  uint32 entry_element_size,  uint32 entry_element_count
HEADER_STRUCT = struct.Struct("<IIIqIIIIII")

# Sensor descriptor:
# uint32 id, uint32 instance, char name_original[128], char name_user[128]
SENSOR_STRUCT = struct.Struct("<II128s128s")

# Entry descriptor:
# uint32 type, uint32 sensor_index, uint32 id,
# char name_original[128], char name_user[128], char unit[16],
# double value, value_min, value_max, value_avg
ENTRY_STRUCT = struct.Struct("<III128s128s16sdddd")


def _cstr(b: bytes) -> str:
    """Convert C-style char array to Python string."""
    return b.split(b"\0", 1)[0].decode(errors="ignore").strip()


class HWInfoReader:
    """
    Reads live sensor data from HWiNFO shared memory.

    Usage:
        r = HWInfoReader()
        snap = r.snapshot()   # dict[label] = value
    """

    def __init__(self):
        try:
            self.shm = shared_memory.SharedMemory(
                name=HWINF0_SHM_NAME, create=False
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                "HWiNFO shared memory not found.\n"
                "Make sure:\n"
                "  - HWiNFO is running\n"
                "  - The sensors window is open\n"
                "  - 'Shared Memory Support' is enabled in HWiNFO settings"
            ) from e

        self.buf = self.shm.buf
        self._parse_header()

    def _parse_header(self):
        (
            magic,
            version,
            version2,
            last_update,
            sensor_off,
            sensor_size,
            sensor_count,
            entry_off,
            entry_size,
            entry_count,
        ) = HEADER_STRUCT.unpack_from(self.buf, 0)

        # magic = 'SiWH' in little-endian
        if magic != 0x48576953:
            raise RuntimeError("Invalid HWiNFO shared memory header magic")

        self.version = version
        self.version2 = version2
        self.last_update = last_update
        self.sensor_off = sensor_off
        self.sensor_size = sensor_size
        self.sensor_count = sensor_count
        self.entry_off = entry_off
        self.entry_size = entry_size
        self.entry_count = entry_count

    def snapshot(self) -> dict[str, float]:
        """
        Read one snapshot of all entries as { label: value }.
        Labels correspond to the HWiNFO sensor labels (user label preferred).
        """
        # Sensors are mostly for indexing / debugging; not strictly needed for values
        sensors = []
        for i in range(self.sensor_count):
            offset = self.sensor_off + i * self.sensor_size
            sid, inst, name_orig, name_user = SENSOR_STRUCT.unpack_from(
                self.buf, offset
            )
            sensors.append(
                {
                    "id": sid,
                    "instance": inst,
                    "name_orig": _cstr(name_orig),
                    "name_user": _cstr(name_user),
                }
            )

        snap: dict[str, float] = {}
        for j in range(self.entry_count):
            offset = self.entry_off + j * self.entry_size
            (
                entry_type,
                sensor_index,
                entry_id,
                name_orig,
                name_user,
                unit,
                value,
                vmin,
                vmax,
                vavg,
            ) = ENTRY_STRUCT.unpack_from(self.buf, offset)

            label = _cstr(name_user) or _cstr(name_orig)
            if not label:
                continue
            snap[label] = float(value)

        return snap

    def close(self):
        self.shm.close()


# ========= Feature engineering =========

FEATURE_NAMES = [
    # CPU
    "Total CPU Usage [%]",
    "Core Clocks (avg) [MHz]",
    "Core C0 Residency (avg) [%]",
    "Core C1 Residency (avg) [%]",
    "Core C6 Residency (avg) [%]",
    "CPU Core [°C]",
    "CPU SOC [°C]",
    "CPU Package Power [W]",
    # GPU
    "GPU Temperature [°C]",
    "GPU ASIC Power [W]",
    "GPU Utilization [%]",
    "GPU Memory Usage [MB]",
    # Derived
    "Core Clock StdDev [MHz]",
    "RAM Usage [%]",
    "CPU Physical Cores",
    "CPU Logical Cores",
]


def _core_clock_std(snapshot: dict[str, float]) -> float:
    """
    Std dev of per-core clocks (MHz).
    High -> single/multi-core boost (light load), low -> uniform all-core load.
    """
    core_keys = [
        k
        for k in snapshot.keys()
        if "Clock" in k
        and "[MHz]" in k
        and "Core Clocks (avg)" not in k
        and "GPU" not in k
    ]
    if not core_keys:
        return 0.0
    vals = np.array([snapshot[k] for k in core_keys], dtype=np.float32)
    return float(vals.std())


def _ram_usage_percent() -> float:
    """Global RAM usage percentage."""
    return float(psutil.virtual_memory().percent)


def _core_counts() -> tuple[int, int]:
    """(physical_cores, logical_cores)."""
    return (
        psutil.cpu_count(logical=False) or 0,
        psutil.cpu_count(logical=True) or 0,
    )


def snapshot_to_features(snapshot: dict[str, float]) -> np.ndarray:
    """
    Convert a raw HWiNFO snapshot dict into a fixed-order feature vector.

    The order is defined by FEATURE_NAMES and must be the same
    for both training and inference.
    """
    clock_std = _core_clock_std(snapshot)
    ram_pct = _ram_usage_percent()
    phys, logical = _core_counts()

    base = [
        snapshot.get("Total CPU Usage [%]", 0.0),
        snapshot.get("Core Clocks (avg) [MHz]", 0.0),
        snapshot.get("Core C0 Residency (avg) [%]", 0.0),
        snapshot.get("Core C1 Residency (avg) [%]", 0.0),
        snapshot.get("Core C6 Residency (avg) [%]", 0.0),
        snapshot.get("CPU Core [°C]", 0.0),
        snapshot.get("CPU SOC [°C]", 0.0),
        snapshot.get("CPU Package Power [W]", 0.0),
        snapshot.get("GPU Temperature [°C]", 0.0),
        snapshot.get("GPU ASIC Power [W]", 0.0),
        snapshot.get("GPU Utilization [%]", 0.0),
        snapshot.get("GPU Memory Usage [MB]", 0.0),
        clock_std,
        ram_pct,
        float(phys),
        float(logical),
    ]

    return np.asarray(base, dtype=np.float32)


# Public one-liner
_reader: HWInfoReader | None = None


def get_telemetry() -> np.ndarray:
    """
    One-call helper: grab current feature vector for the system state.
    """
    global _reader
    if _reader is None:
        _reader = HWInfoReader()
    snap = _reader.snapshot()
    return snapshot_to_features(snap)


if __name__ == "__main__":
    # Quick sanity test
    import time

    r = HWInfoReader()
    for _ in range(5):
        s = r.snapshot()
        x = snapshot_to_features(s)
        print("Features:", x)
        time.sleep(0.5)
