"""Measure memory footprint of importing tensorflow."""

import ctypes
import os
import time


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_size_t),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def mem_mb():
    psapi = ctypes.WinDLL("psapi")
    pmc = PROCESS_MEMORY_COUNTERS()
    pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    h = ctypes.windll.kernel32.GetCurrentProcess()
    psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
    return pmc.WorkingSetSize / (1024 * 1024)


if __name__ == "__main__":
    print(f"Before TF import: {mem_mb():.1f} MB")
    t = time.time()
    import tensorflow as tf  # noqa: F401

    print(f"After TF import: {mem_mb():.1f} MB")
    print(f"TF import time: {time.time() - t:.1f}s")
