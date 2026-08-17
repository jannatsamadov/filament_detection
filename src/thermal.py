"""Thermal governor for a laptop RTX 2060 whose fan does not work.

Hard limits for this machine: GPU utilisation at or below 72%, pause above 78 C,
resume near 70 C. The card idles at 56-60 C, so there is only about 18 C of
headroom and it is reached quickly under sustained load.

The controller duty-cycles the training loop -- after each step it sleeps a
multiple of that step's compute time, so utilisation is the ratio of compute to
wall time and a target of 50% costs roughly 2x wall time.

Three behaviours matter and the first version of this file got the first one
wrong, cooking the card to 81 C on the opening epoch:

  * After a thermal pause the duty cycle must be permanently loosened. Resuming
    at the same rate that caused the overheat just reaches it again, and the run
    thrashes between pause and 78 C.
  * There is a soft band below the hard limit. Above `warn_temp` the governor
    backs off progressively instead of waiting for the emergency stop.
  * The duty cycle only tightens again when the card is genuinely cool, never
    merely because utilisation looks low during a pause.
"""
from __future__ import annotations

import subprocess
import time

__all__ = ["Governor", "gpu_status", "other_gpu_processes"]

_QUERY = ("nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,memory.used",
          "--format=csv,noheader,nounits")
_APPS = ("nvidia-smi", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits")


def gpu_status():
    """(utilisation %, temperature C, memory used MiB); (None, None, None) on failure."""
    try:
        out = subprocess.run(_QUERY, capture_output=True, text=True, timeout=10).stdout
        u, t, m = (x.strip() for x in out.strip().splitlines()[0].split(","))
        return int(u), int(t), int(m)
    except Exception:
        return None, None, None


def other_gpu_processes():
    """PIDs computing on the GPU other than this one.

    Duty-cycling bounds only the load of the process doing it, so a second
    training run silently defeats the thermal policy. Check before starting.
    """
    import os
    try:
        out = subprocess.run(_APPS, capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    me = os.getpid()
    pids = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        pid = line.split(",")[0].strip()
        if pid.isdigit() and int(pid) != me:
            pids.append(int(pid))
    return pids


class Governor:
    """Duty-cycles a training loop to respect utilisation and temperature caps.

        gov = Governor()
        for batch in loader:
            with gov.step():
                train_one_step(batch)     # must synchronise before leaving

    Two measurement lessons are baked in, both learned the hard way on this card:

    * **Utilisation is enforced from our own clock, never from nvidia-smi.** The
      governor reads the card right after its own sleep, so an instantaneous
      utilisation sample is biased towards zero; feeding that back made the
      controller accelerate indefinitely. Our compute-to-wall ratio is exact, so
      the cap is honoured by construction and `sleep_factor` has a hard floor.

    * **Temperature alone drives extra backoff, and only near the ceiling.** This
      card sits at 71-74 C at just 25% utilisation, so a soft band at 72 C put
      the controller permanently in retreat -- the backoff ran to 8x and training
      effectively stopped. The band now starts at 76 C, two degrees under the
      78 C stop, and the total backoff is capped so it cannot run away.
    """

    def __init__(self, target_util=50, warn_temp=76, pause_temp=78, resume_temp=70,
                 check_every=8, max_sleep=10.0, max_backoff=3.0, verbose=True):
        self.target_util = target_util
        self.warn_temp = warn_temp
        self.pause_temp = pause_temp
        self.resume_temp = resume_temp
        self.check_every = check_every
        self.max_sleep = max_sleep
        self.verbose = verbose

        # Floor implied by the utilisation target; the controller may slow down
        # from here for heat, but never speed up past it.
        self.floor_factor = max(1.0 / (target_util / 100.0) - 1.0, 0.0)
        self.max_factor = self.floor_factor * max_backoff + 1.0
        self.sleep_factor = self.floor_factor

        self.n = 0
        self.pauses = 0
        self.paused_seconds = 0.0
        self.backoffs = 0
        self.peak_temp = 0
        self.peak_mem = 0
        self.last_temp = None
        self.work_time = 0.0      # seconds actually spent computing
        self.sleep_time = 0.0     # seconds spent duty-cycling
        self._t0 = None

    @property
    def measured_duty(self):
        """Fraction of wall time spent computing -- the real utilisation bound."""
        total = self.work_time + self.sleep_time
        return 100.0 * self.work_time / total if total > 0 else 0.0

    def step(self):
        return _Step(self)

    def _begin(self):
        self._t0 = time.perf_counter()

    def _end(self):
        work = time.perf_counter() - self._t0
        self.work_time += work
        self.n += 1
        if self.sleep_factor > 0:
            nap = min(work * self.sleep_factor, self.max_sleep)
            time.sleep(nap)
            self.sleep_time += nap
        if self.n % self.check_every == 0:
            self._regulate()

    def _set_factor(self, value, why):
        old = self.sleep_factor
        self.sleep_factor = max(self.floor_factor, min(value, self.max_factor))
        if self.verbose and abs(self.sleep_factor - old) > 0.05:
            print(f"  [thermal] {why}: duty {old:.2f} -> {self.sleep_factor:.2f}x "
                  f"(measured {self.measured_duty:.0f}% of wall time computing)",
                  flush=True)

    def _regulate(self):
        _, temp, mem = gpu_status()
        if temp is None:
            return
        self.last_temp = temp
        self.peak_temp = max(self.peak_temp, temp)
        if mem is not None:
            self.peak_mem = max(self.peak_mem, mem)

        if temp >= self.pause_temp:
            self.pauses += 1
            self._set_factor(self.sleep_factor * 1.5 + 0.2, f"{temp}C pause #{self.pauses}")
            t_start = time.time()
            if self.verbose:
                print(f"  [thermal] {temp}C >= {self.pause_temp} -- pausing until "
                      f"{self.resume_temp}C", flush=True)
            while True:
                time.sleep(10.0)
                _, t2, _ = gpu_status()
                if t2 is None or t2 <= self.resume_temp:
                    break
            waited = time.time() - t_start
            self.paused_seconds += waited
            self.sleep_time += waited
            if self.verbose:
                print(f"  [thermal] resumed after {waited:.0f}s at {t2}C", flush=True)
            return

        if temp >= self.warn_temp:
            self.backoffs += 1
            self._set_factor(self.sleep_factor * 1.12 + 0.05, f"{temp}C near ceiling")
        elif temp <= self.resume_temp and self.sleep_factor > self.floor_factor:
            self._set_factor(self.sleep_factor * 0.9, f"{temp}C cooled")

    def summary(self):
        return (f"steps={self.n} measured_duty={self.measured_duty:.1f}% "
                f"peak_temp={self.peak_temp}C last_temp={self.last_temp}C "
                f"peak_mem={self.peak_mem}MiB pauses={self.pauses} "
                f"backoffs={self.backoffs} paused={self.paused_seconds:.0f}s "
                f"sleep_factor={self.sleep_factor:.2f} (floor {self.floor_factor:.2f}, "
                f"cap {self.max_factor:.2f})")


class _Step:
    def __init__(self, gov):
        self.gov = gov

    def __enter__(self):
        self.gov._begin()
        return self.gov

    def __exit__(self, *exc):
        self.gov._end()
        return False
