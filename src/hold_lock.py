"""Hold the shared GPU lock on behalf of a training that started without it.

The spine_v2 run was launched before the shared lock existed, so the card was
being used while `gpu_lock.py status` reported FREE -- exactly the situation the
lock is there to prevent. This keeps the lock written and its heartbeat fresh
for as long as that process is alive, then releases it.

Only for a run already in flight. New runs claim the lock themselves.
"""
from __future__ import annotations

import json, os, sys, time
from pathlib import Path

SHARED = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SHARED))
from gpu_lock import LOCK_PATH, _pid_alive, _now   # noqa: E402


def main(pid: int, project: str, tag: str, progress: str = "running", every: float = 60.0):
    if not _pid_alive(pid):
        print(f"pid {pid} is not running; nothing to hold")
        return 1
    started = _now()
    print(f"holding the GPU lock for {project}/{tag} on behalf of pid {pid}")
    try:
        while _pid_alive(pid):
            LOCK_PATH.write_text(json.dumps({
                "project": project, "tag": tag, "pid": pid,
                "started": started, "progress": progress, "heartbeat": _now(),
            }, indent=2), encoding="utf-8")
            time.sleep(every)
    finally:
        try:
            info = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
            if int(info.get("pid", -1)) == pid:
                LOCK_PATH.unlink(missing_ok=True)
                print("released")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]),
                  sys.argv[2] if len(sys.argv) > 2 else "filament_detection",
                  sys.argv[3] if len(sys.argv) > 3 else "spine_v2",
                  sys.argv[4] if len(sys.argv) > 4 else "40-epoch consensus+physics run"))
