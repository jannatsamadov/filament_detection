"""Fetch unlabelled GONG H-alpha images for self-supervised pretraining.

The 566 annotated photographs are the binding limit on this problem, and the
encoder currently starts from ImageNet weights -- a network trained on cats and
cars, which knows nothing about the Sun. GONG publishes its whole H-alpha archive
openly, so the encoder can be taught what a solar disk looks like before it is
asked to find filaments in one.

The competition allows external data for training; only inference is restricted
to the provided test images.

Sampling: a few images per day spread across the whole period, balanced over the
six stations that appear in the competition data. More than a couple per day is
waste -- it is the same Sun minutes apart. Dates matching a test image are
skipped, which the rules do not require but keeps the separation clean.

The archive is a public scientific service, so requests are paced.
"""
from __future__ import annotations

import argparse, random, re, sys, threading, time, urllib.error, urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "gong_unlabelled"
BASE = "https://gong2.nso.edu/HA/hag"
STATIONS = ("Bh", "Ch", "Lh", "Mh", "Th", "Uh")
UA = {"User-Agent": "filament-segmentation-2026 research fetch (contact via Kaggle)"}


def listing(day: date, retries=2, pause=1.0):
    url = f"{BASE}/{day:%Y%m}/{day:%Y%m%d}/"
    for k in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            html = urllib.request.urlopen(req, timeout=40).read().decode("utf-8", "ignore")
            return url, re.findall(r'href="([^"]+\.jpg)"', html)
        except Exception:
            if k == retries:
                return url, []
            time.sleep(pause * (k + 1))
    return url, []


def existing_dates():
    """Dates already present in the competition data, to leave the test set alone."""
    out = set()
    for split in ("train", "test"):
        d = ROOT / "MAGFiLO_1.0_Kaggle_2026" / split / f"{split}_images"
        if d.exists():
            for p in d.iterdir():
                out.add(p.stem[:8])
    return out


def fetch_day(day, want_set, per_day, pause, have, lock, counter):
    """One directory listing, then one image for each of several stations.

    The listing is the expensive part -- a day lists nearly 4000 files -- so it is
    amortised over several images rather than fetched again for each one.
    """
    url, files = listing(day)
    if not files:
        return 0
    by_station = defaultdict(list)
    for f in files:
        if len(f) > 16 and f[14:16] in want_set:
            by_station[f[14:16]].append(f)
    if not by_station:
        return 0

    got = 0
    with lock:
        order = sorted(by_station, key=lambda st: counter[st])
    for st in order[:per_day]:
        cand = sorted(by_station[st])
        f = cand[len(cand) // 2]          # middle of the day's run
        stem = f[:-4]
        with lock:
            if stem in have:
                continue
            have.add(stem)
        try:
            req = urllib.request.Request(url + f, headers=UA)
            data = urllib.request.urlopen(req, timeout=60).read()
        except Exception:
            with lock:
                have.discard(stem)
            continue
        if len(data) < 50_000:
            with lock:
                have.discard(stem)
            continue
        (OUT / f).write_bytes(data)
        with lock:
            counter[st] += 1
        got += 1
        time.sleep(pause)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3600)
    ap.add_argument("--start", default="2010-06-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--per-day", type=int, default=6)
    ap.add_argument("--workers", type=int, default=3,
                    help="parallel day fetches; keep small, this is a public archive")
    ap.add_argument("--pause", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    have = {p.stem for p in OUT.glob("*.jpg")}
    start_count = len(have)
    skip_dates = existing_dates()
    print(f"{start_count} already fetched, {len(skip_dates)} competition dates skipped")

    d0 = date.fromisoformat(args.start)
    d1 = date.fromisoformat(args.end)
    days = [d0 + timedelta(days=k) for k in range((d1 - d0).days + 1)]
    days = [d for d in days if f"{d:%Y%m%d}" not in skip_dates]
    random.Random(args.seed).shuffle(days)

    counter = Counter()
    for p in OUT.glob("*.jpg"):
        counter[p.stem[14:16]] += 1
    lock = threading.Lock()
    target_each = (args.n + start_count) // len(STATIONS)
    t0 = time.time()

    def job(day):
        with lock:
            want = {st for st in STATIONS if counter[st] < target_each}
            done = sum(counter.values())
        if not want or done >= args.n + start_count:
            return 0
        return fetch_day(day, want, args.per_day, args.pause, have, lock, counter)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for k, _ in enumerate(pool.map(job, days), 1):
            done = sum(counter.values())
            if done >= args.n + start_count:
                break
            if k % 40 == 0:
                el = time.time() - t0
                rate = (done - start_count) / max(el / 60, 1e-9)
                left = (args.n + start_count - done) / max(rate, 1e-9)
                print(f"  {done} images | {rate:.0f}/min | {left:.0f} min left | "
                      f"{dict(sorted(counter.items()))}", flush=True)

    n = len(list(OUT.glob("*.jpg")))
    size = sum(p.stat().st_size for p in OUT.glob("*.jpg")) / 1e9
    print(f"\n{n} images in {OUT} ({size:.2f} GB) in {(time.time()-t0)/60:.0f} min")
    print(f"per station: {dict(sorted(counter.items()))}")


if __name__ == "__main__":
    main()
