"""Generate a controlled CPU load for the benchmark-variance experiment (EXP-009): N busy processes for S seconds, then they exit by themselves.

  python tools/cpu_burn.py --procs 6 --seconds 900

Purpose: show how much the absolute FPS of this CPU-bound pipeline depends on what else runs on the machine, and that the ratio to the
v0 prototype measured in the same round is much more stable. It is NOT a model of what actually happened in the earlier slow sessions.
"""
import argparse
import multiprocessing as mp
import time


def burn(seconds):
    end = time.time() + seconds
    x = 0
    while time.time() < end:
        for _ in range(200000):
            x = (x * 1664525 + 1013904223) % 4294967296


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=600)
    a = ap.parse_args()
    ps = [mp.Process(target=burn, args=(a.seconds,)) for _ in range(a.procs)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()


if __name__ == "__main__":
    main()
