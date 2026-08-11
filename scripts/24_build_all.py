"""Run 10_build_lines.py until it finishes, surviving renderer crashes.

Verovio can die on a piece with an access violation (0xC0000005). Python cannot
catch that -- the process is gone -- so a build that has been running for hours
loses everything after the fall. The pieces that do it have been narrowed down
and the converter no longer produces the construction that caused the known
case (`synth.pdmx`, restricted duration vocabulary), but "no more crashes" is
not something that can be proven in advance over 3730 pieces.

So: run the build, and if it falls, read the piece name it left in
`work/10_current.txt`, add it to `work/10_quarantine.json`, and start again.
`--skip-existing` makes the restart cost only the pieces not yet built. Every
quarantined piece is written down with the run it died in, so the report can
say which pieces were dropped for this reason and how many.

Writes work/10_quarantine.json and work/24_build_all.json.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "work"
CURRENT = WORK / "10_current.txt"
QUARANTINE = WORK / "10_quarantine.json"
BUILD = REPO / "scripts" / "10_build_lines.py"

MAX_RESTARTS = 60      # a build that falls this often has a different problem


def load_quarantine() -> list[dict]:
    if QUARANTINE.exists():
        raw = json.loads(QUARANTINE.read_text(encoding="utf-8"))
        return [q if isinstance(q, dict) else {"piece": q, "run": None}
                for q in raw]
    return []


def main():
    passthrough = sys.argv[1:] or ["3730", "--source=pdmx", "--skip-existing"]
    if "--skip-existing" not in passthrough:
        passthrough.append("--skip-existing")

    quarantine = load_quarantine()
    runs, t0 = [], time.time()

    for attempt in range(MAX_RESTARTS):
        CURRENT.unlink(missing_ok=True)
        started = time.time()
        log = WORK / f"24_run{attempt:02d}.log"
        with log.open("w", encoding="utf-8") as fh:
            r = subprocess.run([sys.executable, str(BUILD), *passthrough],
                               stdout=fh, stderr=subprocess.STDOUT,
                               cwd=str(REPO))
        minutes = round((time.time() - started) / 60, 1)
        built = sum(1 for d in (WORK / "lines").iterdir() if d.is_dir())
        runs.append({"attempt": attempt, "exit": r.returncode,
                     "minutes": minutes, "pieces_on_disk": built,
                     "log": log.name})
        print(f"run {attempt}: exit {r.returncode}, {minutes} min, "
              f"{built} pieces on disk", flush=True)

        if r.returncode == 0:
            break

        victim = CURRENT.read_text(encoding="utf-8").strip() if CURRENT.exists() else ""
        if not victim:
            print("crashed without naming a piece -- stopping", flush=True)
            break
        if any(q["piece"] == victim for q in quarantine):
            print(f"{victim} is already quarantined and it still fell -- "
                  f"stopping rather than looping", flush=True)
            break
        quarantine.append({"piece": victim, "run": attempt,
                           "exit": r.returncode})
        QUARANTINE.write_text(json.dumps(quarantine, indent=1), encoding="utf-8")
        print(f"  quarantined {victim}", flush=True)

    out = {"args": passthrough, "runs": runs,
           "restarts": max(0, len(runs) - 1),
           "quarantined": quarantine,
           "finished": bool(runs) and runs[-1]["exit"] == 0,
           "minutes_total": round((time.time() - t0) / 60, 1)}
    (WORK / "24_build_all.json").write_text(json.dumps(out, indent=2),
                                            encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
