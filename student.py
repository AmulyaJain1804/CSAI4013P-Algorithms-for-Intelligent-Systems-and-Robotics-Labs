"""Who ran this repository, plus a timestamped run log.

Every experiment script calls start_run() first. It prints the student's name
and the local time, appends the same line to run_log.txt, and stamp() writes
the name and time in small text on every figure that is saved.
"""
import datetime
import os
import platform

STUDENT = "Amulya Jain"
SAP_ID = "5001XXXXX"      # fill in your SAP ID
SEED = 11           # personal seed used by every stochastic experiment

ROOT = os.path.dirname(os.path.abspath(__file__))
_RUN = {}


def cpu_name():
    """Readable CPU name (Windows registry first, then platform)."""
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor() or platform.machine()


def start_run(exp):
    now = datetime.datetime.now().astimezone()
    _RUN.update(exp=exp, ts=now.strftime("%Y-%m-%d %H:%M:%S %Z"))
    line = (f"[{exp}] Run by {STUDENT} (SAP ID {SAP_ID}) | {_RUN['ts']} | "
            f"{platform.system()} {platform.release()} | {cpu_name()} | "
            f"Python {platform.python_version()} | seed {SEED}")
    print(line)
    with open(os.path.join(ROOT, "run_log.txt"), "a", encoding="utf8") as f:
        f.write(line + "\n")
    return now


def stamp(fig):
    """Write 'name | timestamp' in the bottom-right corner of a figure."""
    fig.text(0.995, 0.005, f"{STUDENT} | {_RUN.get('ts', '')}",
             ha="right", va="bottom", fontsize=6, color="0.45")
