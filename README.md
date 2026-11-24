# CSAI4013P Algorithms for Intelligent Systems and Robotics - Lab (Amulya Jain, SAP ID 500122439)

CSAI4014 Algorithms for Intelligent Systems and Robotics - laboratory repository

| | |
|---|---|
| Name | Amulya Jain |
| SAP ID | 500122439 |
| Batch | 1 (B.Tech CSE AI & ML B1) |
| Partner(s) | None (individual report) |

Every script imports `student.py`, which prints my name and a timestamp at the start of each run and
appends the same line to `run_log.txt`; every figure carries a small "name | timestamp" stamp.
Stochastic parts use my personal seed (SEED = 11).

## How to run
```
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe numpy scipy matplotlib
.venv\Scripts\python exp01-signals\exp01.py      # likewise exp02 ... exp06
```
Exp 7 (Webots): see exp07-webots/NOTES.md. Exp 8 (PyBullet, conda-forge env): see exp08-xembody/NOTES.md.

## Sources and AI tools
- AISR Laboratory Manual (Exp 1-6); webots-robotics-sim README/GUIDE (Exp 7); AISR-Cross-Embodiment-PyBullet README/MANUAL (Exp 8)
- Oppenheim & Schafer; Russell & Norvig; Sutton & Barto; Goodfellow et al. (theory)
- Claude (Anthropic) AI assistant: help structuring scripts, timestamp logging, plots and report drafting. All results come from my own runs.
