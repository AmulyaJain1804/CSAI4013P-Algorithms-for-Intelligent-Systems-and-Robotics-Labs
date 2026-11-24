# Exp 7 notes (Webots)
Course code from github.com/reshamrajs/webots-robotics-sim, controllers unchanged.
Commands (run from exp07-webots/code, every setting repeated 3 times with --name ..._r1/_r2/_r3):
    python tools/run_experiment.py swarm --seconds 120 --name swarm_default_r1
    python tools/run_experiment.py swarm --seconds 120 --set cruise=0.9 --name swarm_cruise-0.9_r1
    python tools/run_experiment.py swarm --seconds 120 --set cruise=0.2 --name swarm_cruise-0.2_r1
    python tools/run_experiment.py pioneer --seconds 180 --name pioneer_default
    python tools/run_experiment.py pioneer --seconds 180 --set bubble=0.15 --name pioneer_bubble-0.15
Figures: python ../../tools/exp07_figs.py <repo>  ->  exp07-webots/figs/

Changed file: code/tools/run_experiment.py (Windows fixes)
- world copy written with LF line endings (newline=""), log path with forward slashes
- re.sub replacement given as a lambda (backslashes in Windows paths broke the regex)
- PYTHONHOME / UV_INTERNAL__PYTHONHOME removed from Webots' environment: the uv venv sets them
  and Webots crashed (0xC0000005) on start-up

Changed file: code/controllers/epuck_braitenberg/epuck_braitenberg.py - new "seed" parameter for the wheel
noise. Plain repeats were bit-identical (see deterministic_runs.txt), so Table 7.1 uses
--set noise=0.02 seed=<personal seed*100 + repeat> (script: tools/run_webots_noise.sh).
