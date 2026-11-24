# Exp 8 notes (PyBullet)
PyBullet could not be built with pip/uv on Windows (needs MSVC), so a conda-forge env was used:
    conda create -p .conda-xembody -c conda-forge python=3.11 pybullet numpy matplotlib scikit-learn pillow
Changed file: code/experiments/_common.py (name + timestamp log, SEED_OFFSET = 100*SEED) and the seed
lines of experiments 02-05 (seed = base + SEED_OFFSET + i).
    python experiments/00_check_setup.py
    python run_all.py
Exercises (outputs in exercises/):
    E1: python experiments/04_learn_once_deploy_many.py --source xarm6
    E2: python experiments/04_learn_once_deploy_many.py --demos 20
