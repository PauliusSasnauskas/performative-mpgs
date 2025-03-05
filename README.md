# Independent Learning in Performative Markov Potential Games
Developed during the internship of Rilind Sahitaj.
Theory & algorithms developed by Rilind Sahitaj, and Yiğit Yalin, code by Paulius Sasnauskas, supervision by Debmalya Mandal and Goran Radanović.
Many thanks to Ben Rank for assistance with implementation details and Stelios Triantafyllou for the original code.


## Installation
Install `requirements.txt`.
For efficient use of GPUs (read: orders of magnitude speedup, JIT compilation with JAX), [install JAX for the GPU](https://jax.readthedocs.io/en/latest/installation.html):
```bash
pip install -U "jax[cuda12_pip]==0.4.30" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
```

## Experiments
Example usage:

```bash
# Safe-distancing game:
# IPGA-L:
python dist_main_pga.py
# IPGA-D:
python dist_main_pga.py --ding
# INPG (unreg.):
python dist_main_inpg.py
# INPG (reg.):
python dist_main_inpg.py --log_barrier_reg 0.003

# Stochastic congestion game:
# IPGA-L:
python cong_main_pga.py
# IPGA-D:
python cong_main_pga.py --ding
# INPG (unreg.):
python cong_main_inpg.py
# INPG (reg.):
python cong_main_inpg.py --log_barrier_reg 0.003
```

To get plots in Figures 1 and 2:
```bash
# Figure 1 left
python dist_main_pga.py --n_experiment_replications 10 --n_rounds 30000 --lr 0.00001 --gamma 0.99 --alpha 0.15
python dist_main_pga.py --n_experiment_replications 10 --n_rounds 30000 --lr 0.00001 --gamma 0.99 --alpha 0.01
python dist_main_pga.py --ding --n_experiment_replications 10 --n_rounds 30000 --lr 0.0001 --gamma 0.99 --alpha 0.15
python dist_main_pga.py --ding --n_experiment_replications 10 --n_rounds 30000 --lr 0.0001 --gamma 0.99 --alpha 0.01

python plot_comparison.py \
    --env distancing --n_experiment_replications 10 --n_rounds 30000 --gamma 0.99 \
    --compare leo_lr0.00001_alpha0.15,leo_lr0.00001_alpha0.01,ding_lr0.0001_alpha0.15,ding_lr0.0001_alpha0.01 \
    --colors "#d62728,#fa6f1e,#1145bf,#2cbae6" \
    --labels "IPGA-L alpha=0.15 lr=1e-5,IPGA-L alpha=0.01 lr=1e-5,IPGA-D alpha=0.15 lr=1e-4,IPGA-D alpha=0.01 lr=1e-4" \
    --out comp1l

# Figure 1 right
python cong_main_pga.py --n_experiment_replications 5 --n_rounds 60000 --lr 0.00003 --gamma 0.99 --omega_r 0.1 --omega_p 0.1
python cong_main_pga.py --n_experiment_replications 5 --n_rounds 60000 --lr 0.00003 --gamma 0.99 --omega_r 0.03 --omega_p 0.03
python cong_main_pga.py --ding --n_experiment_replications 5 --n_rounds 60000 --lr 0.0003 --gamma 0.99 --omega_r 0.1 --omega_p 0.1
python cong_main_pga.py --ding --n_experiment_replications 5 --n_rounds 60000 --lr 0.0003 --gamma 0.99 --omega_r 0.03 --omega_p 0.03

python plot_comparison.py \
    --env congestion2 --n_experiment_replications 5 --n_rounds 60000 --gamma 0.99 \
    --compare leo_lr0.0003_omegar0.1_omegap0.1,leo_lr0.0003_omegar0.03_omegap0.03,ding_lr0.0003_omegar0.1_omegap0.1,ding_lr0.0003_omegar0.03_omegap0.03 \
    --colors "#d62728,#fa6f1e,#1145bf,#2cbae6" \
    --labels "IPGA-L omega=0.1,IPGA-L omega=0.03,IPGA-D omega=0.1,IPGA-D omega=0.03" \
    --out comp1r

# Figure 2 left
python dist_main_pga.py --ding --n_experiment_replications 10 --n_rounds 10000 --lr 0.0001 --gamma 0.99 --alpha 0.01
python dist_main_inpg.py --n_experiment_replications 10 --n_rounds 10000 --lr 0.0001 --gamma 0.99 --alpha 0.01
python dist_main_inpg.py --n_experiment_replications 10 --n_rounds 10000 --lr 0.0001 --gamma 0.99 --alpha 0.01 --log_barrier_reg 0.003

python plot_comparison.py \
    --env distancing --n_experiment_replications 10 --n_rounds 10000 --gamma 0.99 \
    --compare ding_lr0.0001_alpha0.01,inpg_lr0.0001_alpha0.01,inpg_reg0.003_lr0.0001_alpha0.01 \
    --labels "IPGA-D,INPG (unreg.),INPG (reg.)" \
    --colors "#1145bf,#014a2a,#83d941" \
    --out comp2l

# Figure 2 right
python cong_main_pga.py --ding --n_experiment_replications 5 --n_rounds 20000 --lr 0.0006 --gamma 0.99 --omega_r 0.03 --omega_p 0.03
python cong_main_inpg.py --n_experiment_replications 5 --n_rounds 20000 --lr 0.0006 --gamma 0.99 --omega_r 0.03 --omega_p 0.03
python cong_main_inpg.py --n_experiment_replications 5 --n_rounds 20000 --lr 0.0006 --gamma 0.99 --omega_r 0.03 --omega_p 0.03 --log_barrier_reg 0.003

python plot_comparison.py \
    --env congestion2 --n_experiment_replications 5 --n_rounds 20000 --gamma 0.99 \
    --compare ding_lr0.0006_omegar0.03_omegap0.03,inpg_lr0.0006_omegar0.03_omegap0.03,inpg_reg0.003_lr0.0006_omegar0.03_omegap0.03 \
    --colors "#1145bf,#014a2a,#83d941" \
    --labels "IPGA-D,INPG (unreg.),INPG (reg.)" \
    --out comp2r
```

(add `--logging wandb` to track experiments in Weights & Biases)


## Runtime
One retraining round with 10 experiment replications in parallel on the safe-distancing game on an NVIDIA A100 GPU takes approx. 0.7 seconds (first round takes approx. 12 s for JIT compilation). Runtime for 10000 rounds: approx. 2 hours.
One retraining round with 5 experiment replications in parallel on the stochastic congestion game on an NVIDIA A100 GPU takes approx. 1.6 seconds (first round takes approx. 2 min for JIT compilation). Runtime for 10000 rounds: approx. 4.5 hours.

## Code

The code is generally not object-oriented for easier composition with JAX's `vmap`, `jit` functions (for easy vectorization and JIT compilation).

## Tests
```bash
pip install pytest ddt
python -m pytest -v -s ./test
```

## Citing
To appear in AISTATS 2025.
