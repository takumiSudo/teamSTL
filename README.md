# Continuous N-vs-M Team STL Game under TMECor via PSRO + STLCG

This repository contains the code and documentation for our research on solving continuous‐action, two‐team zero‐sum Signal Temporal Logic (STL) games using a Policy‐Space Response Oracles (PSRO) framework with a differentiable STL backbone (STLCG) to compute Team‐Maxmin Equilibria with ex‐ante Correlation (TMECor).

## 📄 Project Overview

Many real‐world multi‐agent scenarios (e.g. coordinated drone teams, robotic fleets) must satisfy complex temporal‐logic specifications in adversarial environments. Prior work (STLGame) used differentiable STL robustness to solve 2‐player zero‐sum games via generalized fictitious play. We extend this line of research to:
- **N vs. M team games** (arbitrary team sizes).  
- **TMECor** solution concept capturing correlated joint strategies.  
- **PSRO meta‐algorithm** for rapid convergence and richer strategy supports.  
- **STLCG‐based gradient oracles** for sample‐efficient best‐response computation.

## 🚀 Key Contributions

1. **Problem formulation**: Continuous‐state/action N‐agent vs. M‐agent STL game under TMECor.  
2. **Differentiable Oracle**: STLCG‐based best‐response via robustness gradient ascent.  
3. **PSRO pipeline**: Iterative restricted subgame construction, meta‐game LP solver, and strategy expansion (including Mix‐and‐Match).  
4. **Theoretical guarantees**: Convergence bounds for ε‐approximate oracles.  
5. **Empirical evaluation**: 1v1 through 4v4 benchmarks, ablation of oracles (gradient vs. PPO‐RL) and solvers (PSRO vs. GWFP).

## Install
The implementation has been tested with `Python 3.10` under `Mac M1`. We recommend installing the simulation inside a virtualenv. You can install the environment by running:

```bash
git clone https://github.com/takumiSudo/teamSTL.git
cd teamSTL
python3 -m venv STLGame_env
source STLGame_env/bin/activate
pip install -r requirements.txt
```
We use [stlcg](https://github.com/StanfordASL/stlcg) to compute the gradient of STL formula.

## Reproduce the results

#### Python path
We run all experiments from the project directory so please add the project directory to the PYTHONPATH environment variable:
```
export PYTHONPATH=$PYTHONPATH:$
```

#### Implementation
🚧 **Coming Soon**: Detailed implementation instructions and example scripts for running experiments. 

## Citation and Contact
Big thanks to the underlying authors of STLGame for providing the building blocks of this codebase.

```
@article{yang2024STLGame,
  title={STLGame: Signal Temporal Logic Games in Adversarial Multi-Agent Systems},
  author={Yang, Shuo and Zheng, Hongrui and Vasile, Cristian-Ioan and Pappas, George J and Mangharam, Rahul}
  journal={arXiv preprint arXiv:2412.01656},
  year={2024}
}
```

