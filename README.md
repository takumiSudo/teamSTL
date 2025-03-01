# STLGame
[Website](https://sites.google.com/view/stlgame), [Preprint](https://arxiv.org/pdf/2412.01656).

News: This paper is accepted by the 2025 Learning for Dynamics & Control Conference (L4DC'25).

## Install
The implementation has been tested with `Python 3.10` under `Mac M3`. We recommend installing the simulation inside a virtualenv. You can install the environment by running:

```bash
git clone git@github.com:shuoyang2000/STLgame.git
cd STLgame
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

## Training

The trained model is saved in the folder `models/`, and trajectory and exploitability data is saved in `data/`. To change the directories used, change `scripts/config.py` and `scripts/config_3d.py` for `kinematic` and `rotor` dynamics correspondingly.
To reproduce the training, please run (for instance, for autonomous drones case):
```bash
python3 scripts/main.py --dynamic rotor
```

It may take around 10 minutes for each FSP iteration, depending on your machines.

## Test Nash policy and best response against seen and unseen opponents

```bash
python3 scripts/test_nash.py --dynamic rotor
```

## RL Training for Best Response
```bash
python3 scripts/main_rlbr.py --dynamic rotor
```

## Citation and Contact
If you find this work useful, please consider citing:

```
@article{yang2024STLGame,
  title={STLGame: Signal Temporal Logic Games in Adversarial Multi-Agent Systems},
  author={Yang, Shuo and Zheng, Hongrui and Vasile, Cristian-Ioan and Pappas, George J and Mangharam, Rahul}
  journal={arXiv preprint arXiv:2412.01656},
  year={2024}
}
```

If you have any question on this repo, please feel free to contact the author Shuo Yang (yangs1 at seas dot upenn dot edu) or raise an issue.
