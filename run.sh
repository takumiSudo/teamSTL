#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd nSTL
python tests/main.py --dyn single_integrator --iters 60