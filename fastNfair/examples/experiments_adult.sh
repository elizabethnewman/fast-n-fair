#!/bin/sh
export PYTHONPATH=".."

for r in $(seq 0.1 0.02 0.2);
do
  python ex_adult.py --save --robust --radius $r --robustOptimizer 'trust' --verbose
done

for r in $(seq 0.1 0.02 0.2);
do
  python ex_adult.py --save --robust --radius $r --robustOptimizer 'pgd' --verbose
done

for r in $(seq 0.1 0.02 0.2);
do
  python ex_adult.py --save --robust --radius $r --robustOptimizer 'rand' --verbose
done

python ex_adult.py --save --verbose