#!/bin/sh
export PYTHONPATH=".."

for a in $(seq 0 0.02 1);
do
  python ex_color_mnist_binary.py --save --verbose --alpha $a
done

for a in $(seq 0 0.02 1);
do
  python ex_color_mnist_binary.py --save --verbose --robust --alpha $a
done

for p_test in $(seq 0.1 0.05 0.95)
do
    python ex_color_mnist_binary.py --save --verbose --robust --p_test $p_test
done

#for r in $(seq 0.1 0.02 0.2);
#do
#  python ex_adult.py --save --robust --radius $r --robustOptimizer 'pgd' --verbose
#done
#
#for r in $(seq 0.1 0.02 0.2);
#do
#  python ex_adult.py --save --robust --radius $r --robustOptimizer 'rand' --verbose
#done
#
#python ex_adult.py --save --verbose