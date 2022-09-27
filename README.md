# Contextual Bandits
For experimenting with different contextual bandit algorithms, includes multiple algorithms: Currently, this repo has
CoLin (Wu, Q. et al. "Contextual bandits in a collaborative environment." SIGIR, 2016), and 
FactorUCB (Wang, H. et al. "Factorization bandits for interactive recommendation." AAAI, 2017).

It will be working for the Yahoo dataset (https://webscope.sandbox.yahoo.com/catalog.php?datatype=r&did=49) and MovieLens 1M (https://grouplens.org/datasets/movielens/1m/)

This version also has a warm-start functionality (including stopping at a predecided time step and noting the relevant parameters) and uses a faster inverse method than the original algorithms.
This inverse (rank-one updates but faster) is due to https://timvieira.github.io/blog/post/2021/03/25/fast-rank-one-updates-to-matrix-inverse/ with the relevant code snippet living in https://github.com/timvieira/arsenal/blob/6d2eed1a94d0c19f3d2610039be67b5de26514aa/arsenal/maths/inv.py
