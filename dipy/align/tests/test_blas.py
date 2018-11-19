import numpy as np

from dipy.align.bundlemin import (_bundle_minimum_distance,
                                  _bundle_minimum_distance_blas)

np.random.seed(1234)
rows = 1000
num_s = 1
A = np.ones((rows, 3))
B = np.zeros((rows, 3))

from time import time

t1 = time()

for i in range(100000):
    res = _bundle_minimum_distance(A, B, num_s, num_s, rows)

t2 = time()

print(res, t2-t1)

t3 = time()

for i in range(100000):
    res2 = _bundle_minimum_distance_blas(A, B, num_s, num_s, rows)

t4 = time()
print(res2, t2-t1)
