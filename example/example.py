# -*- coding: utf-8 -*-

from allpy import eq2p
import numpy as np
import matplotlib.pyplot as plt

# define functions
def v(si, sj):
    return 9 / 10 + np.exp(-30 * (si + sj)) / (np.exp(-40) + np.exp(-30 * (si + sj)))

def c1(si):
    return si ** 2

def c2(si):
    return si

# use the package
eq = eq2p(v=(v, v), c=(c1, c2), b=1)

# get objects to plot
S = eq["s"]
Sbar = eq["sbar"]
G1, G2 = eq["cdf"]


# plot CDFs
plt.plot(S, G1)
plt.plot(S, G2)

plt.legend(["$G_1$", "$G_2$"])

plt.xlabel("Score")

plt.ylim(0, 1)
plt.xlim(0, Sbar)

plt.show()