# -*- coding: utf-8 -*-

from typing import Callable
import numpy
from inteq import SolveVolterra

#%% find atomless pdf and cdf for one player


def gtilde(
    vi: Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray] = 1,
    ci: Callable[[numpy.ndarray], numpy.ndarray] = lambda x: x,
    b: float = 1,
    num: int = 1000,
    method: str = "midpoint",
) -> dict:
    """
    Calculate the atomless PDF and CDF for each player.
    vi  : function for the player's value that takes two arguments. The first argument is the player's own score and the second argument is the score of the opponent.
    ci  : function for the player's cost with respect to her own score.
    b   : optional float for the upper bound of the estimate (will fail if too low) defaults to 1
    num : optional integer for the number of estimation points. A larger num is more accurate, but is slower
    """
    if not isinstance(num, int):
        num = int(num)
    # make a grid of `num` points from (eps > 0) to `b`
    if callable(vi):
        sgrid, pdfi = SolveVolterra(k=vi, f=ci, a=0, b=b, num=num, method=method)
        # cumsum the PDF to get atomless CDF
        cdfi = numpy.cumsum(pdfi * (b/num))
    else:
        # presumably, it's a number of some sort
        # then we have an exact solution for CDF
        sgrid = numpy.linspace((b / num), b, num)
        cdfi = ci(sgrid) / vi
        # invert cumsum to get (scaled) PDF
        pdfi = numpy.insert(numpy.diff(cdfi), 0, cdfi[0])
    # find the index of sbar
    bari = numpy.argmax(sgrid[cdfi <= 1])
    return {
        "s": sgrid,
        "pdf": (pdfi * num),
        "cdf": cdfi,
        "sbari": bari,
        "sbar": sgrid[bari],
        "success": (cdfi[-1] > 1),
    }


#%% 2 player equilibrium function


def eq2p(
    v: tuple = (1.0,), c: tuple = (lambda x: x,), b: float = None, num: int = 1000
) -> dict:
    """
    Calculate the equilibrium strategies for two players.
    v   : tuple of functions or constants for the players' values which each take two arguments. The first argument is the player's own score and the second argument is the score of the opponent.
    c   : tuple of functions for the players' costs with respect to their own scores.
    b   : optional float for the upper bound of the estimate. Heuristics will be used if not specified
    num : optional integer for the number of estimation points. A larger num is more accurate, but is slower
    """

    # try to intelligently interpret v
    if isinstance(v, (Callable, int, float)):
        v1 = v
        v2 = v
    elif len(v) == 2:
        v1, v2 = v
    elif len(v) == 1:
        (v1,) = v
        (v2,) = v
    else:
        raise ValueError("v should be a tuple of length 1 or 2")

    # try to intelligently interpret c
    if callable(c):
        c1 = c
        c2 = c
    elif len(c) == 2:
        c1, c2 = c
    elif len(v) == 1:
        (c1,) = c
        (c2,) = c
    else:
        raise ValueError("c should be a tuple of length 1 or 2")

    # if b is undefined, make guess with fixed prize and linear cost
    if b is None:
        if callable(v1):
            b1 = v1(0, 0) / (c1(1 / num) * num)
        else:
            b1 = v1 / (c1(1 / num) * num)

        if callable(v2):
            b2 = v2(0, 0) / (c2(1 / num) * num)
        else:
            b2 = v2 / (c2(1 / num) * num)

        b = min(b1, b2)

    while True:
        player2 = gtilde(v1, c1, b, num)
        player1 = gtilde(v2, c2, b, num)
        success = any((player1["success"], player2["success"]))
        # if we succeed, then stop -- otherwise, double b and try again
        if success:
            break
        else:
            b = 2 * b
    # record the sbar index that we got from the above loop
    bari = min([player1["sbari"], player2["sbari"]])
    # cut the distributions off
    g = player1["pdf"][0 : (bari + 1)], player2["pdf"][0 : (bari + 1)]
    G1, G2 = player1["cdf"][0 : (bari + 1)], player2["cdf"][0 : (bari + 1)]
    # find the atoms
    G = (G1 - G1[-1] + 1), (G2 - G2[-1] + 1)
    # get the grid of s
    sgrid = player1["s"][0 : (bari + 1)]
    return {
        "s": sgrid,
        "pdf": g,
        "cdf": G,
        "sbari": bari,
        "sbar": sgrid[-1],
        "b": b,
    }


# %%
