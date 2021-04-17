# -*- coding: utf-8 -*-

from typing import Callable
import numpy
from scipy.linalg import solve_triangular

#%% find atomless pdf and cdf for one player

def gtilde(
    vi: Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray],
    ci: Callable[[numpy.ndarray], numpy.ndarray],
    b: float = 1,
    num: int = 1000,
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
    sgrid = numpy.linspace((b / num), b, num)
    # create a lower triangular matrix of vlaues
    vitril = numpy.tril(vi(sgrid[:, numpy.newaxis], sgrid))
    # find the PDF (/num) by solving the system of equations
    pdfi = solve_triangular(vitril, ci(sgrid), lower=True)
    # cumsum the PDF to get atomless CDF
    cdfi = numpy.cumsum(pdfi)
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

def eq2p(v: tuple, c: tuple, b: float = None, num: int = 1000) -> dict:
    """
    Calculate the equilibrium strategies for two players.
    v   : tuple of functions for the players' values which each take two arguments. The first argument is the player's own score and the second argument is the score of the opponent.
    ci  : tuple of functions for the players' costs with respect to their own scores.
    b   : optional float for the upper bound of the estimate. Heuristics will be used if not specified
    num : optional integer for the number of estimation points. A larger num is more accurate, but is slower
    """
    v1, v2 = v
    c1, c2 = c
    # if b is undefined, guess a prize of v(1,0)/c(1) or v(0,1)/c(1) with log cost
    if b is None:
        b = numpy.max(
            numpy.exp(
                [v1(1, 0) / c1(1), v2(1, 0) / c2(1),
                 v1(0, 1) / c1(1), v2(0, 1) / c2(1)]
            )
        )
    while True:
        player1 = gtilde(v1, c1, b, num)
        player2 = gtilde(v2, c2, b, num)
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
    # add the atom
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



