import numpy as np
from scipy import integrate

def quadratic_quadrature(func):
    weight_1 = 1/6
    weight_2 = 2/3
    return func(weight_1, weight_1)*weight_1 + \
        func(weight_2, weight_1)*weight_1 + \
        func(weight_1, weight_2)*weight_1


def inner(s, t):
    return 1

if __name__ == "__main__":
    print("First:", quadratic_quadrature(inner))

    gfun = lambda x: 0
    hfun = lambda x: 1-x
    print("Second:", integrate.dblquad(inner, 0, 1, gfun, hfun)[0])