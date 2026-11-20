"""Exact checks for the complete-graph example.

1. For small n, the stationary mass of v under the kernel Q = (1-eps)P + eps*B on K_n
   (simple random walk P, B sends every x != v to v, B(v,.) = P(v,.)) is computed from
   the full n-state kernel by Gaussian elimination over the rationals and compared with
   the closed form a/(1+a), a = eps + (1-eps)/(n-1).
2. For n = 65537 and eps = 2^-11 it prints a/(1+a), its ratio to pi(v) = 1/n, and the
   right-hand side of Zuckerman's bound with t_up = 1, d = n-1, S = {v}.
"""
from fractions import Fraction as Fr
import math


def stationary(Q):
    n = len(Q)
    # solve nu (Q - I) = 0, sum nu = 1: rows of the system are columns of Q - I
    A = [[Q[j][i] - (1 if i == j else 0) for j in range(n)] for i in range(n)]
    A[-1] = [Fr(1)] * n
    b = [Fr(0)] * (n - 1) + [Fr(1)]
    for c in range(n):
        p = next(r for r in range(c, n) if A[r][c] != 0)
        A[c], A[p] = A[p], A[c]
        b[c], b[p] = b[p], b[c]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c] / A[c][c]
                A[r] = [x - f * y for x, y in zip(A[r], A[c])]
                b[r] -= f * b[c]
    return [b[i] / A[i][i] for i in range(n)]


def kernel(n, eps, v=0):
    Q = []
    for x in range(n):
        P = [Fr(0) if y == x else Fr(1, n - 1) for y in range(n)]
        B = P[:] if x == v else [Fr(1) if y == v else Fr(0) for y in range(n)]
        Q.append([(1 - eps) * P[y] + eps * B[y] for y in range(n)])
    return Q


for n in (5, 9, 17):
    eps = Fr(1, 3)
    nu = stationary(kernel(n, eps))
    a = eps + (1 - eps) / (n - 1)
    assert nu[0] == a / (1 + a), (n, nu[0], a / (1 + a))
    assert sum(nu) == 1 and min(nu) > 0
    print("n=%d eps=1/3: full-kernel nu(v) = %s = a/(1+a)" % (n, nu[0]))

n = 2 ** 16 + 1
d = n - 1
eps = Fr(1, 2 ** 11)
a = eps + (1 - eps) / d
mass = a / (1 + a)
print("a =", a, " a/(1+a) =", mass, " ratio to 1/n =", float(mass * n))
expo = 2 * float(eps) * math.sqrt(2 * d * math.log(n))
print("exponent 2 eps sqrt(2 d ln n) =", expo)
for alpha in (Fr(1, d), Fr(1, 4), Fr(1, 2), Fr(999, 1000)):
    bound = (1 + float(alpha)) * math.exp(expo) / n
    print("alpha=%s: bound*n = %.4f, mass/bound = %.3f" % (alpha, bound * n, float(mass) / bound))
print("sup over alpha<1 of bound*n <", 2 * math.exp(expo), "; mass/sup >", float(mass) * n / (2 * math.exp(expo)))
k_eps_d = math.sqrt(d * math.log(d) / 2)
print("k eps d at t=1, p=1/(n-1):", k_eps_d)
print("Cor 3.5 at t=1: bound*d =", math.exp(2 * float(eps) * math.sqrt(2 * d * math.log(d))), " q_1*d =", float(a * d))
