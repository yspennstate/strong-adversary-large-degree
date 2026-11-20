"""Exact checks for the complete-graph example.

1. For small n, the stationary mass of v under the kernel Q = (1-eps)P + eps*B on K_n
   (simple random walk P, B sends every x != v to v, B(v,.) = P(v,.)) is computed from
   the full n-state kernel by Gaussian elimination over the rationals and compared with
   the closed form a/(1+a), a = eps + (1-eps)/(n-1).
2. For n = 65537 and eps = 2^-11 it prints a/(1+a), its ratio to pi(v) = 1/n, and the
   right-hand side of Zuckerman's bound with t_up = 1, d = n-1, S = {v}.
3. It computes exactly the total variation mixing time of the simple random walk on the rook's
   graph K_q x K_q for 3 <= q <= 1000.
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


# 3. Mixing time of the simple random walk on the rook's graph K_q x K_q (Lemma on the rook's graph).
# A step picks one of the two coordinates with probability 1/2 and moves it to a uniform other value.
# Started at (0,0), the law at time t is constant on four classes of pairs (both coordinates at the
# start value, only the first, only the second, neither), so the total variation distance to the
# uniform law is computed exactly from four numbers.
def rook_tv(q, t):
    def at_start(m):  # simple random walk on K_q: probability of the start value after m steps
        return Fr(1, q) + (1 - Fr(1, q)) * Fr(-1, q - 1) ** m
    acc = [Fr(0)] * 4
    for j in range(t + 1):
        w = Fr(math.comb(t, j), 2 ** t)
        a0, b0 = at_start(j), at_start(t - j)
        a1, b1 = (1 - a0) / (q - 1), (1 - b0) / (q - 1)
        for i, x in enumerate((a0 * b0, a0 * b1, a1 * b0, a1 * b1)):
            acc[i] += w * x
    u = Fr(1, q * q)
    sizes = (1, q - 1, q - 1, (q - 1) ** 2)
    return sum(s * abs(x - u) for s, x in zip(sizes, acc)) / 2


def rook_tau(q):
    t = 1
    while rook_tv(q, t) > Fr(1, 4):
        t += 1
    return t


taus = {q: rook_tau(q) for q in range(3, 1001)}
assert all(rook_tv(q, 3) <= Fr(1, 4) for q in range(3, 16))
assert all(taus[q] == 2 for q in range(3, 8))
assert all(taus[q] == 3 for q in range(8, 1001))
print("rook's graph: tau = 2 for 3 <= q <= 7, tau = 3 for 8 <= q <= 1000; distance <= 1/4 at t = 3 for 3 <= q <= 15")
