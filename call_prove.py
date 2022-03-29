import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')


def call_div(S, K, r, T, t, vol, q):
    d1 = (np.log(S / K) + (r - q + (vol ** 2) / 2) * (T - t)) / (vol * np.sqrt(T - t))
    d2 = d1 - vol * np.sqrt(T - t)
    F = S * np.exp((r - q) * (T - t))
    c = np.exp(-r * (T - t)) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return (c)


S = np.arange(0, 20);
K = 10;
r = -0.1;
T = 5;
t = T - 1;
vol = 0.4;
q = 0

p = np.zeros(len(S))
p_2 = np.zeros(len(S))

for i in range(len(S)):
    p[i] = max(0, S[i] - K)
    p_2[i] = call_div(S[i], K, r, T, t, vol, q)

fig, ax = plt.subplots(1, 1)
ax.plot(p, label='Payoff')
ax.plot(p_2, label='Option value')
plt.xlabel('S')
plt.ylabel('C(S,t)')
plt.legend()
plt.show()
print('blabla')
