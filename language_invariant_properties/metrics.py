from collections import Counter
from scipy.stats import entropy
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder

def get_kl(a, b):
    """
    Computes the KL divergence
    """
    epsilon = 0.00000001
    c1 = Counter(a)
    total1 = sum(c1.values()) + len(c1) * epsilon
    c1c = [(c1[cat] + epsilon) / total1 for cat in sorted(c1.keys())]

    c2 = Counter(b)
    total2 = sum(c2.values()) + len(c1) * epsilon
    c2c = [(c2[cat] + epsilon) / total2 for cat in sorted(c1.keys())]

    return '%.3f' % entropy(c1c, c2c)

def get_significance(P, Q):
    """
    Computes the significance using the X^2
    """
    le = LabelEncoder()

    P = le.fit_transform(P)
    Q = le.transform(Q)

    A = Counter(P)
    B = Counter(Q)

    a = np.zeros((2, len(A)), dtype=int)

    for key, value in sorted(A.items()):
        a[0, key] = value
    for key, value in sorted(B.items()):
        a[1, key] = value

    g, p, dof, E = chi2_contingency(a)
    significance = ''
    if p <= 0.01:
        significance = '$^{**}$'
    elif p <= 0.05:
        significance = '$^{*}$'
    return significance
