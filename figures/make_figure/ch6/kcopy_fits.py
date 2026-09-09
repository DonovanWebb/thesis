import numpy as np
from numpy import kron
from numpy.linalg import eig, solve
from scipy.optimize import curve_fit


def decay_model(x, e_corr, e_L, e_R, spam):
    """
    Correlated rotations model for two-qubit RB.
    """
    atr = 1 - e_L - e_R
    adt = 1 - e_L - e_R - 3 * e_corr
    aL = 1 - e_L - e_corr
    aR = 1 - e_R - e_corr
    sp = 1 - 2 * spam
    S1 = sp * aL**x
    S2 = sp * aR**x
    M1 = 2 * sp**2 * adt**x / 3
    M2 = sp**2 * atr**x / 3
    P00 = (1 + S1 + S2 + M1 + M2) / 4
    P01 = (1 + S1 - S2 - M1 - M2) / 4
    P10 = (1 - S1 + S2 - M1 - M2) / 4
    P11 = (1 - S1 - S2 + M1 + M2) / 4
    P = np.concatenate([P00, P01, P10, P11])
    return P


def fit_decays(num_gates, p_00, p_01, p_10, p_11):
    """
    Fit the decay data (p_00, p_01, p_10, p_11) vs num_gates to extract e_corr and e_L, e_R, and stateprep error, e_sp.
    Returns popt, perr, and the full fit result object.
    """
    # prepare data
    num_gates = np.asarray(num_gates)
    ydata = np.concatenate(
        [np.asarray(p_00), np.asarray(p_01), np.asarray(p_10), np.asarray(p_11)]
    )

    # initial guess and bounds
    p0 = [0.01, 0.01, 0.01, 0.01]  # initial guess for [e_corr, e_L, e_R, e_sp]
    lower_bounds = [0.0001, 0.0001, 0.0001, 0.0001]
    upper_bounds = [0.1, 0.1, 0.1, 0.1]

    popt, pcov = curve_fit(
        lambda t, e_corr, e_L, e_R, e_sp: decay_model(t, e_corr, e_L, e_R, e_sp),
        num_gates,
        ydata,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=10000,
    )
    perr = np.sqrt(np.diag(pcov))

    residuals = ydata - decay_model(num_gates, *popt)
    rss = float(np.sum(residuals**2))

    # return popt, perr and some diagnostics
    return {
        "popt": popt,
        "perr": perr,
        "rss": rss,
    }


# ===== Below is PTM method for generating decay curves ================

# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def pauli_transfer_matrix4(m: np.ndarray) -> np.ndarray:
    """
    Convert a 4x4 matrix `m` to its 16x16 Pauli transfer matrix representation.
    """
    paulis = [
        kron(I, I),
        kron(I, X),
        kron(I, Y),
        kron(I, Z),
        kron(X, I),
        kron(Y, I),
        kron(Z, I),
        kron(X, X),
        kron(X, Y),
        kron(X, Z),
        kron(Y, X),
        kron(Y, Y),
        kron(Y, Z),
        kron(Z, X),
        kron(Z, Y),
        kron(Z, Z),
    ]
    ptm = np.zeros((16, 16), dtype=complex)
    m_dag = m.conj().T
    for i in range(16):
        for j in range(16):
            ptm[i, j] = 0.25 * np.trace(paulis[i] @ m @ paulis[j] @ m_dag)
    return ptm


def construct_projectors():
    """
    Returns r00, r11, r01, r10 as length-16 Pauli-basis state vectors (numpy arrays).
    """
    r00 = np.zeros(16, dtype=complex)
    # Julia used 1/4 placed at indices 1,4,7,16 (1-based). Convert to 0-based: 0,3,6,15
    r00[0] = 1 / 4
    r00[3] = 1 / 4
    r00[6] = 1 / 4
    r00[15] = 1 / 4
    r00 = 2 * r00

    r11 = np.zeros(16, dtype=complex)
    r11[0] = 1 / 4
    r11[3] = -1 / 4
    r11[6] = -1 / 4
    r11[15] = 1 / 4
    r11 = 2 * r11

    r01 = np.zeros(16, dtype=complex)
    r01[0] = 1 / 4
    r01[3] = 1 / 4
    r01[6] = -1 / 4
    r01[15] = -1 / 4
    r01 = 2 * r01

    r10 = np.zeros(16, dtype=complex)
    r10[0] = 1 / 4
    r10[3] = -1 / 4
    r10[6] = 1 / 4
    r10[15] = -1 / 4
    r10 = 2 * r10

    return r00, r11, r01, r10


def construct_ptm(theta_g: float, theta_l: float) -> np.ndarray:
    """
    Construct a 16x16 two-qubit PTM M = G * L (global then local error),
    """
    G = np.zeros((16, 16), dtype=float)
    L = np.zeros((16, 16), dtype=float)

    offset = 7  # keeps the same indexing logic as the Julia code (0-based offset)
    s1_g = 1.0 - theta_g**2 / 3.0
    s2 = 1.0 - 2.0 * theta_g**2 / 3.0
    sm = theta_g**2 / 3.0
    s1_l = 1.0 - theta_l**2 / 3.0
    s2_l = 1.0 - 2.0 * theta_l**2 / 3.0

    # Global errors: G[0,0] = 1 and G[1..6,1..6] = s1_g (1-based Julia: 1: index 0)
    G[0, 0] = 1.0
    for i in range(1, 7):  # indices 1..6 (0-based)
        G[i, i] = s1_g

    # I9 block: add s2 to G[i+offset, i+offset] for i=0..8  (Julia's 1:9 -> 0..8)
    for i in range(0, 9):
        G[i + offset, i + offset] += s2

    # Mm block: indices (converted from Julia's 1-based tuples to 0-based)
    indices_p = [(4, 0), (8, 0), (0, 4), (0, 8), (8, 4), (4, 8)]
    indices_m = [(3, 1), (6, 2), (1, 3), (2, 6), (7, 5), (5, 7)]

    for i, j in indices_p:
        G[i + offset, j + offset] += sm
    for i, j in indices_m:
        G[i + offset, j + offset] += -sm

    # Local errors: L[0,0] = 1 and L[1..6,1..6] += s1_l
    L[0, 0] = 1.0
    for i in range(1, 7):
        L[i, i] += s1_l

    for i in range(0, 9):
        L[i + offset, i + offset] += s2_l

    M = G @ L
    return M


def ptm_decay_model(t_array: np.ndarray, theta_g: float, theta_l: float) -> np.ndarray:
    # build PTM and eigen-decompose
    ptm = construct_ptm(theta_g, theta_l)
    eigvals, eigvecs = eig(ptm)  # eigvecs columns are eigenvectors

    # projectors in pauli basis
    r00, r11, r01, r10 = construct_projectors()

    # compute coefficients c0 in eigenbasis: solve eigvecs * c0 = r00  -> c0 = eigvecs^{-1} r00
    # Note: eigvecs is complex in general
    c0 = solve(eigvecs, r00)

    t_array = np.asarray(t_array)
    n = len(t_array)

    overlaps00 = np.zeros(n, dtype=float)
    overlaps11 = np.zeros(n, dtype=float)
    overlaps0110 = np.zeros(n, dtype=float)

    # iterate steps
    for j, step in enumerate(t_array):
        # eigenvalues^step (elementwise), then scale coefficients
        ck = (eigvals**step) * c0
        at = eigvecs @ ck  # back to original basis

        overlaps00[j] = float(np.real(np.vdot(r00, at)))
        overlaps11[j] = float(np.real(np.vdot(r11, at)))
        overlaps0110[j] = float(np.real(np.vdot(r01, at) + np.vdot(r10, at)))

    return np.concatenate([overlaps00, overlaps11, overlaps0110])
