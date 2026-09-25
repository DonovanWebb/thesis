"""
Helper functions for figure scripts.
"""

import numpy as np
from scipy.optimize import curve_fit
from oitg.results import load_result
import pandas as pd
from statsmodels.stats.proportion import proportion_confint


def load_data(rid, day):
    f = load_result(rid=rid, day=day, experiment="fastgates")
    return f


def str_to_cliff(seq_str):
    match_letter = "b"
    count = seq_str.decode("utf-8").count(match_letter)
    return count


def get_rb_data(rid, date, sim_outcomes):
    data_dict = load_data(rid, date)
    run_order = data_dict["datasets"]["data.circuits.run_order"]
    is_brights = data_dict["datasets"]["is_brights"]
    sequences = data_dict["datasets"]["data.circuits.sequences"]
    pt_cliffs = np.array([str_to_cliff(seq) for seq in sequences])[run_order]
    expected_outcomes = np.array(sim_outcomes)[run_order]

    return is_brights, pt_cliffs, expected_outcomes


def errorbars(data):
    k = np.sum(data)
    N = len(data)
    # Calculate error bars using Clopper-Pearson method
    confint = proportion_confint(k, N, alpha=0.3173, method="beta")
    if np.isnan(confint[0]):
        confint = (0, confint[1])
    elif np.isnan(confint[1]):
        confint = (confint[0], 1)
    p = k / N
    uncertainty = (confint[1] - confint[0]) / 2
    return p, uncertainty


def theta_to_error(theta, theta_err):
    p = 1 - (1 + 2 * np.cos(theta)) / 3
    p_err = (2 / 3) * np.sin(theta) * theta_err
    return p, p_err


def two_qubit_pops(is_brights, cliff_lens, expected_outcomes):
    """
    Get correlated populations for two-qubit RB data.
    This function analyzes the shot-by-shot outcomes to determine the probabilities
    of being in each of the four two-qubit states (00, 01, 10, 11) as a function of Clifford length.
    """

    # Analyze two-qubit correlations shot by shot
    OO_all = np.array([])
    OX_all = np.array([])
    XO_all = np.array([])
    XX_all = np.array([])
    OO_all_sd = np.array([])
    OX_all_sd = np.array([])
    XO_all_sd = np.array([])
    XX_all_sd = np.array([])

    for i, m in enumerate(is_brights):
        OO_shots = []
        OX_shots = []
        XO_shots = []
        XX_shots = []

        for shot in m:
            OO_shots.append(
                shot[0] != expected_outcomes[i] and shot[1] != expected_outcomes[i]
            )
            OX_shots.append(
                shot[0] != expected_outcomes[i] and shot[1] == expected_outcomes[i]
            )
            XO_shots.append(
                shot[0] == expected_outcomes[i] and shot[1] != expected_outcomes[i]
            )
            XX_shots.append(
                shot[0] == expected_outcomes[i] and shot[1] == expected_outcomes[i]
            )

        OO_all = np.append(OO_all, np.mean(OO_shots))
        OX_all = np.append(OX_all, np.mean(OX_shots))
        XO_all = np.append(XO_all, np.mean(XO_shots))
        XX_all = np.append(XX_all, np.mean(XX_shots))
        OO_all_sd = np.append(OO_all_sd, errorbars(OO_shots)[1])
        OX_all_sd = np.append(OX_all_sd, errorbars(OX_shots)[1])
        XO_all_sd = np.append(XO_all_sd, errorbars(XO_shots)[1])
        XX_all_sd = np.append(XX_all_sd, errorbars(XX_shots)[1])

    # Find population in each state
    OO_means = []
    OO_errs = []
    OX_means = []
    OX_errs = []
    XO_means = []
    XO_errs = []
    XX_means = []
    XX_errs = []

    seq_uniq = np.unique(cliff_lens)

    for l in seq_uniq:
        OO_subset = OO_all[cliff_lens == l]
        OO_means.append(np.mean(OO_subset))
        OO_errs.append(np.std(OO_subset) / np.sqrt(len(OO_subset)))

        OX_subset = OX_all[cliff_lens == l]
        OX_means.append(np.mean(OX_subset))
        OX_errs.append(np.std(OX_subset) / np.sqrt(len(OX_subset)))

        XO_subset = XO_all[cliff_lens == l]
        XO_means.append(np.mean(XO_subset))
        XO_errs.append(np.std(XO_subset) / np.sqrt(len(XO_subset)))

        XX_subset = XX_all[cliff_lens == l]
        XX_means.append(np.mean(XX_subset))
        XX_errs.append(np.std(XX_subset) / np.sqrt(len(XX_subset)))

    df = pd.DataFrame(
        {
            "Clifford Length": seq_uniq,
            "P00": OO_means,
            "P00 Std": OO_errs,
            "P01": OX_means,
            "P01 Std": OX_errs,
            "P10": XO_means,
            "P10 Std": XO_errs,
            "P11": XX_means,
            "P11 Std": XX_errs,
        }
    )
    # df.to_csv(f"sequences/two_qubit_populations_{rid}.csv", index=False)
    return df


class ErrModelBase:
    name = "model"
    p1_name = "param1"
    p2_name = "param2"
    p3_name = "param3"
    p4_name = "param4"
    bounds = ([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0])
    p0 = [0, 0, 0, 0]

    def model(self, x, p1, p2, p3, p4):
        raise NotImplementedError("Subclasses must implement the model method.")

    def convert_pops(self, pops, errs):
        """
        Convert populations in computational basis to populations in error basis.
        This is a linear transformation that depends on the specific error model.
        For example, for the GT model, we can define:
        Par = P00 - P01 - P10 + P11
        P0 = P00 + P01
        P1 = P00 + P10
        """
        return pops, errs

    def fit_model(self, df):
        P00 = df["P00"].values
        P01 = df["P01"].values
        P10 = df["P10"].values
        P11 = df["P11"].values
        pops = [P00, P01, P10, P11]
        errs = [
            df["P00 Std"].values,
            df["P01 Std"].values,
            df["P10 Std"].values,
            df["P11 Std"].values,
        ]
        pops, errs = self.convert_pops(pops, errs)
        P = np.concatenate(pops)
        sigmas = np.concatenate(errs)
        # check for zero uncertainties and replace with small value to avoid issues in curve_fit
        for i in range(len(sigmas)):
            if sigmas[i] == 0:
                print(
                    f"Warning: zero uncertainty for data point {i}, replacing with large value."
                )
                sigmas[i] = 1.0
        x = df["Clifford Length"]
        popt, pcov = curve_fit(
            self.model,
            x,
            P,
            sigma=sigmas,
            p0=self.p0,
            absolute_sigma=True,
            bounds=self.bounds,
        )
        return popt, pcov

    def rb_to_fits(self, date, rid, sim_outcomes):
        is_brights, pt_cliffs, expected_outcomes = get_rb_data(rid, date, sim_outcomes)
        df = two_qubit_pops(is_brights, pt_cliffs, expected_outcomes)
        popt, pcov = self.fit_model(df)
        return df, popt, pcov

    def pairwise_rb_to_fits(self, date, rid, sim_outcomes):
        is_brights_n, cliff_lens, expected_outcomes = get_rb_data(
            rid, date, sim_outcomes
        )
        num_ions = is_brights_n.shape[-1]
        z_len = len(self.p0)
        popt_matrix = np.zeros((num_ions, num_ions, z_len))
        perr_matrix = np.zeros((num_ions, num_ions, z_len))
        for ion_1 in range(num_ions):
            for ion_2 in range(num_ions):
                is_brights_1 = is_brights_n[:, :, ion_1]
                is_brights_2 = is_brights_n[:, :, ion_2]
                is_brights = np.stack([is_brights_1, is_brights_2], axis=-1)
                df = two_qubit_pops(is_brights, cliff_lens, expected_outcomes)
                popt, pcov = self.fit_model(df)
                popt_matrix[ion_1, ion_2] = popt
                perr_matrix[ion_1, ion_2] = np.sqrt(np.diag(pcov))
                if ion_1 == ion_2:
                    # Set diagonal entries to zero since they are not meaningful (single-ion RB)
                    popt_matrix[ion_1, ion_2] = np.zeros(z_len)
                    perr_matrix[ion_1, ion_2] = np.zeros(z_len)
        return num_ions, popt_matrix, perr_matrix

    def rb_time_to_fits(self, date, rids_arr, times, ions, sim_outcomes):
        is_brights_1, cliff_lens, expected_outcomes = get_rb_data(
            rids_arr[0], date, sim_outcomes
        )
        is_brights_2, _, _ = get_rb_data(rids_arr[1], date, sim_outcomes)
        is_brights_arr = [is_brights_1, is_brights_2]
        df = two_times_analysis(
            is_brights_arr, cliff_lens, expected_outcomes, rids_arr, date, times, ions
        )
        popt, pcov = self.fit_model(df)
        return df, popt, pcov

    def fit_mult_rbs(self, date, rids, sim_outcomes):
        clip_val = 1.0
        param1_arr = []
        param2_arr = []
        param3_arr = []
        param4_arr = []
        param5_arr = []
        for rid in rids:
            df, popt, pcov = self.rb_to_fits(date, rid, sim_outcomes)
            params = popt
            param_errs = np.sqrt(np.diag(pcov))

            if len(params) == 4:
                param1, param2, param3, param4 = params
                param1_err, param2_err, param3_err, param4_err = param_errs
            elif len(params) == 5:
                param1, param2, param3, param4, param5 = params
                param1_err, param2_err, param3_err, param4_err, param5_err = param_errs

            # param1_err = clip_errs(param1_err, clip_val)
            # param2_err = clip_errs(param2_err, clip_val)
            # param3_err = clip_errs(param3_err, clip_val)
            # param4_err = clip_errs(param4_err, clip_val)
            # if len(params) == 5:
            #     param5_err = clip_errs(param5_err, clip_val)

            param1_arr.append((param1, param1_err))
            param2_arr.append((param2, param2_err))
            param3_arr.append((param3, param3_err))
            param4_arr.append((param4, param4_err))
            if len(params) == 5:
                param5_arr.append((param5, param5_err))
        if len(params) == 4:
            return param1_arr, param2_arr, param3_arr, param4_arr
        return param1_arr, param2_arr, param3_arr, param4_arr, param5_arr

    def fit_mult_1q_rbs(self, date, rids, sim_outcomes):
        param1_arr = []
        param2_arr = []
        for rid in rids:
            df, _, _ = self.rb_to_fits(date, rid, sim_outcomes)
            x, P_L, P_L_err, P_R, P_R_err = self.get_single_qubit_pops(df)
            popt, pcov = curve_fit(
                self.single_qubit_model,
                x,
                P_L,
                sigma=P_L_err,
                p0=[0.01, 0.01],
                absolute_sigma=True,
            )
            param1, param2 = popt
            param1_err, param2_err = np.sqrt(np.diag(pcov))
            param1_arr.append((param1, param1_err))
            param2_arr.append((param2, param2_err))
        return param1_arr, param2_arr

    def get_single_qubit_pops(self, df):
        P_00 = df["P00"].values
        P_01 = df["P01"].values
        P_10 = df["P10"].values
        P_11 = df["P11"].values
        P_00_err = df["P00 Std"].values
        P_01_err = df["P01 Std"].values
        P_10_err = df["P10 Std"].values
        P_11_err = df["P11 Std"].values

        x = df["Clifford Length"].values

        P_L = P_00 + P_01
        P_R = P_00 + P_10
        P_L_err = np.sqrt(P_00_err**2 + P_01_err**2)
        P_R_err = np.sqrt(P_00_err**2 + P_10_err**2)
        return x, P_L, P_L_err, P_R, P_R_err

    def get_single_qubit_errors(self, popt):
        e_corr = popt[0]
        e_L = popt[1]
        e_R = popt[2]
        e_cL = e_corr + e_L
        e_cR = e_corr + e_R
        return e_cL, e_cR

    def single_qubit_model(self, m, e_c, spam):
        # From Hughes thesis eq. 3.120
        return (1 - 2 * spam) / 2 * (1 - 2 * e_c) ** m + 1 / 2


class PTMModel(ErrModelBase):
    name = "PTM"
    p1_name = "Global Noise (theta_g)"
    p2_name = "Local Noise (theta_l1)"
    p3_name = "Local Noise (theta_l2)"
    p4_name = "SPAM"
    bounds = ([0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 10.0])
    p0 = [0.1, 0.1, 0.1, 0.01]

    def model(self, x, theta_g, theta_l1, theta_l2, spam):
        """
        PTM model for two-qubit RB. Uses small angle approx.
        """
        Ba1 = 1 - (theta_g**2 + theta_l1**2) / 3
        Ba2 = 1 - (theta_g**2 + theta_l2**2) / 3
        Bb = 1 - theta_g**2 - (theta_l1**2 + theta_l2**2) / 3
        Bc = 1 - (theta_l1**2 + theta_l2**2) / 3
        sp = 1 - 2 * spam
        S1 = sp * Ba1**x
        S2 = sp * Ba2**x
        M1 = 2 * sp**2 * Bb**x / 3
        M2 = sp**2 * Bc**x / 3
        P00 = (1 + S1 + S2 + M1 + M2) / 4
        P01 = (1 + S1 - S2 - M1 - M2) / 4
        P10 = (1 - S1 + S2 - M1 - M2) / 4
        P11 = (1 - S1 - S2 + M1 + M2) / 4
        P = np.concatenate([P00, P01, P10, P11])
        return P

    def parity_model(self, x, theta_g, theta_l1, theta_l2, spam):
        """using global and local parameters"""
        term1 = (1 - (theta_l1**2 + theta_l2**2) / 3) ** x
        term2 = (1 - theta_g**2 - (theta_l1**2 + theta_l2**2) / 3) ** x
        return (2 / 3) * (term1 - term2)


class SO3Model_SUM(ErrModelBase):
    """Random small SO3 rotations mode for two-qubit RB, with sum params."""

    name = "SO3"
    p1_name = "$\\varepsilon_\mathrm{corr}$"
    p2_name = "$\\varepsilon_{L}+\\varepsilon_{R}$"
    p3_name = "$\\varepsilon_{L}-\\varepsilon_{R}$"
    p4_name = "SPAM"
    bounds = ([5e-4, 5e-4, 5e-4, 5e-4], [10.0, 10.0, 10.0, 0.1])
    p0 = [0.1, 0.01, 0.01, 0.01]

    def model(self, x, e_corr, e_LpR, e_LmR, spam):
        """
        Correlated rotations model for two-qubit RB.
        """
        atr = 1 - 2 * e_LpR
        adt = 1 - 2 * e_LpR - 2 * 3 * e_corr
        aL = 1 - (e_LpR + e_LmR) - 2 * e_corr
        aR = 1 - (e_LpR - e_LmR) - 2 * e_corr
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


class SO3Model(ErrModelBase):
    """Random small SO3 rotations mode for two-qubit RB."""

    name = "SO3"
    p1_name = "$\\varepsilon_\mathrm{corr}$"
    p2_name = "$\\varepsilon_{L}$"
    p3_name = "$\\varepsilon_{R}$"
    p4_name = "SPAM"
    bounds = ([1e-4, 1e-4, 1e-4, 1e-4], [10.0, 10.0, 10.0, 0.1])
    p0 = [0.1, 0.01, 0.01, 0.01]

    def model(self, x, e_corr, e_L, e_R, spam):
        """
        Correlated rotations model for two-qubit RB.
        """
        atr = 1 - 2 * e_L - 2 * e_R
        adt = 1 - 2 * e_L - 2 * e_R - 2 * 3 * e_corr
        aL = 1 - 2 * e_L - 2 * e_corr
        aR = 1 - 2 * e_R - 2 * e_corr
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

    def parity_model(self, x, e_corr, e_L_e_R):
        atr = 1 - 2 * e_L_e_R
        adt = 1 - 2 * e_L_e_R - 2 * 3 * e_corr
        return (1 / 3) * (atr**x - adt**x)


class CliffModel(ErrModelBase):
    """Stochastic Clifford model for two-qubit RB. Similar to SO3 but with different coefficients due to anistropy."""

    name = "Clifford"
    p1_name = "Correlated noise (\\varepsilon_{corr})"
    p2_name = "Left qubit depolarisation (\\varepsilon_{L})"
    p3_name = "Right qubit depolarisation (\\varepsilon_{R})"
    p4_name = "Spam (SPAM)"
    bounds = ([0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 10.0])
    p0 = [0.1, 0.1, 0.1, 0.1]

    def model(self, x, e_corr, e_L, e_R, spam):
        """
        Correlated Clifford model for two-qubit RB.
        """
        atr = 1 - e_L - e_R
        adt = 1 - e_L - e_R - e_corr
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

    def parity_model(self, x, e_corr, e_L, e_R):
        """using global and local parameters"""
        atr = 1 - e_L - e_R
        adt = 1 - e_L - e_R - e_corr
        return (2 / 3) * (atr**x - adt**x)


class GTModel(ErrModelBase):
    name = "GT"
    p1_name = "$\\theta_{\\mathrm{com}}$"
    p2_name = "$\\theta_{\\mathrm{diff}}$"
    p3_name = "Angle between axes ($\\theta_{\\mathrm{ax}}$)"
    p4_name = "$\\epsilon_{\\mathrm{SP}}$"
    bounds = ([0.0, 0.0, 0.0, 0.001], [np.pi, np.pi, np.pi, 1.0])
    p0 = [0.5, 0.1, 0.1, 0.01]

    def rod_form(self, theta, n):
        """Rodrigues' rotation formula to get rotation matrix from angle and axis"""
        return (
            np.cos(theta) * np.identity(3)
            + (1 - np.cos(theta)) * np.outer(n, n)
            + np.sin(theta)
            * np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
        )

    def rot_angle(self, R):
        # clamp trace-derived cosine to avoid NaNs from floating point drift
        val = (np.trace(R) - 1.0) / 2.0
        val = max(-1.0, min(1.0, val))
        return np.arccos(val)

    def rot_axis(self, R):
        theta = self.rot_angle(R)
        if abs(theta) < 1e-8:
            # arbitrary axis when rotation is (near) identity
            return np.array([1.0, 0.0, 0.0])
        s = 2.0 * np.sin(theta)
        return np.array(
            [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
        )

    def nn2cd(self, theta_nn, theta_1, theta_2):
        """
        Forward map:
        (theta_nn, theta_1, theta_2) -> (theta_com, theta_diff, theta_cd)

        n1 = (0,0,1)
        n2 = (sin(theta_nn), 0, cos(theta_nn))
        """
        R1 = self.rod_form(theta_1, [0.0, 0.0, 1.0])
        R2 = self.rod_form(theta_2, [np.sin(theta_nn), 0.0, np.cos(theta_nn)])
        Rs = R2 @ np.linalg.inv(R1)
        phi = self.rot_angle(Rs)
        theta_diff = phi / 2.0
        n_d = self.rot_axis(Rs)
        # Rc chosen so that R1 = Rc * Rd
        Rc = R1 @ np.linalg.inv(self.rod_form(-theta_diff, n_d))
        theta_com = self.rot_angle(Rc)
        n_c = self.rot_axis(Rc)
        # angle between axes (clamped)
        theta_cd = np.arccos(np.clip(np.dot(n_c, n_d), -1.0, 1.0))
        return float(theta_com), float(theta_diff), float(theta_cd)

    def cd2nn(self, theta_com, theta_diff, theta_cd):
        """
        Inverse map (one canonical choice):
        (theta_com, theta_diff, theta_cd) -> (theta_nn, theta_1, theta_2)

        Strategy:
        - pick n_c = e_z, pick n_d in xz-plane so angle(n_c,n_d)=theta_cd
        - form Rc = Rot(theta_com, n_c), Rd = Rot(-theta_diff, n_d)
        - build R1 = Rc*Rd, R2 = Rc*Rd^{-1}
        - find rotation Q that maps axis(R1) -> e_z (so R1 becomes single-axis about e_z)
        - conjugate R1,R2 by Q and extract theta_1 (from R1) and theta_2 (from R2)
        - extract theta_nn from axis of R2 after conjugation (axis should be in xz-plane by construction)
        """
        # choose canonical axes in internal representation
        n_c = np.array([0.0, 0.0, 1.0])
        n_d = np.array(
            [np.sin(theta_cd), 0.0, np.cos(theta_cd)]
        )  # x positive convention

        Rc = self.rod_form(theta_com, n_c)
        Rd = self.rod_form(-theta_diff, n_d)

        R1 = Rc @ Rd
        R2 = Rc @ np.linalg.inv(Rd)

        # compute axis of R1 and rotation Q that maps it to e_z
        n1p = self.rot_axis(R1)
        ez = np.array([0.0, 0.0, 1.0])
        cross = np.cross(n1p, ez)
        if np.linalg.norm(cross) < 1e-8 and np.dot(n1p, ez) > 0.9999999:
            Q = np.eye(3)
        else:
            v = cross
            vnorm = np.linalg.norm(v)
            if vnorm < 1e-12:
                # n1p nearly opposite ez: rotate pi about x (arbitrary choice)
                Q = self.rod_form(np.pi, np.array([1.0, 0.0, 0.0]))
            else:
                v = v / vnorm
                ang = np.arccos(np.clip(np.dot(n1p, ez), -1.0, 1.0))
                Q = self.rod_form(ang, v)

        # conjugate so R1 is about ez
        R1p = Q @ R1 @ np.linalg.inv(Q)
        R2p = Q @ R2 @ np.linalg.inv(Q)

        # extract final parameters
        theta_1 = self.rot_angle(R1p)
        theta_2 = self.rot_angle(R2p)
        n2p = self.rot_axis(R2p)

        # recover theta_nn from axis of R2p: axis is (sin(theta_nn), 0, cos(theta_nn)) up to numerical noise.
        theta_nn = np.arctan2(n2p[0], n2p[2])  # signed; bias toward positive x
        if theta_nn < 0:
            # force positive x by reflecting about z-axis (equivalent representation)
            theta_nn = -theta_nn
            Ry = self.rod_form(np.pi, np.array([0.0, 1.0, 0.0]))
            R1p = Ry @ R1p @ np.linalg.inv(Ry)
            R2p = Ry @ R2p @ np.linalg.inv(Ry)
            theta_1 = self.rot_angle(R1p)
            theta_2 = self.rot_angle(R2p)

        # clamp into range [0, pi]
        theta_nn = float(np.clip(theta_nn, 0.0, np.pi))
        return theta_nn, float(theta_1), float(theta_2)

    def model(self, x, theta_com, theta_diff, theta_cd, spam):
        """
        GT model for two-qubit RB.
        """
        theta_nn, theta_1, theta_2 = self.cd2nn(theta_com, theta_diff, theta_cd)
        nn = np.cos(theta_nn)
        s_diag = np.cos(theta_1) + np.cos(theta_2) + np.cos(theta_1) * np.cos(theta_2)
        s_tr = 1 + 2 * (
            np.cos(theta_1) * np.cos(theta_2) + np.sin(theta_1) * np.sin(theta_2) * (nn)
        )
        Ba1 = (1 + 2 * np.cos(theta_1)) / 3
        Ba2 = (1 + 2 * np.cos(theta_2)) / 3
        Bb = s_diag / 2 - s_tr / 6
        Bc = s_tr / 3
        sp = 1 - 2 * spam
        S1 = sp * Ba1**x
        S2 = sp * Ba2**x
        M1 = 2 * sp**2 * Bb**x / 3
        M2 = sp**2 * Bc**x / 3
        P00 = (1 + S1 + S2 + M1 + M2) / 4
        P01 = (1 + S1 - S2 - M1 - M2) / 4
        P10 = (1 - S1 + S2 - M1 - M2) / 4
        P11 = (1 - S1 - S2 + M1 + M2) / 4
        P = np.concatenate([P00, P01, P10, P11])
        return P

    def parity_model(self, x, theta_com, theta_diff, theta_cd):
        """Using common and differential parameters"""
        theta_cd = 1.0
        a_tr = 1 - 4 * theta_diff**2 / 3
        a_dt = (3 - 2 * (theta_com**2 + theta_diff**2)) / 2 - a_tr / 2
        return (2 / 3) * (a_tr**x - a_dt**x)


class GTReducedModel(ErrModelBase):
    name = "GTRed"
    p1_name = "$\\theta_{\\mathrm{com}}$"
    p2_name = "$\\theta_{\\mathrm{diff}}$"
    p3_name = "$\\theta_{1}$"
    p4_name = "$\\theta_{2}$"
    p5_name = "SPAM"
    bounds = ([0.0, 0.0, 0.0, 0.0, 0.0149], [np.pi, np.pi, np.pi, np.pi, 0.015])
    p0 = [0.5, 0.1, 0.1, 0.1, 0.015]

    def convert_pops(self, pops, errs):
        P00, P01, P10, P11 = pops
        P00_err, P01_err, P10_err, P11_err = errs
        Par = P00 - P01 - P10 + P11
        P0 = P00 + P01
        P1 = P00 + P10
        Par_err = np.sqrt(P00_err**2 + P01_err**2 + P10_err**2 + P11_err**2)
        P0_err = np.sqrt(P00_err**2 + P01_err**2)
        P1_err = np.sqrt(P10_err**2 + P11_err**2)
        pops = [Par, P0, P1]
        errs = [Par_err, P0_err, P1_err]
        return pops, errs

    def model(self, x, theta_com, theta_diff, theta_1, theta_2, spam):
        """
        GT model for two-qubit RB. (uses small angle approx)
        """

        # a_tr = 1 - 4 * theta_diff**2 / 3
        # a_dt = (3 - 2 * (theta_com**2 + theta_diff**2)) / 2 - a_tr / 2
        # print("note using small angle approx for GT reduced model")

        a_tr = (1 + 2 * np.cos(2 * theta_diff)) / 3
        a_dt = (1 + np.cos(2 * theta_com) + np.cos(2 * theta_diff)) / 2 - a_tr / 2

        Ba1 = (1 + 2 * np.cos(theta_1)) / 3
        Ba2 = (1 + 2 * np.cos(theta_2)) / 3

        sp = 1 - 2 * spam
        S1 = sp * Ba1**x
        S2 = sp * Ba2**x
        M1 = sp**2 / 3 * a_dt**x
        M2 = sp**2 / 6 * a_tr**x
        Par = 2 * (M1 + M2)
        P0 = 1 / 2 + S1 / 2
        P1 = 1 / 2 + S2 / 2
        P = np.concatenate([Par, P0, P1])
        return P

    def parity_model(self, x, theta_com, theta_diff, spam):
        """Using common and differential parameters"""
        sp = 1 - 2 * spam
        # a_tr = 1 - 4 * theta_diff**2 / 3
        # a_dt = (3 - 2 * (theta_com**2 + theta_diff**2)) / 2 - a_tr / 2
        a_tr = (1 + 2 * np.cos(2 * theta_diff)) / 3
        a_dt = (1 + np.cos(2 * theta_com) + np.cos(2 * theta_diff)) / 2 - a_tr / 2
        return (2 / 3) * sp**2 * (a_tr**x - a_dt**x)


def clip_errs(err, clip_value):
    return min(err, clip_value)


# outdated
# def inject_model(x, y0, a):
#     return np.sqrt(y0**2 + (x * a) ** 2)

# def inject_model_e(x, y0, a):
#     return y0 + a*x

# def detuning_model(x, y0, a, d):
#     return np.sqrt(y0**2 + ((x - d) * a) ** 2)

# def detuning_model_e(x, y0, a, d):
#     return y0**2 + ((x - d) * a) ** 2


def two_times_analysis(
    is_brights_arr, cliff_lens, expected_outcomes, rids_arr, date, times, ions
):
    """
    Need to:
    - Extract sequences from the sequence dataset.
    - Calculate matrix for each rotation in the sequence.
    - Calculate overall matrix for the sequence.
    - Find expected outcomes for each sequence.
    - Look at survival as a function of sequence length.
    """

    # Analyze two-qubit correlations shot by shot
    OO_all = []
    OX_all = []
    XO_all = []
    XX_all = []
    OO_all_sd = []
    OX_all_sd = []
    XO_all_sd = []
    XX_all_sd = []

    t_1, t_2 = times
    i_1, i_2 = ions

    for i, m in enumerate(is_brights_arr[0]):
        OO_shots = []
        OX_shots = []
        XO_shots = []
        XX_shots = []

        for j, shot in enumerate(m):

            shot1 = is_brights_arr[t_1][i][j][i_1]
            shot2 = is_brights_arr[t_2][i][j][i_2]
            OO_shots.append(
                shot1 != expected_outcomes[i] and shot2 != expected_outcomes[i]
            )
            OX_shots.append(
                shot1 != expected_outcomes[i] and shot2 == expected_outcomes[i]
            )
            XO_shots.append(
                shot1 == expected_outcomes[i] and shot2 != expected_outcomes[i]
            )
            XX_shots.append(
                shot1 == expected_outcomes[i] and shot2 == expected_outcomes[i]
            )
        OO_all.append(np.mean(OO_shots))
        OX_all.append(np.mean(OX_shots))
        XO_all.append(np.mean(XO_shots))
        XX_all.append(np.mean(XX_shots))
        OO_all_sd.append(errorbars(OO_shots)[1])
        OX_all_sd.append(errorbars(OX_shots)[1])
        XO_all_sd.append(errorbars(XO_shots)[1])
        XX_all_sd.append(errorbars(XX_shots)[1])

    OO_all = np.array(OO_all)
    OX_all = np.array(OX_all)
    XO_all = np.array(XO_all)
    XX_all = np.array(XX_all)
    OO_all_sd = np.array(OO_all_sd)
    OX_all_sd = np.array(OX_all_sd)
    XO_all_sd = np.array(XO_all_sd)
    XX_all_sd = np.array(XX_all_sd)

    # Find population in each state
    OO_means = []
    OO_errs = []
    OX_means = []
    OX_errs = []
    XO_means = []
    XO_errs = []
    XX_means = []
    XX_errs = []

    seq_uniq = np.unique(cliff_lens)

    for l in seq_uniq:
        OO_subset = OO_all[cliff_lens == l]
        OO_means.append(np.mean(OO_subset))
        OO_errs.append(np.std(OO_subset) / np.sqrt(len(OO_subset)))

        OX_subset = OX_all[cliff_lens == l]
        OX_means.append(np.mean(OX_subset))
        OX_errs.append(np.std(OX_subset) / np.sqrt(len(OX_subset)))

        XO_subset = XO_all[cliff_lens == l]
        XO_means.append(np.mean(XO_subset))
        XO_errs.append(np.std(XO_subset) / np.sqrt(len(XO_subset)))

        XX_subset = XX_all[cliff_lens == l]
        XX_means.append(np.mean(XX_subset))
        XX_errs.append(np.std(XX_subset) / np.sqrt(len(XX_subset)))

    df = pd.DataFrame(
        {
            "Clifford Length": seq_uniq,
            "P00": OO_means,
            "P00 Std": OO_errs,
            "P01": OX_means,
            "P01 Std": OX_errs,
            "P10": XO_means,
            "P10 Std": XO_errs,
            "P11": XX_means,
            "P11 Std": XX_errs,
        }
    )
    return df


class population_data:
    def __init__(
        self,
        map,
        shots=np.array([]),
        pops=np.array([]),
        means=np.array([]),
        stds=np.array([]),
        errs=np.array([]),
    ):
        self.map = map
        self.shots = shots
        self.pops = pops
        self.means = means
        self.stds = stds
        self.errs = errs


def three_qubit_analysis(
    is_brights, cliff_lens, expected_outcomes, rid, date, offset=0
):
    """
    Need to:
    - Extract sequences from the sequence dataset.
    - Calculate matrix for each rotation in the sequence.
    - Calculate overall matrix for the sequence.
    - Find expected outcomes for each sequence.
    - Look at survival as a function of sequence length.
    """

    # Analyze two-qubit correlations shot by shot
    P000 = population_data([0, 0, 0])
    P001 = population_data([0, 0, 1])
    P010 = population_data([0, 1, 0])
    P011 = population_data([0, 1, 1])
    P100 = population_data([1, 0, 0])
    P101 = population_data([1, 0, 1])
    P110 = population_data([1, 1, 0])
    P111 = population_data([1, 1, 1])

    pop_arr = [P000, P001, P010, P011, P100, P101, P110, P111]

    for i, m in enumerate(is_brights):
        for p in pop_arr:
            p.shots = np.array([])

            for j, shot in enumerate(m):
                s = (expected_outcomes[i] + 1) % 2
                p.shots = np.append(
                    p.shots,
                    shot[0] == (s + p.map[0]) % 2
                    and shot[1] == (s + p.map[1]) % 2
                    and shot[2] == (s + p.map[2]) % 2,
                )

            p.pops = np.append(p.pops, np.mean(p.shots))
            p.stds = np.append(p.stds, errorbars(p.shots)[1])

    # Find population in each state
    seq_uniq = np.unique(cliff_lens)
    for p in pop_arr:
        for l in seq_uniq:
            subset = p.pops[cliff_lens == l]
            p.means = np.append(p.means, np.mean(subset))
            p.errs = np.append(p.errs, np.std(subset) / np.sqrt(len(subset)))

    PXOO = population_data([0, 0, "X"])
    PXXO = population_data([0, "X", "X"])

    PXOO.means = np.array(P001.means) + np.array(P010.means) + np.array(P100.means)
    PXOO.errs = np.sqrt(
        np.array(P001.errs) ** 2 + np.array(P010.errs) ** 2 + np.array(P100.errs) ** 2
    )
    PXXO.means = np.array(P011.means) + np.array(P101.means) + np.array(P110.means)
    PXXO.errs = np.sqrt(
        np.array(P011.errs) ** 2 + np.array(P101.errs) ** 2 + np.array(P110.errs) ** 2
    )
    df = pd.DataFrame(
        {
            "Clifford Length": seq_uniq,
            "P000": P000.means,
            "P000 Std": P000.errs,
            "P001": PXOO.means,
            "P001 Std": PXOO.errs,
            "P011": PXXO.means,
            "P011 Std": PXXO.errs,
            "P111": P111.means,
            "P111 Std": P111.errs,
        }
    )
    return df
