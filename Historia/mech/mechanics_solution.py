import re

import numpy as np


def extract_data_from_logfile(tag, parameters):
    params_compiled_list = [
        re.compile(r"[-]" + f"{p}" + r"\s+=\s+(\S+)$") for p in parameters
    ]
    p0 = re.compile(r"[=]+\sInitialization\sdone[\s\S]+")
    p0_lv = re.compile(r"LV\sCavity\svolume\s=\s(\d+\.\d+)")
    p0_rv = re.compile(r"RV\sCavity\svolume\s=\s(\d+\.\d+)")
    p1 = re.compile(
        r"(?<=\*\sSolving\spreload\sloop\slambda\s=\s)(\d*\.\d+|\d+)"
    )
    p2_lv = re.compile(
        r"LV\sPhase\s=\s([0-4])\t\sLV\s/\\Volume\s=\s([-]?\d+\.\d+)\sul\s+LVP\s=\s([-]?\d+\.\d+)\skPa"
    )
    p2_rv = re.compile(
        r"RV\sPhase\s=\s([0-4])\t\sRV\s/\\Volume\s=\s([-]?\d+\.\d+)\sul\s+RVP\s=\s([-]?\d+\.\d+)\skPa"
    )
    p3 = re.compile(r"[=]+\sPreload\sphase\sdone[\s\S]+")
    p4 = re.compile(r"(?<=\*\sT\s->\s)\d+\.\d+")
    p5 = re.compile(r"[=]+\sSimulation\scompleted\ssuccessfully[\s\S]+")

    f = open(tag, "r")
    line = f.readlines()
    n = len(line)

    i = 1
    params_count = 0
    params_dict = {}
    while i < n:
        m0 = re.search(params_compiled_list[params_count], line[i])
        if m0 is not None:
            params_dict[parameters[params_count]] = float(m0.groups()[0])
            params_count += 1
            i = 1
        if params_count == len(parameters):
            break
        i += 1

    while i < n:
        m0 = re.search(p0, line[i])
        if m0 is not None:
            break
        i += 1

    lvv0 = float(re.search(p0_lv, line[i + 1]).groups()[0])
    rvv0 = float(re.search(p0_rv, line[i + 2]).groups()[0])

    t = [0.0]

    lv_phase = []
    lv_dv = []
    lvp = []

    rv_phase = []
    rv_dv = []
    rvp = []

    start_collecting_t = False
    i += 3
    while i < n:
        m1 = re.search(p1, line[i])
        if m1 is not None:
            t.append(float(m1.group(0)))

        m2_lv = re.search(p2_lv, line[i])
        if m2_lv is not None:
            lv_phase.append(float(m2_lv.group(1)))
            lv_dv.append(float(m2_lv.group(2)))
            lvp.append(float(m2_lv.group(3)))

        m2_rv = re.search(p2_rv, line[i])
        if m2_rv is not None:
            rv_phase.append(float(m2_rv.group(1)))
            rv_dv.append(float(m2_rv.group(2)))
            rvp.append(float(m2_rv.group(3)))

        if re.match(p3, line[i]) is not None:
            start_collecting_t = True
            break

        i += 1

    if len(t) > len(lv_phase):
        t = t[:-1]

    if start_collecting_t:
        while i < n:
            m4 = re.search(p4, line[i])
            if m4 is not None:
                t.append(float(m4.group(0)))

            m2_lv = re.search(p2_lv, line[i])
            if m2_lv is not None:
                lv_phase.append(float(m2_lv.group(1)))
                lv_dv.append(float(m2_lv.group(2)))
                lvp.append(float(m2_lv.group(3)))

            m2_rv = re.search(p2_rv, line[i])
            if m2_rv is not None:
                rv_phase.append(float(m2_rv.group(1)))
                rv_dv.append(float(m2_rv.group(2)))
                rvp.append(float(m2_rv.group(3)))

            if re.match(p5, line[i]) is not None:
                break

            i += 1

        if len(t) > len(lv_phase):
            t = t[:-1]

    f.close()

    lvv = [lvv0 + x for x in lv_dv]
    rvv = [rvv0 + x for x in rv_dv]

    output_list = [
        np.array(t),
        np.array(lv_phase),
        np.array(lvv),
        np.array(lvp),
        np.array(rv_phase),
        np.array(rvv),
        np.array(rvp),
    ]

    M = np.zeros((len(t), 0), dtype=float)
    for elem in output_list:
        M = np.hstack((M, elem.reshape(-1, 1)))

    return M, params_dict


def divide_et_impera_phases(phase):
    phase = [int(elem) for elem in list(phase)]
    n = len(phase)
    c = 0
    p = [phase[c]]
    idx = [c]
    while c < n - 1:
        c += 1
        if phase[c] != phase[c - 1]:
            p.append(phase[c])
            idx.append(c - 1)
            idx.append(c)
    idx.append(n - 1)

    return p, idx


def extract_limit_cycle(M, nbeats=4, ventricle="LV", shifted=False):
    if ventricle == "LV":
        phase = M[:, 1]

    elif ventricle == "RV":
        phase = M[:, 4]

    else:
        raise ValueError("Not a valide ventricle! Choose either 'LV' or 'RV'.")

    p, idx = divide_et_impera_phases(phase)

    lp = len(p)
    convp = [0, 4] + nbeats * [
        1,
        2,
        3,
        4,
    ]  # starts with preload filling (Phase 0) and diastolic filling (Phase 4) before running the first beat (Phases 1-4)
    lconvp = len(convp)
    if (
        lp > lconvp
    ):  # assuming you have successfully run more than 4 beats, extract the 4th beat cycle
        init = idx[: 2 * lconvp][-8]
        end = idx[: 2 * lconvp][-1] + 1
        M_lc = M[init:end, :]

        if shifted:
            M_lc[:, 0] = (
                M_lc[:, 0] - M_lc[0, 0]
            )  # shift time to start from 0 ms

        return M_lc

    else:
        raise ValueError("Simulation did not converge to limit cycle!")


def check_derivative(y):
    delta = y[:-1] - y[1:]
    mean, std = np.mean(delta), np.std(delta)
    l = list(np.where(np.abs(delta) > mean + 3 * std)[0])
    if l:
        for idx in l:
            if delta[idx] > 0:
                y[idx + 1] = y[idx]
            elif delta[idx] < 0:
                y[idx] = y[idx + 1]
    return y


def calculate_features(M_lc, ventricle="LV"):
    features = [
        "EDV",
        "ESV",
        "SV",
        "EF",
        "IVCT",
        "ET",
        "IVRT",
        "Tdiast",
        "PeakP",
        "Tpeak",
        "ESP",
        "maxdP",
        "mindP",
        "Tau",
    ]  # hard-coded for the moment

    t = M_lc[:, 0]

    if ventricle == "LV":
        phase = M_lc[:, 1]
        volume = M_lc[:, 2]
        pressure = M_lc[:, 3]

    elif ventricle == "RV":
        phase = M_lc[:, 4]
        volume = M_lc[:, 5]
        pressure = M_lc[:, 6]

    else:
        raise ValueError("Not a valide ventricle! Choose either 'LV' or 'RV'.")

    p, idx = divide_et_impera_phases(phase)

    phase_dict = {p[i]: [idx[2 * i], idx[2 * i + 1]] for i in range(len(p))}

    f1 = np.mean(volume[phase_dict[1][0] : phase_dict[1][-1] + 1])
    f2 = volume[phase_dict[3][0]]
    f3 = f1 - f2
    f4 = 100 * f3 / f1
    f5 = t[phase_dict[1][-1]] - t[phase_dict[1][0]]
    f6 = t[phase_dict[2][-1]] - t[phase_dict[2][0]]
    f7 = t[phase_dict[3][-1]] - t[phase_dict[3][0]]
    f8 = t[phase_dict[4][-1]] - t[phase_dict[3][0]]

    imax = np.argmax(pressure)

    f9 = pressure[imax]
    f10 = t[imax] - t[0]
    f11 = pressure[phase_dict[2][-1]]

    dP = np.gradient(pressure, t)
    dP = check_derivative(dP)
    imin = phase_dict[3][0] + np.argmin(dP[phase_dict[3][0] :])

    f12 = np.max(dP)
    f13 = dP[imin]
    f14 = -(pressure[imin] - np.min(pressure[imin:])) / (2 * dP[imin])

    output_list = [
        f1,
        f2,
        f3,
        f4,
        f5,
        f6,
        f7,
        f8,
        f9,
        f10,
        f11,
        f12,
        f13,
        f14,
    ]  # can possibly be expanded to extract more features
    feats_dict = {key: val for key, val in zip(features, output_list)}

    return feats_dict


def save_stats(
    loadpath,
    savepath,
    parameters=[
        "p",
        "ap",
        "z",
        "c1",
        "ca50",
        "beta1",
        "koff",
        "ntrpn",
        "kxb",
        "nperm",
        "perm50",
        "Tref",
    ],  # pass parameter names as they chronologically appear in the logfile, otherwise they won't be regex-ed
    ventricle="LV",
):

    M, parameters_dict = extract_data_from_logfile(
        loadpath + "output_log.txt", parameters
    )
    x = np.array(list(parameters_dict.values()))
    np.savetxt(savepath + "M.txt", M, fmt="%.6f")
    np.savetxt(savepath + "x.txt", x.reshape(1, -1), fmt="%.6f")

    try:
        M_lc = extract_limit_cycle(
            M, nbeats=4, ventricle=ventricle, shifted=True
        )
        features_dict = calculate_features(M_lc, ventricle=ventricle)
        y = np.array(list(features_dict.values()))
        np.savetxt(savepath + "M_lc.txt", M_lc, fmt="%.6f")
        np.savetxt(
            savepath + f"y_{ventricle}.txt", y.reshape(1, -1), fmt="%.6f"
        )
        conv = True

    except:
        conv = False

    return conv
