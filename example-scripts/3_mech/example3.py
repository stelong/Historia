from Historia.mech.mechanics_solution import (
    calculate_features,
    extract_data_from_logfile,
    extract_limit_cycle,
)
from Historia.shared.plot_utils import plot_pvloop


def main():
    path = "data/"
    nbeats = 4  # extract 4th-beat simulation solution (assuming you have run at least nbeats)

    # -------------------------------------------------------------------------
    parameters = [
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
    ]  # pass parameter names as they chronologically appear in the logfile, otherwise they won't be regex-ed

    M, parameters = extract_data_from_logfile(
        path + "output_log.txt", parameters
    )
    print(
        parameters
    )  # dictionary of param: param value-s used for the run simulation
    print(M.shape)  # full mechanics solution, a 7-column matrix where:
    # column 1: lambda+time vector; columns 2-3-4: phase, lvv, lvp; columns 5-6-7 phase, rvv, rvp

    conv = True
    try:
        M_lc = extract_limit_cycle(M, nbeats=nbeats)
        features = calculate_features(
            M_lc
        )  # if we are able to extract the limit cycle, we can also extract the LV features of interest from it
        print(
            features
        )  # dictionary of feat: feat value-s extracted from the limit cycle LV volume and pressure transients and PV loop
    except:
        conv = (
            not conv
        )  # if we are not able to extract the nbeats-th limit cycle: either we didn't run nbeats or most likely the simulation did not converge
    else:
        pass

    if (
        conv
    ):  # plot LV volume and pressure transients and corresponding PV loop
        plot_pvloop(M, M_lc)


if __name__ == "__main__":
    main()
