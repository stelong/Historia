import random

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import torch
from gpytGPE.gpe import GPEmul

from Historia.history import hm
from Historia.shared.design_utils import get_minmax, lhd, read_labels
from Historia.shared.plot_utils import get_col, interp_col

SEED = 8

def plot_one_pair(fig, axis, x, y, z):
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)

    try:
        triang = tri.Triangulation(x, y)
        interpolator = tri.LinearTriInterpolator(triang, z)
        Xi, Yi = np.meshgrid(xi, yi)
        zi = interpolator(Xi, Yi)

        cntr = axis.contourf(xi, yi, zi, cmap="RdBu_r")
        fig.colorbar(cntr, ax=axis)
    except:
        pass

    return fig, axis


def plot_pairwise(Xdata, z, xlabels, color="C0"):
    sample_dim = Xdata.shape[0]
    in_dim = Xdata.shape[1]
    fig, axes = plt.subplots(
        nrows=in_dim,
        ncols=in_dim,
        sharex="col",
        sharey="row",
        figsize=(20, 11.3),
    )
    for i, axis in enumerate(axes.flatten()):
        if i % in_dim < i // in_dim:
            x, y = Xdata[:, i % in_dim], Xdata[:, i // in_dim]
            fig, axis = plot_one_pair(fig, axis, x, y, z)
        
        if i // in_dim == in_dim - 1:
            axis.set_xlabel(xlabels[i % in_dim])
        if i % in_dim == 0:
            axis.set_ylabel(xlabels[i // in_dim])
    for i in range(in_dim):
        for j in range(in_dim):
            if i <= j:
                axes[i, j].set_visible(False)
    plt.suptitle(
        "Sample dimension = {} points".format(sample_dim),
        x=0.1,
        y=0.95,
        ha="left",
        va="top",
    )
    plt.show()
    return


def main():
    # ----------------------------------------------------------------
    # Make the code reproducible
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ----------------------------------------------------------------
    path = "data/"

    # ----------------------------------------------------------------
    # Load experimental values (mean +- std) you aim to match
    path_match = path + "match/"
    exp_mean = np.loadtxt(path_match + "exp_mean.txt", dtype=float)
    exp_std = np.loadtxt(path_match + "exp_std.txt", dtype=float)
    exp_var = np.power(exp_std, 2)

    # ----------------------------------------------------------------
    # Load input parameters and output features' names
    path_lab = path + "labels/"
    xlabels = read_labels(path_lab + "xlabels.txt")
    ylabels = read_labels(path_lab + "ylabels.txt")
    features_idx_dict = {key: idx for idx, key in enumerate(ylabels)}

    # ----------------------------------------------------------------
    # Define the list of features to match (these would normally correspond to the (sub)set of output features for which you have experimental values)
    active_features = ["EDV", "ESV", "ET", "IVRT", "PeakP", "maxdP", "mindP"]
    active_idx = [features_idx_dict[key] for key in active_features]

    exp_mean = exp_mean[active_idx]
    exp_var = exp_var[active_idx]
    ylabels = [ylabels[idx] for idx in active_idx]

    # ----------------------------------------------------------------
    # Load a pre-trained univariate Gaussian process emulator (GPE) for each output feature to match
    path_gpes = path + "gpes/"
    emulator = []
    for idx in active_idx:
        loadpath = path_gpes + str(idx) + "/"
        X_train = np.loadtxt(loadpath + "X_train.txt", dtype=float)
        y_train = np.loadtxt(loadpath + "y_train.txt", dtype=float)
        emul = GPEmul.load(
            X_train, y_train, loadpath=loadpath
        )  # NOTICE: GPEs must have been trained using gpytGPE library (https://github.com/stelong/gpytGPE)
        emulator.append(emul)

    I = get_minmax(
        X_train
    )  # get the spanning range for each of the parameters from the training dataset

    # ----------------------------------------------------------------

    waveno = 1  # wave id number
    cutoff = 3.0  # threshold value for the implausibility criterion
    maxno = 1  # max implausibility will be taken across all the output feature till the last worse impl. measure. If maxno=2 --> till the previous-to-last worse impl. measure and so on.

    W = hm.Wave(
        emulator=emulator,
        Itrain=I,
        cutoff=cutoff,
        maxno=maxno,
        mean=exp_mean,
        var=exp_var,
    )  # instantiate the wave object

    n_samples = 100000
    X = lhd(
        I, n_samples
    )  # initial wave is performed on a big Latin hypercube design using same parameter ranges of the training dataset

    W.find_regions(
        X
    )  # enforce the implausibility criterion to detect regions of non-implausible and of implausible points
    W.print_stats()  # show statistics about the two obtained spaces
    
    # ----------------------------------------------------------------
    X = W.NIMP
    active_feature_idx = 0
    y_mean, _ = emulator[active_feature_idx].predict(X)

    plot_pairwise(X, y_mean, xlabels)

if __name__ == '__main__':
    main()