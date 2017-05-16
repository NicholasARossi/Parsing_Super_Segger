import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import math
import entropy_estimators as ee

def num_bins_calculator(num_points):
    return round(np.sqrt(num_points/5))


def calc_entropy(x,bins):
    c_x = np.histogram(x, bins)[0]/sum(np.histogram(x, 5)[0])

    entropy=0
    for prob in c_x:
        if prob!=0:

            entropy += - prob * math.log(prob, 2)


    return entropy


def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

def mutual_info_score(labels_true, labels_pred, contingency=None):

    # if contingency is None:
    #     labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    #     contingency = contingency_matrix(labels_true, labels_pred, sparse=True)

    if isinstance(contingency, np.ndarray):
        # For an array
        nzx, nzy = np.nonzero(contingency)
        nz_val = contingency[nzx, nzy]
    elif sp.issparse(contingency):
        # For a sparse matrix
        nzx, nzy, nz_val = sp.find(contingency)
    else:
        raise ValueError("Unsupported type for 'contingency': %s" %
                         type(contingency))

    contingency_sum = contingency.sum()
    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))
    log_contingency_nm = np.log(nz_val)
    contingency_nm = nz_val / contingency_sum
    # Don't need to calculate the full outer product, just for non-zeroes
    outer = pi.take(nzx) * pj.take(nzy)
    log_outer = -np.log(outer) + np.log(pi.sum()) + np.log(pj.sum())
    mi = (contingency_nm * (log_contingency_nm - np.log(contingency_sum)) +
          contingency_nm * log_outer)
    return mi.sum()

if __name__ == "__main__":
    from os import listdir

    names = listdir('output_files')
    # for name in names:


    mat_contents = sio.loadmat('output_files/' + names[0])
    color_mat = mat_contents['colors']
    fig, ax = plt.subplots()
    fig2, ax1 = plt.subplots()
    ax2 = ax.twinx()
    t_len = mat_contents['colors'].shape[1]
    print(t_len)
    t_vect = np.linspace(0, (t_len - 1) * 3, t_len)
    # ax3 = fig2.add_subplot(111, label="1")
    entropy_mat = np.zeros((3, t_len))
    mi_mat = np.zeros((3, t_len))

    # for cell in range(color_mat.shape[0]):
    #     ax2.plot(t_vect, color_mat[cell, :, 0], linewidth=3, color='orange', alpha=0.25)
    #     # ax3.plot(color_mat[cell, :, 1], color_mat[cell, :, 2], linewidth=3, color='orange', alpha=0.25)

    for j in np.arange(3):

        mat_contents = sio.loadmat('output_files/' + names[j])
        color_mat = mat_contents['colors']
        t_len = color_mat.shape[1]
        print(t_len)
        # for cell in range(color_mat.shape[0]):
        # ax.plot(t_vect,color_mat[cell, :, 0],linewidth=3,color='teal',alpha=0.25)
        bins = num_bins_calculator(color_mat.shape[0])
        entropies = []
        mis = []
        for t in range(t_len):
            entropies.append(calc_entropy(color_mat[:, t, 0], 10))

            mis.append(calc_MI(color_mat[:, t, 1], color_mat[:, t, 2], bins))


        entropy_mat[j, :] = entropies
        mi_mat[j, :] = mis

    entropy_mat2 = np.zeros((3, t_len))
    mi_mat2 = np.zeros((3, t_len))
    for k in np.arange(3):

        mat_contents = sio.loadmat('output_files/' + names[k+3])
        color_mat = mat_contents['colors']
        t_len = color_mat.shape[1]
        print(t_len)
        # for cell in range(color_mat.shape[0]):
        # ax.plot(t_vect,color_mat[cell, :, 0],linewidth=3,color='teal',alpha=0.25)
        bins = num_bins_calculator(color_mat.shape[0])
        entropies = []
        mis = []
        for t in range(t_len):
            entropies.append(calc_entropy(color_mat[:, t, 0], 10))

            mis.append(calc_MI(color_mat[:, t, 1], color_mat[:, t, 2], bins))

        entropy_mat2[k, :] = entropies
        mi_mat2[k, :] = mis

    ax.errorbar(t_vect, np.mean(entropy_mat, 0), yerr=np.std(entropy_mat, 0), linewidth=3, color='teal')
    ax.errorbar(t_vect, np.mean(entropy_mat2, 0), yerr=np.std(entropy_mat2, 0), linewidth=3, color='orange')

    ax1.errorbar(t_vect, np.mean(mi_mat, 0), yerr=np.std(mi_mat, 0), linewidth=3, color='teal')
    ax1.errorbar(t_vect, np.mean(mi_mat2, 0), yerr=np.std(mi_mat2, 0), linewidth=3, color='orange')
    ax1.set_ylabel('Mutual Information (bits)')
    ax1.set_xlabel('Time (Minutes)')
    ax.set_xlim([-5, 253])
    ax1.set_xlim([-5, 253])
    ax.set_xlabel('time (minutes)')
    ax.set_ylabel('Entropy (bits)')
    ax2.set_ylabel('Flourescence (AU)')
    fig.savefig('figures/2a.png', bbox_inches='tight')

    fig2.savefig('figures/2b.png', bbox_inches='tight')