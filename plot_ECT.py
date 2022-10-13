from ECT import *

# plot functions for ECT work

def plot_ecc(coords, Cells, filtration, T, title='', ylim=(-5,5), lw=3):
    fig, ax = plt.subplots(2,3, figsize=(15,6))
    fig.suptitle('ECC for filter: {}'.format(title), fontsize=30)
    for j in range(2):
        for k in range(3):
            ax[j,k].plot((0,0),ylim, c='white')
            ax[j,k].set_ylabel('Euler characteristic', fontsize=12)
            ax[j,k].set_xlabel('Sublevel set', fontsize=12)
            ax[j,k].plot(ECC(coords, Cells, filtration, T[3*j+k]),
                         lw=lw, label = 'T = {}'.format(T[3*j+k]))
            ax[j,k].legend(fontsize=14)
    fig.tight_layout()

# plots the ECC for a given filtration and a set of resolutions along with a filtration-colored network
def plot_ecc_filtration(coords, Cells, filtration, T, TT=32, title='', ylim=(-5,5), s=50):
    bins = np.linspace(np.min(filtration), np.max(filtration), TT+1)
    indices = np.digitize(filtration, bins=bins, right=False)

    fig = plt.figure(constrained_layout=True, figsize=(20,7))
    gs = fig.add_gridspec(2,4)
    for j in range(2):
        for k in range(2):
            ax = fig.add_subplot(gs[j,k])
            ax.plot((0,0),ylim, c='white')
            ax.set_ylabel('Euler characteristic', fontsize=12)
            ax.set_xlabel('Sublevel set', fontsize=12)
            ax.plot(ECC(coords, Cells, filtration, T[2*j+k]), lw=3, label = 'T = {}'.format(T[2*j+k]))
            ax.legend(fontsize=20)
    ax = fig.add_subplot(gs[:,2:])
    scatter = ax.scatter(coords[:,1], -coords[:,0], s=s, c=indices, cmap='magma', label='T = {}'.format(TT))
    ax.legend(fontsize=20)
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.01);
    ax.axis('equal');
    cbar.ax.tick_params(labelsize=20)

    fig.suptitle('ECC for filter: {}'.format(title), fontsize=30);
