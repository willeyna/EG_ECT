import numpy as np
import networkx as nx
import matplotlib as mpl

def set_directional_distances(G, angles):

    # makes sure angles is iterable
    angles = np.atleast_1d(angles)

    # for ease of calling in ECT function
    direction_labels = []

    for i, angle in enumerate(angles):
        # 2d only implementation of angle unit vector
        v = np.array([np.cos(angle), np.sin(angle)])

        # computes a proxy for directional distance using the dot product
        directional_dist = [np.dot(v, d) for d in nx.get_node_attributes(G, 'pos').values()]
        # puts directional distance/node index information into dict for graph node input
        dd = dict(enumerate(directional_dist))
        label = 'dir_' + str(i)
        nx.set_node_attributes(G, dd, label)

        direction_labels.append(label)
    return direction_labels

def ECT(G, angles, T):

    # adds dir. dist. values as node attributes and returns names of attributes in a list
    direction_labels = set_directional_distances(G, angles)

    # will store full ECT as n_dir*resolution array for clarity; flatten to get old format
    ect = np.zeros([len(direction_labels), T])

    # loop over dir and slices to compute euler char.
    for i, direction in enumerate(direction_labels):

        d = nx.get_node_attributes(G, direction).values()
        m, M = min(d), max(d)

        # effectively a "startpoint=False" option; creates cross_section limits
        cs_lowerlims = np.linspace(M, m, T, endpoint=False)[::-1]

        for j, lim in enumerate(cs_lowerlims):
            # grab the node if the data v[dir] is < threshold
            cs_nodes = [n for n,v in G.nodes(data=True) if v[direction] <= lim]
            # take subgraph up to threshold
            cs_G = G.subgraph(cs_nodes)

            ect[i, j] = len(cs_G.nodes) - len(cs_G.edges)

    return ect

    # reallys computes (adds to graph) *and* plots
def plot_directional_distance(G, angle, ax = None):

    # compute directional distances
    set_directional_distances(G, angle)
    dd = nx.get_node_attributes(G, 'dir_0').values()

    # creates a normalized colormap for the directional 'distances'
    low, *_, high = sorted(dd)
    norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)

    nx.draw(G,
        nx.get_node_attributes(G, 'pos'),
        node_color=[mapper.to_rgba(i) for i in dd],
        with_labels=True,
        ax = ax)
