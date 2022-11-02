import numpy as np
import networkx as nx
import matplotlib as mpl
from itertools import combinations

def compute_ect(G, angles, T):
    '''
    Computes the Euler Characteristic Transform for a (2D) embedded graph in networkx

        Parameters:
            G (networkx graph): A networkx graph whose nodes have attribute 'pos' giving the (x,y) coordinates
            angles (float array): An array of angles (radian) along which to compute the Euler Characteristic Curve
            T (int): Resolution of ECCs; How many linearly spaced slices to take along each direction between the
                        first node and the last node

        Returns:
            ect (int array) [len(angles), T]:  Ect[i] gives the Euler Characteristic Curve along the ith direction.
                                            Use ect.flatten() to get the ECT in its usual presentation as a 1D vector
    '''

    if 'pos' not in G.nodes[0].keys():
        raise AttributeError("Graph G must have node attribute 'pos' for all nodes")

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

    return ect.astype('int')

# helper function for ECT computation
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

def critical_angles(G):
    '''
    Computes the two normal angles (from the x-axis) for every edge in the complete graph of G

        Parameters:
            G (networkx graph): A networkx graph whose nodes have attribute 'pos' giving the (x,y) coordinates

        Returns:
            crit (float array) [n_crit_angles, 1]: Contains every unique normal angle to an edge in the completed
                                                    graph of G (2 per edge)
    '''

    if 'pos' not in G.nodes[0].keys():
        raise AttributeError("Graph G must have node attribute 'pos' for all nodes")

    pos = nx.get_node_attributes(G, 'pos')
    # takes care of duplicate normal angles
    crit = set()

    # loops over each node pairing
    for i,j in combinations(list(G.nodes()), 2):
        p1 = pos[i]
        p2 = pos[j]
        delt = np.array([p2[0]-p1[0], p2[1]-p1[1]])
        # computes angle to x-axis using dot-product with e1
        theta = np.arccos(delt[0]/np.linalg.norm(delt))
        # finds the 2 normal angles
        phi1 = theta + np.pi/2
        phi2 = theta - np.pi/2
        crit.add(phi1)
        crit.add(phi2)

    # makes sure angles are in [0, 2pi] and turns set into np array
    crit = np.mod(list(crit), 2*np.pi)

    return crit

def plot_directional_distance(G, angle, ax = None, cmap = mpl.cm.Blues):
    '''
    Plots a networkx graph G with node coloring representing the height of each node along dir given by angle

        Parameters:
            G (networkx graph): A networkx graph whose nodes have attribute 'pos' giving the (x,y) coordinates
            angle (float): An angle (radian) along which to compute node neights
            cmap (mpl colormap obj): colormap for node coloring

        Returns:
            None
    '''

    if 'pos' not in G.nodes[0].keys():
        raise AttributeError("Graph G must have node attribute 'pos' for all nodes")

    # compute directional distances
    set_directional_distances(G, angle)
    dd = nx.get_node_attributes(G, 'dir_0').values()

    # creates a normalized colormap for the directional 'distances'
    low, *_, high = sorted(dd)
    norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    nx.draw(G,
        nx.get_node_attributes(G, 'pos'),
        node_color=[mapper.to_rgba(i) for i in dd],
        with_labels=True,
        ax = ax)
