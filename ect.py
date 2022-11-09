import numpy as np
import networkx as nx
import matplotlib as mpl
from itertools import combinations

# come up with a better name for it
def ecc_map(G):
    '''
    Computes the Euler Characteristic Curve Matrix for a (2D) embedded graph in networkx
    The ECC Matrix holds the full information needed to compute the ECC

        Parameters:
            G (networkx graph): A networkx graph whose nodes have attribute 'pos' giving the (x,y) coordinates
            angles (float array): An array of angles (radian) along which to compute the Euler Characteristic Curve

        Returns:
            ecc_map (dict {float: int}): Dictionary where ecc_map[i] gives a condensed representation of the Euler
                                            Characterstic Curve between angles i and i+1.
                                         Keys are the critical angles of the embedded graph
                                         ecc_map[i] is exactly the Euler Characterstic of the subgraph formed by nodes
                                            up to node_i along an angle between i and i+1
    '''
    mapping = dict()
    # find critical anlges and midpoints between them
    crit = critical_angles(G)
    crit_mp = angular_midpoint(crit)
    # compute directional distances between each crit angle and add to G; labels tracks the dir index naming
    labels =  set_directional_distances(G, crit_mp)

    E = np.zeros([len(labels), len(G.nodes())])

    for i, label in enumerate(labels):
        # get the height of each node along the chosen direction
        node_heights = np.sort(list(nx.get_node_attributes(G, label).values()))

        # compute euler characteristic at each node height's induced subgraph
        for j, lim in enumerate(node_heights):
            cs_nodes = [n for n,v in G.nodes(data=True) if v[label] <= lim]

            # take subgraph up to threshold
            cs_G = G.subgraph(cs_nodes)

            # store E.C. at jth node along direction i
            E[i,j] = len(cs_G.nodes) - len(cs_G.edges)

    # store as a dictionary so one doesn't need to track the critical angles to index the matrix in computing ECC
    for i, row in enumerate(E):
        mapping[crit[i]] = row
    return mapping


def ecc(G, ecc_map, theta):
    '''
    Given an Euler Characteristic Curve Map (from S1 to ECC), returns a function that computes the E.C. at a given
        percentage along the height function

    Function structured logically as follows:
        1. Compute fractional node heights along theta
        2. Find which critical angles theta is between
        3. Return the piecewise constant function mapping fractional height to an Euler Charactersitic

    Function works as a proof of concept of computing the Euler Characterstic Curve along a direction, without needing
        to further compute any more Euler Characterstics (and thus remove the computation used in computing subgraphs)

        Parameters:
            G (networkx graph): A networkx graph whose nodes have attribute 'pos' giving the (x,y) coordinates
            ecc_map (dict, {float: int}): object returned by ecc_map()
            theta (float): Angle on S1 between [0, 2pi) (under mod 2pi)

        Returns:
            ecc (function): Piecewise linear function taking in a fractional height (or an array of such) along theta
                                and returning the Euler Characterstic at each height.
                            ex) ecc(linspace(0,1,T)) gives the traditional computation of ECC using "resolution" T
    '''
    dd = set_directional_distances(G, theta)[0]
    node_heights = np.sort(list(nx.get_node_attributes(G, dd).values()))
    # shift heights to start at 0 (important to remove negative direction)
    node_heights = node_heights - np.min(node_heights)
    # normalize to percentile heights
    node_heights = node_heights / np.max(node_heights)

    # left bound of each critical angle interval
    left_edges = np.array(list(ecc_map.keys()))

    # fixes any problems with modulus 2pi
    if np.sum(theta >= left_edges) == 0:
        theta = 2*np.pi

    # find the interval theta is in
    interval_key = np.max(left_edges[(theta >= left_edges)])

    # grabs the e.c.c. at each node for this interval
    euler_characteristics = ecc_map[interval_key]

    # function finds the euler characteristic at a given fraction along the height function theta
    # may be a better way to actually vectorize this, but it works for now
    ecc = lambda x: [euler_characteristics[np.sum(xi >= node_heights)-1] for xi in x]

    return(ecc)

########################

def compute_ect(G, angles, T):
    '''
    Computes the Euler Characteristic Transform for a (2D) embedded graph in networkx (naively)

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
        theta = np.arctan2((p2[1]-p1[1]), (p2[0]-p1[0])) + np.pi/2
        # finds one of the normal angles
        crit.add(theta)

    # get rid of set formatting
    crit = list(crit)
    # finds the complementary normal angle to each existing one in parallel
    crit = np.concatenate([crit, np.array(crit) - np.pi])
    # makes sure angles are in [0, 2pi] and turns set into np array
    crit = np.mod(list(crit), 2*np.pi)

    return np.sort(crit)

def angular_midpoint(theta):
    '''
    Computes the midpoint angles for a sequence of angles

        Parameters:
            theta (array): Array of angles on S1 between [0, 2pi) (under mod 2pi)

        Returns:
            midpts (array) [len(theta),]: midpts[i] is the midpoint angle between theta_i and theta_{i+1} where
                                            midpts[-1] is the midpoint between theta[-1] and theta[0] (mod 2pi)
    '''
    phi = np.append(theta, theta[0]+(2*np.pi))
    midpts = np.mod((phi[1:] + phi[:-1])/2, 2*np.pi)
    return midpts

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

    def random_graph(N=30, E=30, xrange = [-1,1], yrange = [-1,1]):
        G = nx.Graph()

        indices = np.arange(0,N)
        X = np.random.uniform(xrange[0], xrange[1], N)
        Y = np.random.uniform(yrange[0], yrange[1], N)

        # gets node positions
        pos = dict(zip(indices, np.stack([X,Y]).T))

        # creates edges
        edges = [np.random.choice(indices, 2) for i in range(E)]

        G.add_nodes_from(pos.keys())
        G.add_edges_from(edges)
        nx.set_node_attributes(G, pos, 'pos')

        return G
