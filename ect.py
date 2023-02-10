import numpy as np
import networkx as nx
import matplotlib as mpl
from itertools import combinations

def ecs_matrix(G):
    '''
    Computes the Euler Characteristic Curve Matrix [E] for a (2D) embedded graph in networkx
    The ECC Matrix holds the full information needed to compute the ECT along any direction with O(n) time scaling

    Uses propagation of EC between critical angle regions to compute [E] quickly

        Parameters:
            G (networkx graph): A networkx graph whose nodes have attribute 'pos' giving the (x,y) coordinates
            angles (float array): An array of angles (radian) along which to compute the Euler Characteristic Curve

        Returns:
            ecc_map (dict {float: int}): Dictionary where ecc_map[i] gives a condensed representation of the Euler
                                            Characterstic Curve between angles i and i+1.
                                         Keys are the critical angles of the embedded graph

                                         ecc_map[theta][i] is exactly the Euler Characterstic of the subgraph formed by nodes
                                            up to node_i along an angle between theta and the next critical angle
    '''
    crit = critical_angles(G)
    # pull a list of increasing critical angles
    sorted_angles = sorted(crit)

    # finds an angle between the first two crit angles to initialize iterative process
    theta0 = angular_midpoint(sorted_angles[:2])[0]

    E = np.zeros([len(crit), len(G.nodes())])
    # naively compute first ECS; ECS used to compute all ECS via updates to ECS itself
    ECS, V = ecs(G, theta0)
    # store first ECS
    E[0] = ECS

    # iterate through critical angles in sorted increasing order
    for i, theta in enumerate(sorted_angles[1:]):

        # loop over the line segments that create this critical angle
        for theta_nodes in crit[theta]:

            # type casting for np in1d
            J = list(theta_nodes)
            # find which nodes get swapped in order upon passing theta (find indices in V)
            swap = np.in1d(V, J).nonzero()[0]

            # makes sure the reversing happens in the right order if not a trivial pair swap
            if len(swap) > 2:
                swap = np.sort(swap)

            # reverse node chain order in height sequence
            V[swap] = V[swap[::-1]]

            # re-compute E.C. at each swapped node by counting edges (last one remains the same)
            for swap_ind in swap[:-1]:
                # counts edges coming from the node and going into lower nodes
                n_edges_below = len([edge for edge in G.edges(V[swap_ind]) if
                                     np.argwhere(V==edge[0]) <= swap_ind and np.argwhere(V==edge[1]) <= swap_ind])

                # leave unchanged if i=0 since initial E.C. is always one
                if swap_ind != 0:
                    ECS[swap_ind] = ECS[swap_ind-1] + (1 - n_edges_below)

        # store in E array for all crit angles
        E[i+1] = ECS

    # re-format ECC_map to dictionary format to store angles *and* EC sequence information
    ECC_map = {theta:E[i] for i, theta in enumerate(sorted_angles)}

    return ECC_map


def to_ecc(G, ecs_map, theta):
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
    left_edges = np.array(list(ecs_map.keys()))

    # fixes any problems with modulus 2pi
    if np.sum(theta >= left_edges) == 0:
        theta = 2*np.pi

    # find the interval theta is in
    interval_key = np.max(left_edges[(theta >= left_edges)])

    # grabs the e.c.c. at each node for this interval
    euler_characteristics = ecs_map[interval_key]

    # function finds the euler characteristic at a given fraction along the height function theta
    # may be a better way to actually vectorize this, but it works for now
    ecc = lambda x: [euler_characteristics[np.sum(xi >= node_heights)-1] for xi in x]

    return ecc, euler_characteristics

# computes the ecs along a given angle
# returns \mathcal{E} and \mathcal{V}
def ecs(G, angle):
    n = len(G.nodes())
    ECS = np.zeros(n)

    v = np.array([np.cos(angle), np.sin(angle)])
    # computes a proxy for directional distance using the dot product
    heights = [np.dot(v, d) for d in nx.get_node_attributes(G, 'pos').values()]
    height_ord = np.argsort(heights)

    ECS[0] = 1
    for i in range(1,n):
        ECS[i] = ECS[i-1] + 1
        node = height_ord[i]
        for e in G.edges(node):
            if heights[e[0]] <= heights[node] and heights[e[1]] <= heights[node]:
                ECS[i] -= 1

    return ECS, height_ord

# mostly don't use this anymore
def ect(G, angles, T):
    '''
    Naive computation of finite direction ECT the Euler Characteristic Transform for a (2D) embedded graph in networkx

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
            crit (dict): Dictiomary whose keys are the critical angles, theta, of the graph and values are each a list
                            where entries of these lists are tuples (node_1, node_2, ..., node_n) giving the nodes
                            in each connected line segment ('chain of pairs') at angle theta
    '''

    pos = nx.get_node_attributes(G, 'pos')
    crit = dict()

    # loops over each node pairing
    for i,j in combinations(list(G.nodes()), 2):
        # finds one of the normal angles within [0, 2pi)
        theta =  np.mod(np.arctan2((pos[j][1]-pos[i][1]), (pos[j][0]-pos[i][0])) + np.pi/2, 2*np.pi)
        # finds the other complementary angle
        theta2 = np.mod(theta - np.pi, 2*np.pi)

        # handles case where multiple pairs form an edge at angle theta
        if theta in crit.keys():
            crit[theta].append((i,j))
            crit[theta2].append((i,j))

        # adds theta: (pair) to crit dict
        else:
            crit[theta] = [(i,j)]
            crit[theta2] = [(i,j)]

    # merge any connected segments that are longer than 2 nodes
    crit = {k: merge(v) for k, v in crit.items()}

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

#################################### HELPER FUNCTIONS / UTILS ##########################################################

def random_graph(N=30, E=30, xrange = [-1,1], yrange = [-1,1]):
    '''
    Creates a random embedded graph-- for testing purposes only
    '''
    G = nx.Graph()

    indices = np.arange(0,N)
    X = np.random.uniform(xrange[0], xrange[1], N)
    Y = np.random.uniform(yrange[0], yrange[1], N)

    # gets node positions
    pos = dict(zip(indices, np.stack([X,Y]).T))

    # creates edges
    edges = [np.random.choice(indices, 2, replace = False) for i in range(E)]

    G.add_nodes_from(pos.keys())
    G.add_edges_from(edges)
    nx.set_node_attributes(G, pos, 'pos')

    return G

def set_directional_distances(G, angles):
    '''
    Given an embedded graph G and a list of directions adds an attribute "dir_i" to each node acting as a proxy
        for the height of the node along angle i.
    Helper function for computing ECT.
    '''

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

# from https://stackoverflow.com/questions/9110837/python-simple-list-merging-based-on-intersections
def merge(lsts):
    '''
    Given a list of lists, merges together any which share an element.
    Used in computing which edges (chains of pairs) create each critical angle in critical_angles()
    '''
    sets = [set(lst) for lst in lsts if lst]
    merged = True
    while merged:
        merged = False
        results = []
        while sets:
            common, rest = sets[0], sets[1:]
            sets = []
            for x in rest:
                if x.isdisjoint(common):
                    sets.append(x)
                else:
                    merged = True
                    common |= x
            results.append(common)
        sets = results
    return sets

def list_reverse(L, swap):
    '''
    Reverses order of elements in a multiindex "swap"
    Helper function for ecc_map()
    '''
    L = np.array(L, dtype = int)
    L[swap] = L[swap[::-1]]
    return list(L)
