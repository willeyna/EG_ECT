from matplotlib import pyplot as plt
import scipy.special as special
import itertools
import numpy as np

def ECC(verts, cells, filtration, T, bbox=None):

    if bbox is None:
        minh = np.min(filtration)
        maxh = np.max(filtration)
    else:
        minh,maxh = bbox

    buckets = [None for i in range(len(cells))]

    buckets[0], bins = np.histogram(filtration, bins=T, range=(minh, maxh))

    for i in range(1,len(buckets)):
        buckets[i], bins = np.histogram(np.max(filtration[cells[i]], axis=1), bins=T, range=(minh, maxh))

    ecc = np.zeros_like(buckets[0])
    for i in range(len(buckets)):
        ecc = np.add(ecc, ((-1)**i)*buckets[i])

    return np.cumsum(ecc)

# 2D directional ect vector computation
def directional_ect(graph, angles = [0, np.pi/2, np.pi, 3*np.pi/4], T = 10, concat = True):
    ect = []
    # needed for dot product
    graph_coords = centerVertices(graph[0])

    circle_dirs = np.column_stack((np.cos(angles), np.sin(angles)))
    for direction in circle_dirs:
        heights = np.sum(graph_coords*direction, axis=1)
        ect.append(ECC(graph_coords, graph, heights, T))

    if concat:
        ect = np.concatenate(ect)

    return ect

# 2D radial distance from center of network ect vector computation
def radial_ect(graph, T = 10):

    graph_coords = centerVertices(graph[0])
    rad_dist = np.linalg.norm(graph_coords, axis = 1)
    ect= ECC(graph_coords, graph, rad_dist, T)

    return ect

def complexify(img, center=True):
    coords = np.nonzero(img)
    coords = np.vstack(coords).T
    keys = [tuple(coords[i,:]) for i in range(len(coords))]
    dcoords = dict(zip(keys, range(len(coords))))
    neighs, subtuples = neighborhood_setup(img.ndim)
    binom = [special.comb(img.ndim, k, exact=True) for k in range(img.ndim+1)]

    hood = np.zeros(len(neighs)+1, dtype=int)-1
    cells = [[] for k in range(img.ndim+1)]

    for voxel in dcoords:
        hood.fill(-1)
        hood = neighborhood(voxel, neighs, hood, dcoords)
        nhood = hood > -1
        if np.all(nhood):
            c = 0
            for k in range(1, img.ndim + 1):
                for j in range(binom[k]):
                    cell = hood[subtuples[neighs[c]]]
                    cells[k].append(cell)
                    c += 1
        else:
            c = 0
            for k in range(1, img.ndim):
                for j in range(binom[k]):
                    cell = nhood[subtuples[neighs[c]]]
                    if np.all(cell):
                        cells[k].append(hood[subtuples[neighs[c]]])
                    c += 1

    cells = [np.array(cells[k]) for k in range(len(cells))]
    if center:
        cells[0] = centerVertices(coords)
    else:
        cells[0] = coords

    return cells

def neighborhood_setup(dimension):
    neighs = sorted(list(itertools.product(range(2), repeat=dimension)), key=np.sum)[1:]
    subtuples = dict()
    for i in range(len(neighs)):
        subtup = [0]
        for j in range(len(neighs)):
            if np.all(np.subtract(neighs[i], neighs[j]) > -1):
                subtup.append(j+1)
        subtuples[neighs[i]] = subtup

    return neighs, subtuples

def neighborhood(voxel, neighs, hood, dcoords):
    hood[0] = dcoords[voxel]
    neighbors = np.add(voxel, neighs)
    for j in range(1,len(hood)):
        key = tuple(neighbors[j-1,:])
        if key in dcoords:
            hood[j] = dcoords[key]
    return hood

def centerVertices(verts):
    return verts -1*np.mean(verts, axis=0)
