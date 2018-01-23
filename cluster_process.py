import numpy as np

def generate_random_directions(n_points, n_dim):
    phi = 2.0 * np.pi * np.random.rand(n_points)

    if n_dim == 2:
        return np.c_[np.cos(phi), np.sin(phi)]

    elif n_dim == 3:
        costheta = 2.0 * np.random.rand(n_points) - np.ones(n_points)
        return np.c_[np.sqrt(1.0 - costheta ** 2) * np.cos(phi), np.sqrt(1.0 - costheta ** 2) * np.sin(phi), costheta]

def generate_random_directions_from_catalogue(points):

    n_points = len(points[:,0])
    n_dim = len(points[0,:])

    return generate_random_directions(n_points, n_dim)

def bias_function(x):
    return 1.0 + 2.0*0.5*(3*x*x - 1.0)

def generate_bias_directions_from_catalogue(points):

    n_points = len(points[:,0])
    n_dim = len(points[0,:])

    # Calculate unit directions from points
    rads = np.zeros(n_points)
    for dim in range(n_dim):
        rads += points[:,dim] * points[:,dim]
    rads = np.sqrt(rads)

    unit_directions = points.copy()
    for dim in range(n_dim):
        unit_directions[:,dim] /= rads

    bias_function_max = bias_function(1.0)

    #Generate random points, keep the ones that make it past the monte carlo rejection sampling
    rejected_mask = np.ones(n_points, dtype=np.bool)
    n_rejected_directions = n_points
    output_directions = generate_random_directions(n_points, n_dim)
    while n_rejected_directions > 0:
        print(n_rejected_directions)
        output_directions[rejected_mask] = generate_random_directions(n_rejected_directions, n_dim)
        test_mu = np.zeros(n_rejected_directions)
        for dim in range(n_dim):
            test_mu += unit_directions[rejected_mask, dim] * output_directions[rejected_mask, dim]
        monte_carlo_random = bias_function_max*np.random.random(n_rejected_directions)
        new_rejected_mask = (monte_carlo_random > bias_function(test_mu))
        rejected_mask[rejected_mask] = new_rejected_mask
        n_rejected_directions = np.sum(rejected_mask)

    return output_directions

def generate_bias_directions(n_points, n_dim):

    if n_dim == 2:
        phi = 2.0 * np.pi * np.random.rand(n_points)
        return np.c_[np.cos(phi), np.sin(phi)]

class periodic_box(object):

    n_dim = 1
    lower_lim = 0.0
    upper_lim = 0.0
    length = 0.0

    def __init__(self, n_dim, lower_lim, upper_lim):

        self.lower_lim = lower_lim
        self.upper_lim = upper_lim
        self.n_dim = n_dim
        self.length = self.upper_lim - self.lower_lim

    def generate_points(self, n_points):
        points = np.zeros((n_points, self.n_dim))
        for dim in range(self.n_dim):
            points[:,dim] = self.lower_lim * np.ones(n_points) + \
               (self.upper_lim - self.lower_lim) * np.random.rand(n_points)
        return points

    def fix_points(self, points, cluster_int):

        new_points = points.copy()
        for dim in range(self.n_dim):
            mask = np.ones(len(new_points[:, 0]))
            while np.sum(mask) > 0:
                mask = (new_points[:, dim] < self.lower_lim)
                new_points[:, dim] += self.length* mask

            mask = np.ones(len(new_points[:, 0]))
            while np.sum(mask) > 0:
                mask = (new_points[:, dim] > self.upper_lim)
                new_points[:, dim] -= self.length* mask

        return new_points, cluster_int

    def rotate(self, points, target_vec):
        return points

class concentric_spherical_volume(object):

    n_dim = 1
    lower_lim = 0.0
    upper_lim = 0.0
    extra_length = 0.0
    extended_lower_lim = 0.0
    extended_upper_lim = 0.0
    volume_ratio = 1.0

    def __init__(self, n_dim, lower_lim, upper_lim, extra_length):
        self.n_dim = n_dim
        self.lower_lim = lower_lim
        self.upper_lim = upper_lim
        self.extra_length = extra_length

        self.extended_lower_lim = np.maximum(self.lower_lim - self.extra_length, 0.0)
        self.extended_upper_lim = self.upper_lim + self.extra_length

        self.volume_ratio = (self.extended_upper_lim ** self.n_dim - self.extended_lower_lim ** self.n_dim) / \
                       (self.upper_lim ** self.n_dim - self.lower_lim ** self.n_dim)

    def generate_points(self, n_points):

        n_points_extended = int(self.volume_ratio * n_points)
        directions = generate_random_directions(n_points_extended, self.n_dim)
        rads = (np.random.rand(n_points_extended) * (self.extended_upper_lim ** self.n_dim - \
                                                     self.extended_lower_lim ** self.n_dim)) ** (1.0 / float(self.n_dim))
        for dim in range(self.n_dim):
            directions[:, dim] *= rads

        return directions

    def fix_points(self, points, cluster_int):

        radius = np.zeros(len(points[:, 0]))
        for dim in range(self.n_dim):
            radius += points[:, dim] * points[:, dim]
        radius = radius ** 0.5
        mask = (radius < self.upper_lim) & (radius > self.lower_lim)
        return points, cluster_int, mask

    def rotate(self, point, target):

        mod_target = np.sqrt(np.sum(target[:] * target[:]))
        target_norm = target / mod_target

        rotated_point = np.zeros(self.n_dim)

        if self.n_dim == 2:
            rotated_point[:] += point[0] * np.array([target_norm[1], -target_norm[0]])
            rotated_point[:] += point[1] * target_norm

        elif self.n_dim == 3:
            ijbasisnorm = np.sqrt(1.0 - target_norm[2] * target_norm[2])
            rotated_point[:] += point[0] * np.array([target_norm[1], -target_norm[0], 0.0]) / ijbasisnorm
            rotated_point[:] += point[1] * np.array([target_norm[0] * target_norm[2], \
                                                     target_norm[1] * target_norm[2], \
                                                     target_norm[2] ** 2 - 1.0]) / ijbasisnorm
            rotated_point[:] += point[2] * target_norm

        return rotated_point

class segment_cox_clusters(object):

    n_clusters = 0
    length = 0.0
    positions = np.array((0.0,))
    directions = np.array((0.0,))
    volume = 0.0
    generate_directions = generate_random_directions_from_catalogue

    def __init__(self, n_clusters, length, volume, generate_directions):

        self.length = length
        self.volume = volume
        self.positions = self.volume.generate_points(n_clusters)
        self.n_clusters = len(self.positions[:, 0])
        self.generate_directions = generate_directions
        self.directions = generate_directions(self.positions)
        #for ii in range(len(self.directions)):
        #    self.directions[ii, :] = self.volume.rotate(self.directions[ii, :], self.positions[ii, :])

    def generate_points(self, n_ppc, add_centrals = False):

        n_points = self.n_clusters * n_ppc
        points = np.zeros((n_points, len(self.positions[0,:])))
        cluster_int = np.zeros(n_points, dtype=np.int)
        for ii in range(n_points):
            cluster_int[ii] = np.random.randint(0, self.n_clusters)
            points[ii, :] = self.positions[cluster_int[ii]] + self.length * (np.random.rand() - 0.5) * self.directions[cluster_int[ii]]

        if add_centrals:
            cluster_int = np.concatenate([np.arange(self.n_clusters), cluster_int])
            points = np.vstack([self.positions, points])

        return self.volume.fix_points(points, cluster_int)

class thomas_process_clusters(object):

    n_clusters = 0
    sigma_transverse = 0.0
    sigma_radial = 0.0
    positions = np.array((0.0,))
    volume = 0.0

    def __init__(self, n_clusters, sigma_transverse, sigma_radial, volume):

        self.sigma_transverse = sigma_transverse
        self.sigma_radial = sigma_radial
        self.volume = volume
        self.positions = self.volume.generate_points(n_clusters)
        self.n_clusters = len(self.positions[:, 0])

    def generate_points(self, n_ppc, add_centrals = False):

        n_points = self.n_clusters * n_ppc
        points = np.zeros((n_points, len(self.positions[0,:])))
        for ii in range(n_points):
            for dim in range(0, self.volume.n_dim - 1):
                points[ii, dim] = np.random.normal(0.0, self.sigma_transverse)
            points[ii, -1] = np.random.normal(0.0, self.sigma_radial)

        cluster_int = np.random.randint(0, self.n_clusters, n_points)
        for ii in range(n_points):
            points[ii, :] = self.volume.rotate(points[ii, :], self.positions[cluster_int[ii]])
            points[ii, :] += self.positions[cluster_int[ii]]

        if add_centrals:
            cluster_int = np.concatenate([np.arange(self.n_clusters), cluster_int])
            points = np.vstack([self.positions, points])

        return self.volume.fix_points(points, cluster_int)

class matern_cluster_process(object):

    n_clusters = 0
    cluster_radius = 0.0
    positions = np.array((0.0,))
    volume = 0.0

    def __init__(self, n_clusters, cluster_radius, volume):

        self.cluster_radius = cluster_radius
        self.volume = volume
        self.positions = self.volume.generate_points(n_clusters)
        self.n_clusters = len(self.positions[:, 0])

    def generate_points(self, n_ppc, add_centrals = False):

        n_points = self.n_clusters * n_ppc
        rad = (np.random.random(n_points) * self.cluster_radius ** self.volume.n_dim) ** (1.0 / self.volume.n_dim)
        points = generate_random_directions(n_points, self.volume.n_dim)
        for dim in range(self.volume.n_dim):
            points[:, dim] *= rad

        cluster_int = np.zeros(n_points, dtype=np.int)
        for ii in range(n_points):
            cluster_int[ii] = np.random.randint(0, self.n_clusters)
            points[ii, :] += self.positions[cluster_int[ii]]

        if add_centrals:
            cluster_int = np.concatenate([np.arange(self.n_clusters), cluster_int])
            points = np.vstack([self.positions, points])

        return self.volume.fix_points(points, cluster_int)
