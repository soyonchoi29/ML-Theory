import numpy as np
import itertools
import matplotlib.pyplot as plt
import math


# Q1: sample n random points from a d-dimensional hypercube with side_length
def sample_hypercube(n, d, side_length=1):
    return np.random.rand(n, d) * side_length


# Q4: sample n random points from a d-dimensional hypersphere of r=0.5
def sample_hypersphere(n, d, r=0.5):
    angles = np.random.rand(n, d-1)*np.pi  # uniformly from 0 to pi
    cosines = np.concatenate([np.cos(angles), np.ones((n, 1))], axis=1)

    sines = np.concatenate([np.ones((n,1)), np.sin(angles)], axis=1)
    sines = np.log(sines)@np.triu(np.ones((d, d)))
    sines = np.exp(sines)
    sines[:, -1] = np.multiply(sines[:, -1], (np.random.randint(0, 2, size=(n, ))*2)-1)

    rad = np.random.rand(n, 1)  # sample uniformly
    rad = (rad**(1./d))*r  # then get the dth root
    rad = rad.repeat(d, axis=0).reshape(cosines.shape)

    points = np.multiply(np.multiply(cosines, sines), rad)
    return points


# Q3: returns volumes of a hypercube up to dimension d
def hypercube_volumes(d, side_length=1):
    return np.ones((d,), dtype=np.float64)*side_length**d


# Q3: returns volumes of a hypersphere from 1 up to dimension d
def hypersphere_volumes(d, r=0.5):
    vols = [2.*r]
    coeff = 2.*math.pi*(r**2)
    for i in range(1, d):
        vols.append((coeff/float(i+1))*vols[-1])
    return vols


# sample T times and return the average of the T r/R ratios
def get_avg_ratio(T, n, d):
    ratios = np.ones((T,))
    for i in range(T):
        sample = sample_hypercube(n, d)
        sample = np.array(list(itertools.combinations(sample, 2)))
        sample = sample.reshape((sample.shape[0]*sample.shape[1], sample.shape[2]))
        diffs = np.diff(sample, axis=0)[::2]
        norms = np.linalg.norm(diffs, axis=1)
        r_R_ratio = np.min(norms)/np.max(norms)
        ratios[i] = r_R_ratio
    avg_ratio = np.mean(ratios)
    return avg_ratio


# return array of vol_sphere/vol_cube ratios
def get_vol_ratios(d):
    v_cube = hypercube_volumes(d)
    v_sphere = hypersphere_volumes(d)
    vol_ratios = np.divide(v_sphere, v_cube)
    return vol_ratios


# Q6: returns the ratio of points out of n samples at most err away from the surface of a hypercube with side_length
def ratio_near_cube_surface(n, d, side_length, err):
    samples = sample_hypercube(n, d, side_length)
    samples_max = np.max(samples, axis=1)
    samples_min = np.min(samples, axis=1)
    num_near_surface = ((1-samples_max)<err).sum()
    return num_near_surface/n


# Q6: returns the ratio of points out of n samples at most err away from the surface of a hypersphere with radius r
# note: for this, the angles don't even matter lol, just use the radii
def ratio_near_sphere_surface(n, d, r, err):
    rad = np.random.rand(n, 1)  # sample uniformly
    rad = (rad**(1./d))*r  # then get the dth root
    num_near_surface = ((r-rad)<err).sum()
    return num_near_surface/n


if __name__ == '__main__':
    T = 15  # number of trials to run
    n = 100  # number of points to draw
    d = 2  # number of dimensions


    # Q2 Part 1: Plot r/R ratios as a function of n
    max_n = 100
    avg_ratios = []
    n_s = range(2, max_n)

    for i in n_s:
        avg_ratios.append(get_avg_ratio(T, i, d))
    plt.plot(n_s, avg_ratios)
    plt.title('average r/R ratio vs. n for d={}, T={}'.format(d, T))
    plt.ylabel('average r/R ratio')
    plt.xlabel('n')
    plt.show()


    # Q2 Part 2: Plot r/R ratios as a function of d = 1...500
    avg_ratios = []
    d_s = list(range(1, 50)) + list(range(50, 100, 10)) + list(range(100, 500, 100))

    for i in d_s:
        avg_ratios.append(get_avg_ratio(T, n, i))
    plt.plot(d_s, avg_ratios)
    plt.title('average r/R ratio vs. d for n={}, T={}'.format(n, T))
    plt.ylabel('average r/R ratio')
    plt.xlabel('d')
    plt.show()


    # Q3: Plot sphere:cube volume ratio as a function of d
    max_d = 50
    vol_ratios = get_vol_ratios(max_d)
    plt.plot(range(1, max_d+1), vol_ratios)
    plt.title('sphere vol/cube vol ratio vs. d')
    plt.ylabel('sphere:cube volume ratio')
    plt.xlabel('d')
    plt.show()


    # Q5: Plot points drawn uniformly from a d2 circle
    samples = sample_hypersphere(100000, 2)
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.show()

    # just for fun (it samples so quickly!!!)
    # samples = sample_hypersphere(10000000, 3)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.set_aspect('equal', adjustable='box')
    # ax.scatter(samples[:,0], samples[:,1], samples[:,2], alpha=0.3)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # plt.show()


    # Q6: Plot |Es|/n and |Ec|/n as a function of d, where Es is the above E for hyperspheres, and Ec is for hypercubes
    epsilon = 0.1
    d_s = range(1, max_d+1)
    cube_ratios = [ratio_near_cube_surface(n, d, 1, epsilon) for d in d_s]
    sphere_ratios = [ratio_near_sphere_surface(n, d, 0.5, epsilon) for d in d_s]

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    fig.set_tight_layout({"pad": 1})

    axs[0].plot(d_s, cube_ratios)
    axs[0].set_title('|Ec|/n vs. d')
    axs[0].set_xlabel('d')
    axs[0].set_ylabel('|Ec|/n')

    axs[1].plot(d_s, sphere_ratios)
    axs[1].set_title('|Es|/n vs. d')
    axs[1].set_xlabel('d')
    axs[1].set_ylabel('|Es|/n')

    plt.show()

