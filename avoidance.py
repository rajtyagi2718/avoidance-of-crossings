
import numpy as np
import matplotlib.pyplot as plt
import os

DATA_PATH = os.getcwd() + '/data/'

# RGB colors from Tableau
TABLEAU20 = [( 31,119,180), (174,199,232), (255,127, 14), (255,187,120),
             ( 44,160, 44), (152,223,138), (214, 39, 40), (255,152,150),
             (148,103,189), (197,176,213), (140, 86, 75), (196,156,148),
             (227,119,194), (247,182,210), (127,127,127), (199,199,199),
             (188,189, 34), (219,219,141), (23, 190,207), (158,218,229)]

# RGB scaled to [0,1] for matplotlib
TABLEAU20 = [(r/255, g/255, b/255) for x in TABLEAU20 for r,g,b in (x,)]

def get_symmetric_matrix(size=5, scale=100):
    """Return random symmetric matrix of real values between -scale, scale."""
    # matrix with random values between (-scale, scale)
    result = scale/2 * np.random.rand(size, size)

    # symmetrize
    d = result.diagonal()
    result += result.T
    np.fill_diagonal(result, d)

    return result

def get_matrices(deg=5, size=22, scale=100):
    result = np.zeros((deg, size, size))
    for i in range(deg):
        result[i] = get_symmetric_matrix(size, scale)
    return result

def get_times(deg=5, start=-5, end=5, steps=1001):
    result = np.zeros((deg, steps))
    result[0,:] = 1
    result[1] = np.linspace(start, end, steps)
    for i in range(2, deg):
        result[i] = result[0] ** i
    return result

def get_eigenfunction_values(matrices, times):
    """Return matrix of eigenvalues of matrix polynomial function for t in T."""
    #result[i,j] = ith eigenvalue of A0 + A1*t + A2*t**2 +... at time t = T[j].
    result = np.zeros((matrices[0].shape[0], times.shape[1]))

    for i in range(times.shape[1]):
        F = sum(A * t for A,t in zip(matrices, times[:,i]))
        result[:,i] = np.linalg.eigh(F)[0]

    return result

def plot_values(values, T):
    """Return plot of T vs values."""
    plt.figure(figsize=(16,12))
    plt.title('Eigenfunctions', y=.97, fontsize=16)
    plt.xticks(np.linspace(np.min(values[:-1]), np.max(values[:-1]), 15), fontsize=10)
    plt.yticks(np.linspace(T[0], T[-1], 11), fontsize=10)

    ax = plt.subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Time', fontsize=12)

    # plot all but smallest and largest eigenfunctions
    for i in range(1,values.shape[0]-1):
        plt.plot(values[i], T, color=TABLEAU20[i-1], lw=3)

def get_plots(num=10):
    for i in range(num):
        matrices = get_matrices()
        times = get_times()
        values = get_eigenfunction_values(matrices, times)
        plot_values(values, times[1])
        np.save(DATA_PATH + 'matrices%d.npy' %i, matrices)
        np.save(DATA_PATH + 'times%d.npy' %i, times)
        np.save(DATA_PATH + 'values%d.npy' %i, values)
        plt.savefig(DATA_PATH + 'plot%d.png' %i, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    get_plots()
