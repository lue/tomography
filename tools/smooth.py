import numpy as np
from numpy.fft import fftn, ifftn
import numpy as np
import scipy.stats
import accelerate.mkl.fftpack as mklfft

def f_filter(N, r_space, rmax, rmin=0):
    '''
    Generates spherical filter in Fourier space. This method avoids ringing.
    :param N: Box dimension.
    :param r_space: Distances in real space.
    :param rmax: Outer shell of the filter.
    :param rmin: Inner shell of the filter.
    :return: Spherical filter in Fourier space.
    '''
    fshape = np.zeros([N, N, N/2+1])
    fshape[(r_space[:, :, :] >= rmin) & (r_space[:, :, :] < rmax)] = 1.0
    fshape = fshape/np.sum(fshape)
    f_fshape = mklfft.fftn(fshape)
    return f_fshape


def generate_smoothed_fields(reference, r_list, shells=False, rescale=1.0):
    '''
    Applies a list of barriers on a given reference field.
    :param reference: Reference field.
    :param r_list: List of scales.
    :param rescale: rescaling factor.
    :return:
    '''
    # The size of the box.
    N = reference.shape[0]
    # The size of the box.
    N_b = len(r_list)
    # The array for storing resulting map.
    res = np.zeros([N, N, N, N_b], dtype=np.float16)
    # Generating array with distances in real space.
    rx, ry, rz = np.mgrid[:N, :N, :(N/2+1)]
    rx[rx > N/2] = rx[rx > N/2] - N
    ry[ry > N/2] = ry[ry > N/2] - N
    rz[rz > N/2] = rz[rz > N/2] - N
    r_space = np.sqrt(rx**2 + ry**2 + rz**2)
    # Removing all extra arrays to free memory.
    del rx
    del ry
    del rz
    # FFT of the reference field.
    data_fft = mklfft.rfftn(reference)
    # data_fft = np.fft.rfftn(reference)
    # Now iterating through all scales of interest.
    for i in range(len(r_list)):
        print(1.0*i/len(r_list))
        # Creating smoothed array.
        if shells:
            if i > 0:
                rmin = r_list[i-1] * rescale
            else:
                rmin = 0
        else:
            rmin = -1
        # temp = np.fft.irfftn(data_fft*f_filter(N, r_space, r_list[i] * rescale, rmin=rmin))
        temp = mklfft.irfftn(data_fft*f_filter(N, r_space, r_list[i] * rescale, rmin=rmin))
        # Transforming it into N(0,1) distributed array.
        # temp = any2sigma(temp.reshape([-1])).reshape([N, N, N])
        res[:,:,:, i] = temp
    print('Done')
    return res
