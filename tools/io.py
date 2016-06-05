import numpy as np

def readifrit(path, nvar=0, moden=2, skipmoden=2):
    import struct
    import numpy as np
    openfile=open(path, "rb")
    dump = np.fromfile(openfile, dtype='>i', count=moden)
    N1,N2,N3=np.fromfile(openfile, dtype='>i', count=3)
    print(N1,N2,N3)
    dump = np.fromfile(openfile, dtype='>i', count=moden)
    data = np.zeros([N1, N2, N3])
    j = 0
    for i in range(nvar):
        openfile.seek(4*skipmoden, 1)
        for j in range(4):
            openfile.seek(N1*N2*N3, 1)
        openfile.seek(4*moden, 1)
    openfile.seek(4*skipmoden, 1)
    data[:, :, :] = np.reshape(np.fromfile(openfile, dtype='>f4', count=N1 * N2 * N3), [N1, N2, N3])
    openfile.close()
    return N1,N2,N3,data


def downsample(data,f=2):
    data_new=(data[::2, ::2, ::2]+data[1::2, ::2, ::2]+
              data[::2, 1::2, ::2]+data[::2, ::2, 1::2]+
              data[1::2, 1::2, ::2]+data[::2, 1::2, 1::2]+
              data[1::2, ::2, 1::2]+data[1::2, 1::2, 1::2])
    return data_new