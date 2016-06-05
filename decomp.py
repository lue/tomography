import numpy as np
import matplotlib.pyplot as plt
from tools.io import *
from tools.smooth import *

a_list = ['a=0.']

N1,N2,N3,t = readifrit('E:\REI\Cai.B80.N512L2.sf=1_uv=0.15_bw=10_res=200.NMA\B\IFRIT\ifrit-a=0.1000.bin', nvar=4, moden=2, skipmoden=2)

# plt.imshow(np.log10(t[:,:,0]), interpolation='nearest')


halos = np.genfromtxt('E:\REI\Cai.B80.N512L2.sf=1_uv=0.15_bw=10_res=200.NMA\B\\a=0.1000\halo_catalog_a0.1000.dat', comments='#', dtype=None)
Lcat = np.genfromtxt('E:\REI\Cai.B80.N512L2.sf=1_uv=0.15_bw=10_res=200.NMA\B\\a=0.1000\gallums.res', comments='#', dtype=None)

x = halos['f3']/80.*1024.
y = halos['f2']/80.*1024.
z = halos['f1']/80.*1024.

pos = np.zeros([len(Lcat), 3])
for i in range(len(Lcat)):
    ind = np.searchsorted(halos['f0'], Lcat['f0'][i])
    pos[i, :] = [x[ind], y[ind], z[ind]]
    print(ind)


plt.imshow(np.log10(t[:,:,0].T), origin='lower', interpolation='nearest', cmap='jet')
plt.plot(pos[(pos[:,2]<15),0], pos[(pos[:,2]<15),1], '.k')


#########################

MAB0 = -2.5 * np.log10(3631.e-23)
pc_cm = 3.08568025e+18
const = 4. * np.pi * (10. * pc_cm)**2.
magAB = -2.5 * np.log10(Lcat['f4']*4e33/const) - MAB0
plt.hist(magAB,100); plt.yscale('log')

##########################

gal21 = np.zeros([len(Lcat),10])

#####

pos_r = np.floor(pos).astype(int)
gal21[:,0] = t[pos_r[:,0],pos_r[:,1],pos_r[:,2]]

t = downsample(t)

pos /= 2.0
pos_r = np.floor(pos).astype(int)
gal21[:,1] = t[pos_r[:,0],pos_r[:,1],pos_r[:,2]]

sm_res = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
for i in range(len(sm_res)):
    print(1.*i/len(sm_res))
    sm = generate_smoothed_fields(t, [sm_res[i]], shells=False, rescale=1.0)
    gal21[:,2+i] = sm[pos_r[:,0],pos_r[:,1],pos_r[:,2]].flatten()


R_list = 2**np.arange(10)*80/1024

II = 6
LNbins = 9
Lbins = np.percentile(magAB, np.linspace(0, 1, LNbins)*100)
BNbins = 9
Bbins = np.percentile(gal21[:,II], np.linspace(0, 1, BNbins)*100)
H = np.histogram2d(magAB, gal21[:,II], bins=[Lbins, Bbins])
img = H[0] / len(Lcat) * (LNbins-1) * (BNbins-1)
img = (img - 1.0) * 100.

fig = plt.figure()
rect = 0.2, 0.1, 0.8, 0.8
ax1 = fig.add_axes(rect)
temp = ax1.imshow(img, extent=[0,100,0,100], interpolation='nearest', cmap='bwr', vmin=-20, vmax=20, origin='lower', aspect='auto')
# ax1.yaxis.tick_right()
# ax1.yaxis.set_label_position('right')
# ax1.xaxis.tick_top()
# ax1.xaxis.set_label_position('top')
plt.ylabel('Luminosity (percentiles)')
plt.xlabel('21cm brightness (percentiles)')
plt.colorbar(temp)
plt.title('z=10 \n Smoothing scale: '+str(R_list[II])+' Mpc/h')

# ax2 = fig.add_axes(rect, frameon=False)
# plt.xlim([0,1])
Bbins_labels = ["%2.1f\n%d"%(Bbins[i]/8/8*1000, 100.*i/(len(Bbins)-1))+'%' for i in range(len(Bbins))]
plt.xticks(np.linspace(0,100,BNbins), Bbins_labels, rotation='horizontal')
# plt.ylim([0,1])
Lbins_labels = ["%2.1f\n%d"%(Lbins[i], 100.*i/(len(Lbins)-1))+'%' for i in range(len(Lbins))]
plt.yticks(np.linspace(0,100,LNbins), Lbins_labels, rotation='horizontal')
plt.ylabel('Luminosity [erg/s/Hz]')
plt.xlabel('21cm brightness [mK]')

