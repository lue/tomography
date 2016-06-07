import numpy as np
import matplotlib.pyplot as plt
from tools.io import *
from tools.smooth import *

a_list =     ['0.0905','0.1000','0.1118','0.1208','0.1281']
xi_list =    ['99.5'  , '97.5' , '73.3' ,'26.9',  '3.16']
a_list_cal = ['0.0905','0.1000','0.1118','0.1118','0.1281']
a_i = 4

N1,N2,N3,t = readifrit('E:\REI\Cai.B80.N512L2.sf=1_uv=0.15_bw=10_res=200.NMA\B\IFRIT\ifrit-a=' + a_list[a_i] + '.bin', nvar=4, moden=2, skipmoden=2)

# plt.imshow(np.log10(t[:,:,0]), interpolation='nearest')
# plt.colorbar()

halos = np.genfromtxt('E:\REI\Cai.B80.N512L2.sf=1_uv=0.15_bw=10_res=200.NMA\B\\a=' + a_list_cal[a_i] + '\halo_catalog_a' + a_list_cal[a_i] + '.dat', comments='#', dtype=None)
Lcat = np.genfromtxt('E:\REI\Cai.B80.N512L2.sf=1_uv=0.15_bw=10_res=200.NMA\B\\a=' + a_list_cal[a_i] + '\gallums.res', comments='#', dtype=None)

x = halos['f3']/80.*1024.
y = halos['f2']/80.*1024.
z = halos['f1']/80.*1024.

pos = np.zeros([len(Lcat), 3])
for i in range(len(Lcat)):
    ind = np.searchsorted(halos['f0'], Lcat['f0'][i])
    pos[i, :] = [x[ind], y[ind], z[ind]]
    print(ind)


# plt.imshow(np.log10(t[:,:,0].T), origin='lower', interpolation='nearest', cmap='jet')
# plt.plot(pos[(pos[:,2]<15),0], pos[(pos[:,2]<15),1], '.k')


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
t /= 8

pos /= 2.0
pos_r = np.floor(pos).astype(int)
gal21[:,1] = t[pos_r[:,0],pos_r[:,1],pos_r[:,2]]

sm_res = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
for i in range(len(sm_res)):
    print(1.*i/len(sm_res))
    sm = generate_smoothed_fields(t, [sm_res[i]], shells=False, rescale=1.0)
    gal21[:,2+i] = sm[pos_r[:,0],pos_r[:,1],pos_r[:,2]].flatten()


R_list = 2**np.arange(10)*80/1024

for II in [0,1,2,3,4,5,6,7,8]:
    LNbins = 17
    Lbins = np.percentile(magAB, np.linspace(0, 1, LNbins)*100)
    BNbins = 11
    Bbins = np.percentile(gal21[:,II], np.linspace(0, 1, BNbins)*100)
    H = np.histogram2d(magAB, gal21[:,II], bins=[Lbins, Bbins])
    img = H[0] / len(Lcat) * (LNbins-1) * (BNbins-1)
    img = (img - 1.0) * 100.
    fig = plt.figure()
    rect = 0.2, 0.15, 0.8, 0.75
    ax1 = fig.add_axes(rect)
    temp = ax1.imshow(img,
                      extent=[0,100,0,100],
                      interpolation='nearest',
                      cmap='bwr',
                      vmin=-np.ceil(np.max(np.abs(img))), vmax=np.ceil(np.max(np.abs(img))),
                      origin='lower', aspect='auto')
    # ax1.yaxis.tick_right()
    # ax1.yaxis.set_label_position('right')
    # ax1.xaxis.tick_top()
    # ax1.xaxis.set_label_position('top')
    plt.ylabel('Luminosity (percentiles)')
    plt.xlabel('21cm brightness (percentiles)')
    plt.colorbar(temp)
    plt.title(('z=%2.1f xHI='+xi_list[a_i]+'\nSmoothing scale: %1.2f Mpc/h')%(1.0/float(a_list[a_i])-1., R_list[II]))
    # ax2 = fig.add_axes(rect, frameon=False)
    # plt.xlim([0,1])
    Bbins_labels = ["%2.1f\n%d"%(Bbins[i]*1000, 100.*i/(len(Bbins)-1))+'%' for i in range(len(Bbins))]
    plt.xticks(np.linspace(0,100,BNbins), Bbins_labels, rotation='horizontal', size=8)
    # plt.ylim([0,1])
    Lbins_labels = ["%2.1f\n%d"%(Lbins[i], 100.*i/(len(Lbins)-1))+'%' for i in range(len(Lbins))]
    plt.yticks(np.linspace(0,100,LNbins), Lbins_labels, rotation='horizontal', size=8)
    plt.ylabel('Magnitude AB')
    plt.xlabel('21cm brightness [mK]')
    plt.savefig(('images/z%02.1f_%02.2fmpch_2'%(1.0/float(a_list[a_i])-1., R_list[II])).replace('.','_')+'.png', dpi=200,bbox_inches='tight')
    plt.close('all')

