
from scipy.stats.stats import pearsonr
i=4

filt = (magAB >= Lbins[0]) & (magAB < Lbins[1])
filt = (magAB < -19.)

sm = generate_smoothed_fields(t, [sm_res[i]], shells=False, rescale=1.0)
gal21[:,2+i]

plt.hist(sm.flatten(), bins=30, normed=True)
plt.hist(gal21[filt,2+i], bins=30, normed=True,alpha=0.2)

plt.plot(gal21[filt,2+i], magAB[filt], '.')
print(R_list[i+2])
pearsonr(gal21[filt,2+i], magAB[filt])