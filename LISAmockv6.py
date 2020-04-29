# Standard useful python module
import numpy as np
import sys, os
import time
# Adding the LDC packages to the python path
sys.path.append("/root/.local/lib/python3.6/site-packages")
# LDC modules
from LISAhdf5 import LISAhdf5,ParsUnits
import tdi
import FastGB as FB
import LISAConstants as LC
# Plotting modules
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

# Specify the hdf5 file path
hdf5_path = "/home/ntan032/Documents/LISAmock/LDC1-6_SGWB_v1_noiseless.hdf5"
# Open the file with the dedicated LISAhdf5 module
FD5 =  LISAhdf5('LDC1-6_SGWB_v1_noiseless.hdf5','r')
# Number of sources
Nsrc = FD5.getSourcesNum()
# Types of sources
GWs = FD5.getSourcesName()
print ("Found %d GW sources: " % Nsrc, GWs)
# Get source parameters
p = FD5.getSourceParameters(GWs[0])
print(p)
# Get the time delay interferometry data
td = FD5.getPreProcessTDI()
print(td)
# Sampling time
#del_t = float(p.get("Cadence"))
# Observation duration
#Tobs = float(p.get("ObservationDuration"))
# Display the source parameters
#p.display()
#///////////////////////////////////////////////////////////////////////////////

# to avoid Pandas warning:
# df1 = pd.DataFrame({'Time': td[:Npts,0], 'TDI X': td[:Npts,1], 'TDI Y': td[:Npts,2], 'TDI Z': td[:Npts,3]})

Npts = 2**22
print(np.shape(td))
df1 = pd.DataFrame({'Time': td[:Npts,0], 'TDI X': td[:Npts,1], 'TDI Y': td[:Npts,2], 'TDI Z': td[:Npts,3]})
df1 = pd.DataFrame(td[:Npts,:],columns=['Time','TDI X', 'TDI Y', 'TDI Z'])
plt.figure(1)
df1.plot(x = 'Time',y = ['TDI X', 'TDI Y', 'TDI Z'])
plt.xlabel("Time [s]")
plt.ylabel("Fractional frequency")
plt.legend(loc='best')
plt.show()

#////////////////////////////////////////////////////////////////////////////////

## Create a new data matrix for fourier-transformed data
td_freq = np.zeros(np.shape(td),dtype = np.complex128)
## Tapering window to mitigate leakage
wind = np.hamming(Npts)
## Take discrete Fourier transform of the noiseless data
#td_freq[:,1:] = np.array( [ np.fft.fft(td[:Npts,i]*wind,axis = 0) for i in range(1,4)] ).T
## Compute corresponding frequency vector
#freq = np.fft.fftfreq(Npts)/del_t
feq = np.fft.fftfreq(Npts)/td[:Npts,0]
#td_freq[:,0] = freq
## Store the amplitude modulus in data frame (positive frequencies)
#td_freq_plot = np.zeros(np.shape(td_freq[freq>0,:]),dtype = np.float64) 
#td_freq_plot[:,0] = freq[freq>0]
#td_freq_plot[:,1:] = np.abs(td_freq[freq>0,1:])*2/np.sum(wind)
#df2 = pd.DataFrame(td_freq_plot,columns=['freq','TDI X', 'TDI Y', 'TDI Z'])
## Plot the Fourier amplitudes
#plt.figure(2)
#df2.plot(x = 'freq',y = ['TDI X', 'TDI Y', 'TDI Z'],logx=True,logy=True)
#plt.xlabel("Frequency [Hz]")
#plt.ylabel("Fractional frequency")
#plt.xlim(1e-3,1e-2)
#plt.legend(loc='best')

#////////////////////////////////////////////////////////////////////////////////

# Duplicate the hdf5 file
hdf5_path_TD = hdf5_path[:-5]+"_TD.hdf5"
FD5_TD =  LISAhdf5(hdf5_path_TD)
FD5_TD.addSource(GWs[0], p, overwrite=True)
# os.system("cp  " + hdf5_path + "  " + hdf5_path_TD )
#////////////////////////////////////////////////////////////////////////////////
FD5_TD =  LISAhdf5(hdf5_path_TD)
dTDI_TD = FD5_TD.getPreProcessTDI()
#df4 = pd.DataFrame(dTDI_TD,columns=['t','TDI X (TD)', 'TDI Y (TD)', 'TDI Z (TD)'])
df4 = pd.DataFrame(td,columns=['t','TDI X (TD)', 'TDI Y (TD)', 'TDI Z (TD)'])
plt.figure(3)
df4.plot(x = 't',y = ['TDI X (TD)'])
#plt.xlabel("Time [s]")
#plt.ylabel("Fractional frequency")
#plt.legend(loc='best')
plt.show()

#////////////////////////////////////////////////////////////////////////////////

# Create a new data matrix for fourier-transformed data
dTDI_TD_fft = np.zeros(np.shape(td[:Npts,:]),dtype = np.complex128)
# Take discrete Fourier transform of the data
dTDI_TD_fft[:,1:] = np.array( [ np.fft.fft(td[:Npts,i]*wind,axis = 0) for i in range(1,4)] ).T
# Compute corresponding frequency vector
#freq = np.fft.fftfreq(Npts)/del_t
freq = np.fft.fftfreq(Npts)/td[:Npts,0]
dTDI_TD_fft[:,0] = freq
# Store the amplitude modulus in data frame (positive frequencies)
dTDI_TD_fft_plot = np.zeros(np.shape(dTDI_TD_fft[freq>0,:]),dtype = np.float64)  
dTDI_TD_fft_plot[:,0] = freq[freq>0]
dTDI_TD_fft_plot[:,1:] = np.abs(dTDI_TD_fft[freq>0,1:])*2/np.sum(wind)
df5 = pd.DataFrame(dTDI_TD_fft_plot,columns=['freq','TDI X (TD)', 'TDI Y (TD)', 'TDI Z (TD)'])

df3 = pd.DataFrame(np.abs(dTDI_FD),columns=['f','TDI X (FD)', 'TDI Y (FD)', 'TDI Z (FD)'])

plt.figure(4)
ax4 = df5.plot(x = 'freq',y = 'TDI X (TD)',logx = True,logy=True)
df3.plot(x = 'f', y = 'TDI X (FD)',ax = ax4, logx = True,logy=True,style='--')
plt.xlabel("Frequency [Hz]")
plt.ylabel("Fractional frequency")
plt.xlim(1e-3,1e-2)
plt.legend(loc='best')










