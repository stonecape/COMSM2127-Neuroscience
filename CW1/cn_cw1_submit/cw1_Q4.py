
# coding: utf-8

# In[5]:

import matplotlib.pyplot as plt
import math
from numpy import *
from pylab import *

T = 1000
dt = 1
timearray = arange(0, T, dt)

# LTF properties
Vm = zeros(len(timearray))
T_m = 10
E_L = -70
V_r = -70
V_t = -40
R_m = 10

I_array = arange(2, 5, 0.1)
spike_count = zeros(len(I_array))


for j, i_e in enumerate(I_array):
    count = 0
    Vm = zeros(len(timearray))
    for i, t in enumerate(timearray):
        if i == 0:
            Vm[i] = V_r
        else:
            Vm[i] = Vm[i-1] + (E_L - Vm[i-1] + i_e * R_m) / T_m * dt
            if Vm[i] > V_t:
                Vm[i] = V_r
                count = count + 1
    spike_count[j] = count

plt.plot(I_array, spike_count)
plt.xlabel("Input Current")
plt.ylabel("Firing Rate")
plt.title("Q4: F-I Curve")

plt.show()



# In[ ]:



