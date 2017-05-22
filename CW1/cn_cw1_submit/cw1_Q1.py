
# coding: utf-8

# In[2]:

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

I_e = 3.1
I_array = arange(2, 5, 0.1)
spike_count = zeros(len(I_array))

# Q1
for i, t in enumerate(timearray):
    if i == 0:
        Vm[i]=V_r
    else:      
        Vm[i] = Vm[i-1] + (E_L - Vm[i-1] + I_e * R_m) / T_m * dt
        if Vm[i] > V_t:
            Vm[i] = V_r

plt.plot(timearray, Vm)
plt.xlabel("t(ms)")
plt.ylabel("v(mv)")
plt.title("Q1: Leaky Integrate-and-Fire I_e=3.1")

plt.show()



# In[ ]:



