
# coding: utf-8

# In[1]:

from pylab import *

T = 1000
dt = 1
timearray = arange(0, T, dt)


# In[2]:

# LTF properties
Vm = zeros(len(timearray))
T_m = 10
E_L = -70
V_r = -70
V_t = -40
R_m = 10

E_k = -80
dgk = 0.005
g_k = 0
T_k = 200


I_e = 3.1
I_array = arange(2, 5, 0.1)


# In[3]:

for i, t in enumerate(timearray):
    if i == 0:
        Vm[i] = V_r
    else:
        Vm[i] = Vm[i-1] + (E_L - Vm[i-1] + I_e * R_m + R_m * g_k * (E_k - Vm[i-1])) / T_m * dt
        g_k = g_k + (-g_k * dt / T_k)

        if Vm[i] > V_t:
            Vm[i] = V_r
            g_k = g_k + dgk


# In[4]:

plt.plot(timearray, Vm)
plt.xlabel("t(ms)")
plt.ylabel("v(mv)")
plt.title("Q6: Potassium Current Simulation")

plt.show()


# In[ ]:



