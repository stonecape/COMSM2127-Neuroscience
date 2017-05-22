
# coding: utf-8

# In[3]:

from pylab import *


# In[4]:

def doSynapticSimulation(E_s):
    T = 1000
    dt = 1
    timearray = arange(0, T, dt)
    # LTF properties
    V1 = zeros(len(timearray))
    V2 = zeros(len(timearray))
    T_m = 20
    E_L = -70
    V_r = -80
    V_t = -54
    R_m_I_e = 18
    
    # Synapses
    S1 = 0
    S2 = 0
    R_m_g_s = 0.15
    P = 0.5
    T_s = 10
    
    for i, t in enumerate(timearray):
        if i == 0:
            V1[0] = randint(V_r, V_t)
            V2[0] = randint(V_r, V_t)
            print("V1[0]=", V1[0], " V2[0]=", V2[0])
        else:
            V1[i] = V1[i-1] + (E_L - V1[i-1] + R_m_I_e + R_m_g_s * S2 * (E_s-V1[i-1])) / T_m * dt
            V2[i] = V2[i-1] + (E_L - V2[i-1] + R_m_I_e + R_m_g_s * S1 * (E_s-V2[i-1])) / T_m * dt
            S2 = S2 + (-S2 * dt) / T_s
            S1 = S1 + (-S1 * dt) / T_s
            if V1[i] > V_t:
                V1[i] = V_r
                S1 = S1 + P

            if V2[i] > V_t:
                V2[i] = V_r
                S2 = S2 + P
            
    plt.plot(timearray, V1, label='1 st neuron')
    plt.plot(timearray, V2, label='2 nd neuron')

    plt.xlabel("t(ms)")
    plt.ylabel("v(mv)")
    plt.title("Q5: Synaptic Connection Simulation E_s=" + str(E_s))

    plt.show()


# In[5]:

doSynapticSimulation(0)
doSynapticSimulation(-80)


# In[ ]:



