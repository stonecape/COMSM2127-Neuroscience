
# coding: utf-8

# In[1054]:

from pylab import *
import random


# In[1055]:

T = 200 # at least 200 or 300 seconds
dt = 1e-3 # 1ms
V_rest = -65e-3 #-65mV
V_reset = -65e-3 #-65mV
V_th = -50e-3 #-50mV
R_m = 100e6 #100MOmega
C_m = 0.1e-9 # 0.1nF

Syn_num = 40 # 40 synapses
A_add = 0.1e-9#0.1nS
A_sub = 0.12e-9#0.12nS
tau_add = tau_sub = 20e-3
upper_g_syn =  2e-9 #upper limitation for g_syn
lower_g_syn = 0 #lower limitation for g_syn
T_s = 2e-3 #2ms
E_s = 0 # excitatory
P = 1

r_0 = 15 #Hz
f = 10 #Hz

output_firing_rate_count=[]

eg_synapse_index = 19
eg_synapse_spike_time=[]
post_spike_time=[]


# In[1056]:

# T = 200 # at least 200 or 300 seconds
# dt = 1e-3 # 1ms
# timearray = arange(0, T, dt)
# sec = []
# # LTF properties

# V = zeros(len(timearray))
# V_rest = -65e-3 #-65mV
# V_reset = -65e-3 #-65mV
# V_th = -50e-3 #-50mV
# R_m = 100e6 #100MOmega
# C_m = 0.1e-9 # 0.1nF

# # Synapses
# Syn_num = 40 # 40 synapses
# S = zeros(Syn_num)
# G = zeros(Syn_num)
# count_spike_syn = zeros(Syn_num)
# T_s = 2e-3 #2ms
# E_s = 0 # excitatory
# P = 1

# # STDP
# stdp_flag = True
# A_add = 0.1e-9#0.1nS
# A_sub = 0.12e-9#0.12nS
# tau_add = tau_sub = 20e-3
# last_spike_time = zeros(Syn_num)
# neuron_spike_time = 0
# #r = 10 # Hz
# #spike_th = r * dt # the threshold for a spike
# upper_g_syn =  2e-9 #upper limitation for g_syn
# lower_g_syn = 0 #lower limitation for g_syn

# output_firing_rate_count=[]


# In[1057]:

def simulate_poisson_spikes(currentTime):
    if(sep_flag):
        # q5
        for i in range(len(last_spike_time)):
            if(i in range(0,int(len(last_spike_time)/2))):
                # group 1 B > 0
                para = B
            else:
                # group 2 B = 0
                para = 0
            rvalue = random.uniform(0,1)
            r_y = r_0 + para * sin(2*pi*f*currentTime)
            thr = r_y * dt
            
            #print('f=',f,' para=',para,' thr=',thr, ' currentTime=',currentTime)
            
            if(rvalue < thr):
                if(i == eg_synapse_index):
                    eg_synapse_spike_time.append(currentTime)

                count_spike_syn[i] = count_spike_syn[i] + 1
                last_spike_time[i] = currentTime
                update_single_g_syn(i)
                S[i] = S[i] + P
            else:
                S[i] = S[i] + (-S[i] * dt) / T_s
    else:       
        for i in range(len(last_spike_time)):
            rvalue = random.uniform(0,1)
            r_y = r_0 + B * sin(2*pi*f*currentTime)
            thr = r_y * dt

            
            if(rvalue < thr):
                if(i == eg_synapse_index):
                    eg_synapse_spike_time.append(currentTime)

                count_spike_syn[i] = count_spike_syn[i] + 1
                last_spike_time[i] = currentTime
                update_single_g_syn(i)
                S[i] = S[i] + P
            else:
                S[i] = S[i] + (-S[i] * dt) / T_s


# In[ ]:




# In[1058]:

def cal_ft(t_post, t_pre):
    if(stdp_flag):
        delta_t = t_post - t_pre
        if(delta_t > 0):
            return A_add * exp(-abs(delta_t)/tau_add)
        else:
            return -A_sub * exp(-abs(delta_t)/tau_sub)
    else:
        return 0


# In[1059]:

# when a spike occured in postsynaptic neuron
def update_global_g_syn():
    for i in range(len(G)):
        
        G[i] = G[i] + cal_ft(neuron_spike_time, last_spike_time[i])
        
        if(G[i] < lower_g_syn):
            G[i] = lower_g_syn
        
        if(G[i] > upper_g_syn):
            G[i] = upper_g_syn
        
# when a spike occured in presynaptic neuron (synapses)
def update_single_g_syn(syn_num):
    G[syn_num] = G[syn_num] + cal_ft(neuron_spike_time, last_spike_time[syn_num])
    if(G[syn_num] < lower_g_syn):
        G[syn_num] = lower_g_syn

    if(G[syn_num] > upper_g_syn):
        G[syn_num] = upper_g_syn


# In[1060]:

def do_simulation_main(): 
    global neuron_spike_time
    print('V_rest->',V_rest)
    print('V_reset->',V_reset)
    print('V_th->',V_th)
    print('R_m->',R_m)
    print('C_m->',C_m)
    print('Syn_num->',Syn_num)
    print('T_s->',T_s)
    print('A_add->',A_add)
    print('A_sub->',A_sub)
    print('tau_add->',tau_add)
    print('tau_sub->',tau_sub)
    print('r_0->',r_0)
    print('STDP status->',stdp_flag)
    print('initial G->',G)
    print('initial S->',S)
    print('initial V->',V[0:100])
    print('B->',B)
    count_spike = 0
    for i, time in enumerate(timearray):
        simulate_poisson_spikes(time)
        if i == 0:
            V[i] = V_rest
        elif (V[i-1] > V_th):
            V[i] = V_reset
        else:
            I_e = sum(G * S * V[i-1])
            dV = ((V_rest - V[i-1]) / (R_m * C_m) - (I_e / (C_m))) * dt
            V[i] = V[i-1] + dV

            if V[i] > V_th:
                sec.append(int(time))
                post_spike_time.append(time)
                count_spike = count_spike + 1
                neuron_spike_time = time
                update_global_g_syn()

    print('count spike for postsynaptic neuron:',count_spike)
    print('count spike for each synapse:',count_spike_syn)
    output_firing_rate_count.append(count_spike)


# In[1061]:

def set_parameter(stdp_flag_i, initial_g_flag, r_hz, B_v, sep_f):
    
    global timearray,sec,V,S,G,count_spike_syn,stdp_flag,last_spike_time,neuron_spike_time,spike_th,output_firing_rate_count,B,r_0,sep_flag

    sep_flag = sep_f
    timearray = arange(0, T, dt)
    sec = []
    # LTF properties

    V = zeros(len(timearray))
    # Synapses
    S = zeros(Syn_num)
    G = zeros(Syn_num)
    count_spike_syn = zeros(Syn_num)

    # STDP
    stdp_flag = stdp_flag_i
    last_spike_time = zeros(Syn_num)
    neuron_spike_time = 0
    #spike_th = r_hz * dt # the threshold for a spike
    r_0 = r_hz
    
    B = B_v

    
    # initialize G
    for i in range(len(G)):
        if(initial_g_flag):
            G[i] = np.random.uniform(0,2)*10**-9# initial peak conductances chosed from the uniform distribution 
        else:
            G[i] = 2*10**-9
            #G[i] = 8.82840513145e-10


# In[1062]:

# set_parameter(True, False,10,0)
# do_simulation_main()
# plt.figure()
# plt.plot(timearray,V)
# plt.show()


# ### Question 1

# In[1063]:

# set_parameter(False, False,15,0)
# do_simulation_main()
# print('G->',G)
# print('mean of G->',mean(G))

# title1 = 'a histogram of the steady-state synaptic weights (STDP:off)'
# plt.figure()
# plt.hist(G, bins=50, histtype='bar')
# plt.title(title1)
# plt.ylabel("count")
# plt.xlabel("synaptic weights (S)")
# plt.savefig('q1_his_steady-state-g_syn-off(gsy_mean_initialize)')
# plt.show()


# In[1064]:

# set_parameter(True, False,15)
# do_simulation_main()
# print('G->',G)
# print('mean of G->',mean(G))

# title1 = 'a histogram of the steady-state synaptic weights (STDP:on)'
# plt.figure()
# plt.hist(G, bins=50, histtype='bar')
# plt.title(title1)
# plt.ylabel("count")
# plt.xlabel("synaptic weights (S)")
# plt.savefig('q1_his_steady-state-g_syn-on')
# plt.show()


# In[1065]:

# set_parameter(True, False,15,0)
# do_simulation_main()
# print('G->',G)
# print('mean of G->',mean(G))
# title = "firing rate of the postsynaptic neuron (STDP:on)"

# plt.figure()
# plt.hist(sec, bins=T, histtype='bar')
# plt.title(title)
# plt.xlabel("Sec")
# plt.ylabel("Firing rates")
# plt.savefig('q1_firing_rate_pos-on')
# plt.show()


# In[1066]:

# title1 = 'a histogram of the steady-state synaptic weights (STDP:on)'
# plt.figure()
# plt.hist(G, bins=50, histtype='bar')
# plt.title(title1)
# plt.ylabel("count")
# plt.xlabel("synaptic weights (S)")
# plt.savefig('q1_his_steady-state-g_syn-on')
# plt.show()


# In[1067]:

# set_parameter(False, False,15,0)
# do_simulation_main()
# title = "firing rate of the postsynaptic neuron (STDP:off)"

# plt.figure()
# plt.hist(sec, bins=T, histtype='bar')
# plt.title(title)
# plt.xlabel("Sec")
# plt.ylabel("Firing rates")
# plt.savefig('q1_firing_rate_pos-off(gsy_mean_initialize)')
# plt.show()


# ### Question 2

# In[1068]:

# for r in range(10,21):
#     set_parameter(False, False,r)
#     do_simulation_main()

# print(output_firing_rate_count)


# In[1069]:

# print(output_firing_rate_count)
# array_count = np.array(output_firing_rate_count)/T
# print(array_count)
# t = np.arange(10, 21, 1)
# plt.figure()
# plt.plot(t,array_count)
# plt.title('Mean Output Firing Rate (STDP:off)')
# plt.xlabel("Input Firing Frequency(Hz)")
# plt.ylabel("Mean Output Firing Rate ")
# plt.savefig('q2_firing_rate_off')
# plt.show()


# In[1070]:

# for r in range(10,21):
#     set_parameter(True, False,r,0)
#     do_simulation_main()

# print(output_firing_rate_count)


# In[1071]:

# print(output_firing_rate_count)
# array_count = np.array(output_firing_rate_count)/T
# print(array_count)
# t = np.arange(10, 21, 1)
# plt.figure()
# plt.plot(t,array_count)
# plt.title('Mean Output Firing Rate (STDP:on)')
# plt.xlabel("Input Firing Frequency(Hz)")
# plt.ylabel("Mean Output Firing Rate ")
# plt.savefig('q2_firing_rate_on')
# plt.show()


# In[1072]:

# set_parameter(True, False,10,0)
# do_simulation_main()
# plt.figure()
# plt.hist(G, bins=20, histtype='bar')
# plt.title('synaptic strength distribution (10Hz STDP:on)')
# plt.ylabel("count")
# plt.xlabel("synaptic weights (S)")
# plt.savefig('q2_10Hz_synaptic_strength_distribution')
# plt.show()


# In[1073]:

# set_parameter(True, False,20,0)
# do_simulation_main()
# plt.figure()
# plt.hist(G, bins=20, histtype='bar')
# plt.title('synaptic strength distribution (20Hz STDP:on)')
# plt.ylabel("count")
# plt.xlabel("synaptic weights (S)")
# plt.savefig('q2_20Hz_synaptic_strength_distribution')
# plt.show()


# ### Question 3

# In[1074]:

# mean_g_syn = []
# std_g_syn = []
# for b in range(0,16):
#     set_parameter(True, False,15,b)
#     do_simulation_main()
#     print('G->',G)
#     print('mean G->',mean(G))
#     print('std G->',std(G))
#     mean_g_syn.append(mean(G))
#     std_g_syn.append(std(G))


# In[1075]:

# b = np.arange(0,16, 1)
# plt.figure()
# plt.plot(b,mean_g_syn)
# plt.title('Mean Of The Synaptic Strengths')
# plt.xlabel("B value")
# plt.ylabel("Mean Of The Synaptic Strengths")
# plt.savefig('q3_mean_g_syn')
# plt.show()


# In[1076]:

# b = np.arange(0,16, 1)
# plt.figure()
# plt.plot(b,std_g_syn)
# plt.title('Std Of The Synaptic Strengths')
# plt.xlabel("B value")
# plt.ylabel("Std Of The Synaptic Strengths")
# plt.savefig('q3_std_g_syn')
# plt.show()


# In[1077]:

# set_parameter(True, False,15,0)
# do_simulation_main()
# plt.figure()
# plt.hist(G, bins=20, histtype='bar')
# plt.title('synaptic strength distribution (B=0 Hz)')
# plt.ylabel("count")
# plt.xlabel("synaptic weights (S)")
# plt.savefig('q3_B-0Hz_synaptic_strength_distribution')
# plt.show()


# In[1078]:

# set_parameter(True, False,15,15)
# do_simulation_main()


# In[1079]:

# plt.figure()
# plt.hist(G, bins=20, histtype='bar')
# plt.title('synaptic strength distribution (B=15 Hz)')
# plt.ylabel("count")
# plt.xlabel("synaptic weights (S)")
# plt.savefig('q3_B-15Hz_synaptic_strength_distribution')
# plt.show()


# ### Question 4

# In[1080]:

# stdp off; randomly initialize G off
# r_0 = 15; B = 0
# set_parameter(False, False,15,0)
# do_simulation_main()


# In[1081]:

# diff = []
# for v in eg_synapse_spike_time:
#         for k in post_spike_time:
#             diff.append(int(v-k))
# print(len(diff))
# title = "Cross-correlogram (STDP: off)"
# plt.figure()
# plt.hist(diff, bins=50, histtype='bar')
# plt.title(title)
# plt.ylabel("coefficient")
# plt.xlabel("sec")
# plt.savefig('q4_cross_corr_off')
# plt.show()


# In[1082]:

# set_parameter(True, False,15,0)
# do_simulation_main()
# print('eg_synapse_spike_time->',eg_synapse_spike_time)
# print('post_spike_time->',post_spike_time)


# In[1083]:

# diff = []
# for v in eg_synapse_spike_time:
#         for k in post_spike_time:
#             diff.append(int(v-k))
# print(len(diff))
# title = "Cross-correlogram (STDP: on)"
# plt.figure()
# plt.hist(diff, bins=50, histtype='bar')
# plt.title(title)
# plt.ylabel("coefficient")
# plt.xlabel("sec")
# plt.savefig('q4_cross_corr_on')
# plt.show()


# In[1084]:

# group 1: 0-19
# mean_g_syn_1 = []#B>0
# mean_g_syn_2 = []#B=0

# for b in range(1,16):
#     set_parameter(True, False,15,b,True)
#     do_simulation_main()
#     print('G->',G)
#     print('G[:int(Syn_num/2)->',G[:int(Syn_num/2)])
#     print('G[int(Syn_num/2):]->',G[int(Syn_num/2):])
#     mean_g1 = mean(G[:int(Syn_num/2)])
#     mean_g2 = mean(G[int(Syn_num/2):])
#     print('mean G1->',mean_g1)
#     print('mean G2->',mean_g2)
#     mean_g_syn_1.append(mean(mean_g1))
#     mean_g_syn_2.append(mean(mean_g2))


# In[1085]:

# b = np.arange(1,16, 1)
# plt.figure()
# plt.plot(b,mean_g_syn_1)
# plt.plot(b,mean_g_syn_2)
# plt.legend(['group 1 (B>0)', 'group 2 (B=0)'])
# plt.title('Mean Of The Synaptic Strengths')
# plt.xlabel("B value")
# plt.ylabel("Mean Of The Synaptic Strengths")
# plt.savefig('q5_mean_g_syn_two_groups')
# plt.show()


# In[1086]:

set_parameter(True, False,15,15,True)
do_simulation_main()


# In[1087]:

plt.figure()
plt.hist(G[0:int(Syn_num/2)], bins=20, histtype='bar')
plt.title('synaptic strength distribution (group 1 B=15)')
plt.ylabel("count")
plt.xlabel("synaptic weights (S)")
plt.savefig('q5_synaptic_strength_distribution_group1')
plt.show()

plt.figure()
plt.hist(G[int(Syn_num/2):], bins=20, histtype='bar')
plt.title('synaptic strength distribution (group 2 B=0)')
plt.ylabel("count")
plt.xlabel("synaptic weights (S)")
plt.savefig('q5_synaptic_strength_distribution_group2')
plt.show()


# In[ ]:



