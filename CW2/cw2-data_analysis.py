
# coding: utf-8

# In[272]:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import operator
from pandas import Series, DataFrame, Panel

get_ipython().magic('matplotlib inline')


# In[273]:

x_data = pd.read_csv("x.csv",header = None)
x_data.columns = ["x_coordinate"]

y_data = pd.read_csv("y.csv",header = None)
y_data.columns = ["y_coordinate"]

time_data = pd.read_csv("time.csv",header = None)
time_data.columns = ["time"]

neuron1_data = pd.read_csv("neuron1.csv",header = None)
neuron1_data.columns = ["spike_time"]

neuron2_data = pd.read_csv("neuron2.csv",header = None)
neuron2_data.columns = ["spike_time"]

neuron3_data = pd.read_csv("neuron3.csv",header = None)
neuron3_data.columns = ["spike_time"]

neuron4_data = pd.read_csv("neuron4.csv",header = None)
neuron4_data.columns = ["spike_time"]


# In[274]:

merged_t_x_y = pd.concat([time_data, x_data, y_data], axis=1)
indexed_merged_t_x_y = merged_t_x_y.set_index(['time'])


# In[275]:

def find_nearest(spiketime):
    index = indexed_merged_t_x_y.index.get_loc(spiketime, method='nearest')
    return merged_t_x_y.iloc[index]['time']


# In[276]:

def filter_spike_coordinates(neuron_data):
    spike_coordinates = pd.DataFrame( columns=['time'])
    spike_coordinates['time'] = neuron_data['spike_time'].apply(find_nearest) 
    merged = pd.merge(spike_coordinates, merged_t_x_y, on='time', how='left')
    return merged


# In[277]:

def firing_rates_plot(neuron_data, number):
    spiketime_list = neuron_data['spike_time'].tolist()
    sec = []
    for v in spiketime_list:
        sec.append(int(v/10000))
    
    bin_num = int(max(spiketime_list) / 10000)-int(min(spiketime_list) / 10000)
    title = "Firing rates (Neuron %d)" % number
    
    plt.figure()
    plt.hist(sec, bins=bin_num, histtype='bar')
    plt.title(title)
    plt.xlabel("Sec")
    plt.ylabel("Firing rates")
    plt.savefig('firing_rate_%d'%number)
    plt.show()


# In[278]:

def autocorrelation_plot(neuron_data, number):
    diff = []
    spiketime_list = neuron_data['spike_time'].tolist()
    for v in spiketime_list:
        for k in spiketime_list:
           if  abs(v-k) <=10000 and abs(v-k) > 0:
            diff.append(float(v-k)/10000)  

    title = "Auto-correlogram (Neuron %d)" % number
    plt.figure()
    plt.hist(diff, bins=100, histtype='bar')
    y,binEdges = np.histogram(diff, bins=100)
    bincenters = 0.5*(binEdges[1:] + binEdges[:-1])
    plt.plot(bincenters,y,'-')
    plt.title(title)
    plt.ylabel("coefficient")
    plt.xlabel("sec")
    plt.savefig('auto_%d'%number)
    plt.show()


# In[279]:

def crosscorrelation_plot(neuron_data1, number1, neuron_data2, number2):
    diff = []
    spiketime_list1 = neuron_data1['spike_time'].tolist()
    spiketime_list2 = neuron_data2['spike_time'].tolist()
    for v in spiketime_list1:
        for k in spiketime_list2:
            if  abs(v-k) <=10000:
                diff.append(float(v-k)/10000)
    
    title = "Cross-correlogram (Neuron %d and Neuron %d)" % (number1, number2)
    plt.figure(figsize=(8,4))
    plt.hist(diff, bins=100, histtype='bar')
    y,binEdges = np.histogram(diff, bins=100)
    bincenters = 0.5*(binEdges[1:] + binEdges[:-1])
    #plt.plot(bincenters,y,'-')
    plt.title(title)
    plt.ylabel("coefficient")
    plt.xlabel("sec")
    plt.savefig('cross_%d_%d' %(number1, number2))
    plt.show()


# In[280]:

def plot_spike_position(x_list,y_list,number):
    title = "Positions where neuron %d fired" % number
    plt.figure()
    plt.scatter(x_list,y_list)
    plt.title(title)
    plt.ylabel("y_coordinate")
    plt.xlabel("x_coordinate")
    plt.savefig('position_%d' % number)
    plt.show()


# ### Task1: Generate plots showing positions in which each neuron fired.

# In[281]:

neuron1_positions = filter_spike_coordinates(neuron1_data)
neuron2_positions = filter_spike_coordinates(neuron2_data)
neuron3_positions = filter_spike_coordinates(neuron3_data)
neuron4_positions = filter_spike_coordinates(neuron4_data)


# In[282]:

plot_spike_position(neuron1_positions['x_coordinate'],neuron1_positions['y_coordinate'],1)
plot_spike_position(neuron2_positions['x_coordinate'],neuron2_positions['y_coordinate'],2)
plot_spike_position(neuron3_positions['x_coordinate'],neuron3_positions['y_coordinate'],3)
plot_spike_position(neuron4_positions['x_coordinate'],neuron4_positions['y_coordinate'],4)


# ### Task2:Plot auto-correlograms of neurons

# In[283]:

autocorrelation_plot(neuron1_data,1)
autocorrelation_plot(neuron2_data,2)
autocorrelation_plot(neuron3_data,3)
autocorrelation_plot(neuron4_data,4)


# ### Task3:Plot cross-correlograms of pairs of neurons

# In[284]:

#demo
crosscorrelation_plot(neuron1_data,1, neuron2_data,2)
crosscorrelation_plot(neuron1_data,1, neuron3_data,3)
crosscorrelation_plot(neuron1_data,1, neuron4_data,4)

crosscorrelation_plot(neuron2_data,2, neuron3_data,3)
crosscorrelation_plot(neuron2_data,2, neuron4_data,4)

crosscorrelation_plot(neuron3_data,3, neuron4_data,4)


# ### Task4:Calculate firing rates of each neuron, and plot histograms of the firing rates

# In[285]:

firing_rates_plot(neuron1_data,1)
firing_rates_plot(neuron2_data,2)
firing_rates_plot(neuron3_data,3)
firing_rates_plot(neuron4_data,4)


# In[ ]:




# In[ ]:



