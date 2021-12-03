import numpy
import glob
def loadJson(f):                
    import json       
    js = ""                        
    with open(f, "r") as f:
        shouldSave = False
        lines = f.readlines()                                                   
        for l in lines:                                                   
            if l == "{\n":                                                  
                shouldSave = True
            if shouldSave:
                js += l
    return json.loads(js)

d = "./evaluation_results_alexa_100ms_12M_2x/*.json"
ds = []
for f in glob.glob(d):
    try:
        ds.append(loadJson(f))
    except:
        pass

import numpy as np 
none_plts = []        
model_plts = []                    
baseline_plts = []     
for t in ds:          
    baseline_plts.append(np.mean(t["replay_server"]["with_baseline_policy"][0]))
    model_plts.append(np.mean(t["replay_server"]["with_model_policy"][0]))
    none_plts.append(np.mean(t["replay_server"]["without_policy"][0]))



def plot(x1, x1_label, x2, x2_label):
    import numpy as np
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    %matplotlib inline
 
    count_x1, bins_count_x1 = np.histogram(x1, bins=100)
    # finding the PDF of the histogram using count values
    pdf = count_x1 / sum(count_x1)
    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf_x1 = np.cumsum(pdf)

    count_x2, bins_count_x2 = np.histogram(x2, bins=100)
    pdf = count_x2 / sum(count_x2)
    cdf_x2 = np.cumsum(pdf)
    plt.plot(bins_count_x1[1:] * 100, cdf_x1, color="red", label=x1_label)
    plt.plot(bins_count_x2[1:] * 100, cdf_x2, label=x2_label)
    plt.xlabel("% Improvement")
    plt.ylabel("CDF")
    plt.legend()
    plt.savefig("plot.png")

x1 = (1 - np.array(baseline_plts) / np.array(none_plts))
x1.sort()
x1_label = "push/preload all"

x2 = (1 - np.array(model_plts) / np.array(none_plts))
x2.sort()
x2_label = "alohamora"

print(x1, x2)

plot(x1, x1_label, x2, x2_label)