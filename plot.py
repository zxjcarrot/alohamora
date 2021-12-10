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

d = "./alexa_top_500_eval_results_20ms_24M_1x_2/*.json"
ds = []
for f in glob.glob(d):
    try:
        ds.append(loadJson(f))
    except:
        pass

import numpy as np 
manifests = []
none_plts = []     
none_cts = []   
model_plts = []                    
model_inference_plts = []                    
baseline_plts = []
inference_overhead = []
for t in ds: 
    if np.mean(t["replay_server"]["without_policy"][0]) <= 0:
        continue
    manifests.append(t["manifest"])         
    baseline_plts.append(np.mean(t["replay_server"]["with_baseline_policy"][0]))
    model_plts.append(np.mean(t["replay_server"]["with_model_policy"][0]))
    
    model_inference_plts.append(np.mean(t["replay_server"]["with_model_policy_and_inference"][1]))
    none_plts.append(np.mean(t["replay_server"]["without_policy"][0]))
    none_cts.append(np.mean(t["replay_server"]["without_policy"][0]))
    inference_overhead.append(t["average_model_inference_time"])


print("none_cts", none_cts)
good_manifests = [site for (site, ct) in list(zip(manifests, none_cts)) if ct > 0]
print(good_manifests)
def plot(x1, x1_label, x2, x2_label, x3, x3_label):
    import numpy as np
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    #%matplotlib inline
 
    count_x1, bins_count_x1 = np.histogram(x1, bins=100)
    # finding the PDF of the histogram using count values
    pdf = count_x1 / sum(count_x1)
    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf_x1 = np.cumsum(pdf)

    count_x2, bins_count_x2 = np.histogram(x2, bins=100)
    pdf = count_x2 / sum(count_x2)
    cdf_x2 = np.cumsum(pdf)

    count_x3, bins_count_x3 = np.histogram(x3, bins=100)
    # finding the PDF of the histogram using count values
    pdf = count_x3 / sum(count_x3)
    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf_x3 = np.cumsum(pdf)


    plt.plot(bins_count_x1[1:] * 100, cdf_x1, color="red", label=x1_label)
    plt.plot(bins_count_x2[1:] * 100, cdf_x2, label=x2_label)
    #@plt.plot(bins_count_x3[1:] * 100, cdf_x3, color="orange", label=x3_label)
    plt.xlabel("% Improvement")
    plt.ylabel("CDF")
    plt.legend()
    plt.savefig("plot.png")

x1 = (1 - np.array(baseline_plts) / np.array(none_plts))
x1.sort()
x1_label = "push/preload all"

x2 = (1 - np.array(model_plts) / np.array(none_plts))

manifests_sorted = [x for _, x in sorted(zip(x2, manifests))]

print("manifests_sorted: ", manifests_sorted)
x2.sort()
x2_label = "alohamora"



t = (np.array(model_inference_plts) / np.array(none_plts))
x3 = 1 - t
x3.sort()
x3_label = "alohamora+inference-overhead"


print(x1,x2, x3)

print("push-preload-all median/95th PLT improvements: ", np.mean(x1), np.percentile(x1, 95))
print("Alohamora median/95th PLT improvements: ", np.mean(x2), np.percentile(x2, 95))

print("median inference overhead:", np.mean(inference_overhead))
print("95th percentile inference overhead:", np.percentile(inference_overhead, 95))
plot(x1, x1_label, x2, x2_label, x3, x3_label)
