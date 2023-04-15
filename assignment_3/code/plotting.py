import matplotlib.pyplot as plt
import numpy as np
plot="h"
if plot=="h":
    false_positives=np.array([82,86,91,100,100])/100
    h=[1,2,3,5,10]
    plt.xlabel("h")
    plt.ylabel("Estimated false positive probability")

    plt.plot(h,false_positives)
    plt.show()
elif plot =="n":
    false_positives=np.array([100,100,100,94,68])/100
    n=[100000,200000,300000,500000,1000000]
    plt.xlabel("n")
    plt.ylabel("Estimated false positive probability")
    plt.plot(n,false_positives)
    plt.show()