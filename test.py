

import os
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import rasterio
import torch
matplotlib.use('Agg')
#from glob import glob
plt.figure()
plt.plot([0],[1])
plt.savefig('plot.png')

plt.show()

print(torch.cuda.is_available())