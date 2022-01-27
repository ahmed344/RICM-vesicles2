# Libraries
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image


# Modules
from RICM_vesicle import RICM

# Read the image
img = io.imread("data/AVG_20161025_dsDNA_SC_36nm_Conc_6.00nM_WOC_Sample_D_03_RICMS_00-1.tif")

# Define the RICM class with certain n_inner
ricm = RICM(img, n_inner = 1.345)

Mask = ricm.mask()

I_norm = ricm.background_normalization()

h = np.linspace(1, 200, 600)
Height = ricm.height(h)

#plt.imshow(Height, cmap = 'gray')
#plt.colorbar()
                    
I_norm_img = []
h_img = []


for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if Mask[i,j] == 1:     # only the contact zone
            I_norm_img.append(I_norm[i,j])
            h_img.append(Height[i,j])


# display results
plt.figure(figsize=(10,5))
            
plt.plot(h_img, I_norm_img, label = "image contact zone")
plt.plot(h, ricm.R5_norm(h), label = "model $R_{norm}(h)$")
plt.title("$R_{norm}$ for different $h$ and $n_{inner} = 1.34$ ")
plt.xlabel("$h_{nm}$")
plt.ylabel("Observed Intensity")
plt.legend()
plt.grid()

# Save the image
plt.savefig('results/img_vs model')
