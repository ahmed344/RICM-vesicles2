# Libraries
import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration, filters, morphology
from scipy import ndimage, linalg, optimize
from FITTING import Fit_Gaussian


class Height_map():
    
    def __init__(self,
                 n_glass = 1.525,
                 n_water = 1.333,
                 n_outer = 1.335,
                 n_lipid = 1.486,
                 n_inner = 1.344,
                 d_water = 1,
                 d_lipid = 4,
                 l       = 546,
                 p       = 0):
       
        # Parameters
        self.n_glass = n_glass      # refractive index of glass 
        self.n_water = n_water      # refractive index of water
        self.n_outer = n_outer      # refractive index of outer solution (PBS)
        self.n_lipid = n_lipid      # refractive index of lipid
        self.n_inner = n_inner      # refractive index of inner buffer (Sucrose)
        self.d_water = d_water      # thikness of water in nm
        self.d_lipid = d_lipid      # thikness of lipid in nm
        self.l       = l            # wave length of the RICM light in nm
        self.p       = p            # phase shift of the cosine function
        

    # Normalized reflactance for 5 interfaces
    def i5_norm(self, h):

        # Wave vector
        k = (2 * np.pi) / self.l

        # Refractive indices
        n0 = self.n_glass    # Glass slid
        n1 = self.n_water    # Layer of water down the SLB
        n2 = self.n_lipid    # SLB 
        n3 = self.n_outer    # Outer buffer PBS
        n4 = self.n_lipid    # Vesicle membrane
        n5 = self.n_inner    # Inner buffer Sucrose

        # Fresnel reflection coefficients
        r01 = (n0-n1) / (n0+n1)
        r12 = (n1-n2) / (n1+n2)
        r23 = (n2-n3) / (n2+n3)
        r34 = (n3-n4) / (n3+n4)
        r45 = (n4-n5) / (n4+n5)

        # Distances traveled by light
        D1 = 2 * n1 * self.d_water
        D2 = 2 * n2 * self.d_lipid
        D3 = 2 * n3 * h
        D4 = 2 * n4 * self.d_lipid  
        
        # Effective reflection coefficient
        R1 = r01
        R2 = ((1-r01**2) * np.exp(-1j*k*D1)) * r12
        R3 = ((1-r01**2)*(1-r12**2) * np.exp(-1j*k*(D1+D2))) * r23
        R4 = ((1-r01**2)*(1-r12**2)*(1-r23**2) * np.exp(-1j*k*(D1+D2+D3))) * r34
        R5 = ((1-r01**2)*(1-r12**2)*(1-r23**2)*(1-r34**2) * np.exp(-1j*k*(D1+D2+D3+D4))) * r45
        
        # Effective reflection coefficient of the adhesion zone
        R = R1 + R2 + R3 + R4 + R5

        # Effective reflection coefficient of the background
        R_b = R1 + R2 + R3

        # Normalized reflactance R_norm
        R_norm = (np.abs(R * np.conjugate(R)) - np.abs(R_b * np.conjugate(R_b))) / np.abs(R_b * np.conjugate(R_b))

        return R_norm

    
    # The dependence of the normalized intensity on hight
    def normalized_intensity(self, h, Y0, A, h0):
        
        n_outer = self.n_outer  #refractive index of PBS
        l = self.l              #wave length of the RICM light

        return Y0 - A * np.cos((4 * np.pi * n_outer / l) * (h - h0) + 2*np.pi*self.p)
    


class RICM(Height_map):
    
    def __init__(self,
                 img,
                 denoise=True, nl_fast_mode=True, nl_patch_size=10, nl_patch_distance=1,
                 hole=3, remove_small=True, min_size=64,
                 n_glass=1.525, n_water=1.333, n_outer=1.335, n_lipid=1.486, n_inner=1.344,
                 d_water=1, d_lipid=4, l=546, p=0):
        
        # The image
        self.img = img
        
        # RICM parameters
        self.n_glass = n_glass      # refractive index of glass 
        self.n_water = n_water      # refractive index of water
        self.n_outer = n_outer      # refractive index of outer solution (PBS)
        self.n_lipid = n_lipid      # refractive index of lipid
        self.n_inner = n_inner      # refractive index of inner buffer (Sucrose)
        self.d_water = d_water      # thikness of water in nm
        self.d_lipid = d_lipid      # thikness of lipid in nm
        self.l = l                  # wave length of the RICM light in nm
        self.p = p                  # phase shift of the cosine function

        # Denoising parameters
        self.denoise = denoise
        self.nl_fast_mode = nl_fast_mode
        self.nl_patch_size = nl_patch_size
        self.nl_patch_distance = nl_patch_distance
        
        # Mask parameters
        self.hole = hole                     # hole filling kernel
        self.remove_small = remove_small     # remove small defects
        self.min_size = min_size             # minimum size for a small defect
        
        
    # Denoise the image using Non-local means denoising algorithm
    def nl_denoise(self):
        # Apply the Non-local means denoising algorithm and return the denoised image
        return restoration.denoise_nl_means(self.img,
                                            h = np.mean(restoration.estimate_sigma(self.img)),
                                            fast_mode = self.nl_fast_mode,
                                            patch_size = self.nl_patch_size,
                                            patch_distance = self.nl_patch_distance)
        


    # Detecting the edges
    def edge_detection(self):
        # Check if denoising is True
        if self.denoise:
            # Return the edge of the denoised image
            return filters.sobel(RICM.nl_denoise(self))
        else:
            # Return the edge of the original image
            return filters.sobel(self.img)

    
    # Determine the contact zone by filling the closed edges inside the binary image of the edges
    def mask(self):

        #Applying some edge operators to the denoised image
        edge = RICM.edge_detection(self)
        
        # 1- Getting the threshold of edge filtered image
        # 2- Making a binary image with 0 and 1 values
        # 3- Fill the detected edge
        # 4- Remove the small objects in case of remove small is True
        
        if self.remove_small: 
            return morphology.remove_small_objects(ndimage.binary_fill_holes(np.multiply(edge > filters.threshold_otsu(edge), 1),
                                                                             structure=np.ones((self.hole, self.hole))),
                                                   min_size=self.min_size)

        return ndimage.binary_fill_holes(np.multiply(edge > filters.threshold_otsu(edge), 1),
                                         structure=np.ones((self.hole, self.hole)))


    # Fitting the background
    def background_fitting(self):

        # Determine the contact zone
        edge_binary_filled = RICM.mask(self)

        # Write the data in terms of 3-dim points excluding the contact zone
        coord_background_intensity = []
        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                if edge_binary_filled[i, j] == False:  # excluding the contact zone
                    coord_background_intensity.append([i, j, self.img[i,j]])

        # 3-dim data points
        data = np.array(coord_background_intensity)

        # best-fit quadratic curve
        A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
        C,_,_,_ = linalg.lstsq(A, data[:,2])

        # Copy the original img
        Background = np.ones(self.img.shape)

        # Fill the bacground with the values came from the fitting
        for X in range(self.img.shape[0]):
            for Y in range(self.img.shape[1]):
                Background[X,Y] = C[0] + C[1]*X + C[2]*Y + C[3]*X*Y + C[4]*X*X + C[5]*Y*Y

        return Background


    # Correcting the image by subtracting the background then add it's average to each pixel
    def correct(self):

        # Fitting the background
        Background = RICM.background_fitting(self)
        
        # Return the corrected image
        return self.img - Background + np.average(Background)

    
    # Normalized reflactance to the background
    def background_normalization(self):

        # Get the corrected image
        img_corrected = RICM.correct(self)

        # Get the background by removing the contact zone from the corrected image
        corrected_background = img_corrected * (1 - RICM.mask(self))

        # Transform it into a histogram excluding the contact zone
        corrected_background = corrected_background[corrected_background != 0].ravel()
        
        # Fit a gaussian on the corrected background histogram then take it's mean
        avg_corrected_background, _ = Fit_Gaussian(corrected_background, normalized=True).hist_fitting()

        return (img_corrected - avg_corrected_background) / avg_corrected_background
    
    
    # RICM height mapping
    def height(self, h=np.linspace(1, 600, 600)):
        
        # Fit the parameters Y_0, A, h_0 of the cosine function
        mapping = Height_map(n_glass = self.n_glass,
                             n_water = self.n_water,
                             n_outer = self.n_outer,
                             n_lipid = self.n_lipid,
                             n_inner = self.n_inner,
                             d_water = self.d_water,
                             d_lipid = self.d_lipid,
                             l       = self.l,
                             p       = self.p)
        
        popt, _ = optimize.curve_fit(mapping.normalized_intensity, h, mapping.i5_norm(h))
        Y0, A, h0 = popt
        print('Y0 = {:.2f}, A = {:.2f}, h0 = {:.2f}'.format(*popt))

        return (self.l/(4*np.pi*self.n_outer)) * (np.arccos((Y0 - RICM.background_normalization(self)) / A) - 2*np.pi*self.p) + h0    
    
    
    # RICM height mapping argument
    def height_argument(self, h=np.linspace(1, 600, 600)):

        # Fit the parameters Y_0, A, h_0 of the cosine function
        mapping = Height_map(n_glass = self.n_glass,
                             n_water = self.n_water,
                             n_outer = self.n_outer,
                             n_lipid = self.n_lipid,
                             n_inner = self.n_inner,
                             d_water = self.d_water,
                             d_lipid = self.d_lipid,
                             l       = self.l,
                             p       = self.p)
        
        popt, _ = optimize.curve_fit(mapping.normalized_intensity, h, mapping.i5_norm(h))
        Y0, A, _ = popt

        return (Y0 - RICM.background_normalization(self)) / A 
    
    
    # Display the way to the RICM height mapping step by step
    def show_summary(self, name='summary', save=False):
        
        plt.figure(figsize=(20,8))

        plt.subplot(241)
        plt.axis('off')
        plt.title('Orginal image')
        plt.imshow(self.img, cmap = "gray")
        plt.colorbar()

        plt.subplot(242)
        plt.axis('off')
        plt.title('Denoised image')
        plt.imshow(RICM.nl_denoise(self) , cmap = 'gray')
        plt.colorbar()

        plt.subplot(243)
        plt.axis('off')
        plt.title('Edge detected image')
        plt.imshow(RICM.edge_detection(self) , cmap = 'gray')
        plt.colorbar();

        plt.subplot(244)
        plt.title('Orginal histogram')
        plt.hist(self.img.ravel(), bins = 200)
        plt.grid();
        
        plt.subplot(245)
        plt.axis('off')
        plt.title('Masked image')
        plt.imshow(RICM.mask(self) , cmap = 'gray')
        plt.colorbar();

        plt.subplot(246)
        plt.axis('off')
        plt.title('Background fitted image')
        plt.imshow(RICM.background_fitting(self) , cmap = 'gray')
        plt.colorbar();

        plt.subplot(247)
        plt.axis('off')
        plt.title('Corrected image')
        plt.imshow(RICM.correct(self) , cmap = 'gray')
        plt.colorbar()
        
        plt.subplot(248)
        plt.title('Corrected histogram')
        plt.hist(RICM.correct(self).ravel(), bins = 200)
        plt.grid();
        
        # Save the image
        if save or name!='summary':
            plt.savefig(name)

        # Show the results
        plt.show()


class Growth_Area():
    
    def __init__(self, movie, background=None, static_threshold=False,
                 denoise=False, nl_fast_mode=True, nl_patch_size=10, nl_patch_distance=1, 
                 consecute=None, keep_dim =True, show_dim=True):
       
        # Parameters
        self.movie            = movie
        self.background       = background
        self.static_threshold = static_threshold
        
        # Denoising parameters
        self.denoise           = denoise
        self.nl_fast_mode      = nl_fast_mode
        self.nl_patch_size     = nl_patch_size
        self.nl_patch_distance = nl_patch_distance
        
        # Consecuting parameters
        self.consecute = consecute
        self.keep_dim  = keep_dim
        self.show_dim  = show_dim

    # Normalized reflactance for 5 interfaces
    def consecuted_movie(self):
        
        # Average each consecute frames
        movie_consecuted = []
        if self.keep_dim:
            for i in range(int(self.movie.shape[0]-self.movie.shape[0]%self.consecute-self.consecute)):
                movie_consecuted.append(np.mean(self.movie[i:i+self.consecute], axis=0))
        else:
            for i in np.arange(int(self.movie.shape[0]-self.movie.shape[0]%self.consecute), step=self.consecute):
                movie_consecuted.append(np.mean(self.movie[i:i+self.consecute], axis=0))
        
        # Transform the movie into numpy array
        movie_consecuted = np.array(movie_consecuted)
        
        # Show the dimension reduction
        if self.show_dim:
            print(f"{self.movie.shape} --> {movie_consecuted.shape}")
        
        return movie_consecuted

    def area_curve(self):

        # Background correction of the movie
        if self.background is None:
            background_correction = 0
        else: 
            background_correction = self.background.mean()-self.background

        # Check if consecutivity is ordered
        if self.consecute != None:

            # Consicutive denoising of the movie
            movie_consecuted = Growth_Area.consecuted_movie(self)

            # Non local denoising of the movie
            if self.denoise:

                # Get the area of each frame in the averaged movie
                movie_corrected = []
                for img in movie_consecuted:

                    # Apply the Non-local means denoising algorithm on the background corrected image
                    img_corrected = restoration.denoise_nl_means(img + background_correction,
                                                                 h = np.mean(restoration.estimate_sigma(movie_consecuted[-1])),
                                                                 fast_mode = self.nl_fast_mode,
                                                                 patch_size = self.nl_patch_size,
                                                                 patch_distance = self.nl_patch_distance)

                    # Corrected movie
                    movie_corrected.append(img_corrected)

                movie_corrected = np.array(movie_corrected)

            else: 
                movie_corrected = movie_consecuted + background_correction

        else:

            # Non local denoising of the movie
            if self.denoise:

                # Get the area of each frame in the averaged movie
                movie_corrected = []
                for img in self.movie:

                    # Apply the Non-local means denoising algorithm on the background corrected image
                    img_corrected = restoration.denoise_nl_means(img + background_correction,
                                                                 h = np.mean(restoration.estimate_sigma(self.movie[-1])),
                                                                 fast_mode = self.nl_fast_mode,
                                                                 patch_size = self.nl_patch_size,
                                                                 patch_distance = self.nl_patch_distance)

                    # Corrected movie
                    movie_corrected.append(img_corrected)

                movie_corrected = np.array(movie_corrected)

            else: 
                movie_corrected = self.movie + background_correction
        
        
        if self.static_threshold:
            # Compute the threshold
            threshold = filters.threshold_otsu(movie_corrected[-20:-1].mean(axis = 0))

            # Return the area
            return np.array([(1-np.multiply(img > threshold, 1)).sum() for img in movie_corrected])
        else:
            # Return the area computed with dynamic threshold
            return np.array([(1-np.multiply(img > filters.threshold_otsu(img), 1)).sum() for img in movie_corrected])