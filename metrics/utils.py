import numpy as np
from scipy.signal import convolve2d
import torch
import torchvision.transforms.functional as gaussFilter

class SCIELAB_filter:
    # apply S-CIELAB filtering
    
    XYZ_to_opp_mat = (
            (0.2787,0.7218,-0.1066),
            (-0.4488,0.2898,0.0772),
            (0.0860,-0.5900,0.5011) )

    def __init__(self, device):
        # Since the matrices are needed for multiple calls, move it to device only once
        self.XYZ_to_opp_mat = torch.as_tensor(self.XYZ_to_opp_mat, device=device)
        self.opp_to_XYZ_mat = torch.inverse(self.XYZ_to_opp_mat)
        self.device = device

    def xyz_to_opp(self, img):
        OPP = torch.empty_like(img)
        for cc in range(3):
            OPP[...,cc,:,:,:] = torch.sum(img*(self.XYZ_to_opp_mat[cc,:].view(1,3,1,1,1)), dim=-4, keepdim=True)
        return OPP
        
    def opp_to_xyz(self, img):
        XYZ = torch.empty_like(img)
        for cc in range(3):
            XYZ[...,cc,:,:,:] = torch.sum(img*(self.opp_to_XYZ_mat[cc,:].view(1,3,1,1,1)), dim=-4, keepdim=True)
        return XYZ
    
    def gauss(self, halfWidth, width):
        # Returns a 1D Gaussian vector.  The gaussian sums to one.
        # The halfWidth must be greater than one.
        # The halfwidth specifies the width of the gaussian between the points
        # where it obtains half of its maximum value.  The width indicates the gaussians width in pixels.

        alpha = 2 * np.sqrt(np.log(2)) / (halfWidth - 1)
        x = np.linspace(1, width, width)
        x = x - np.round(width / 2)

        t = x ** 2
        g = np.exp(-alpha ** 2 * t)
        g = g / np.sum(g)
        return g
        
    def gauss_torch(self, halfWidth, width):
        # Returns a 1D Gaussian vector.  The gaussian sums to one.
        # The halfWidth must be greater than one.
        # The halfwidth specifies the width of the gaussian between the points
        # where it obtains half of its maximum value.  The width indicates the gaussians width in pixels.

        alpha = 2 * np.sqrt(np.log(2)) / (halfWidth - 1)
        x = torch.linspace(1, width, width)
        x = x - torch.round(width / 2)

        t = x ** 2
        g = torch.exp(-alpha ** 2 * t)
        g = g / torch.sum(g)
        return g
     
    def resize(self, orig, newSize, align=[0, 0], padding=0):
        # result = resize(orig, newSize, align, padding)
        #
        # if newSize is larger than orig size, pad with padding
        # if newSize is smaller than orig size, truncate to fit.
        # align specifies alignment. 0=centered
        #                           -1=left (up) aligned
        #                            1=right (down) aligned
        # For example, align=[0 -1] centers on rows (y) and left align on columns (x).
        #              align=1 aligns left on columns and top on rows.

        if len(newSize) == 1:
            newSize = [newSize, newSize]
        if len(align) == 1:
            align = [align, align]

        if len(orig.shape) == 1:  # 1D array
            orig = orig.reshape(-1, 1)  # make it to (mx1) array

        [m1, n1] = orig.shape
        m2 = newSize[0]
        n2 = newSize[1]
        m = np.minimum(m1, m2)
        n = np.minimum(n1, n2)

        result = np.ones((m2, n2)) * padding

        start1 = np.array([np.floor((m1 - m) / 2 * (1 + align[0])), np.floor((n1 - n) / 2 * (1 + align[1]))]) + 1
        start2 = np.array([np.floor((m2 - m) / 2 * (1 + align[0])), np.floor((n2 - n) / 2 * (1 + align[1]))]) + 1

        t1 = np.int_(np.linspace(start2[0], start2[0] + m - 1, m) - 1)
        t2 = np.int_(np.linspace(start2[1], start2[1] + n - 1, n) - 1)
        t3 = np.int_(np.linspace(start1[0], start1[0] + m - 1, m) - 1)
        t4 = np.int_(np.linspace(start1[1], start1[1] + n - 1, n) - 1)

        result[t1[:,np.newaxis],t2] = orig[t3[:,np.newaxis],t4]
        # trickly to broadcast 2D matrix
        # https://towardsdatascience.com/numpy-indexing-explained-c376abb2440d

        # result[np.int_(np.linspace(start2[0], start2[0] + m - 1, m) - 1), np.int_(
        #     np.linspace(start2[1], start2[1] + n - 1, n) - 1)] = \
        #     orig[np.int_(np.linspace(start1[0], start1[0] + m - 1, m) - 1), np.int_(
        #         np.linspace(start1[1], start1[1] + n - 1, n) - 1)]

        return result
        
    def resize_torch(self, orig, newSize, align=[0, 0], padding=0):
        # result = resize(orig, newSize, align, padding)
        #
        # if newSize is larger than orig size, pad with padding
        # if newSize is smaller than orig size, truncate to fit.
        # align specifies alignment. 0=centered
        #                           -1=left (up) aligned
        #                            1=right (down) aligned
        # For example, align=[0 -1] centers on rows (y) and left align on columns (x).
        #              align=1 aligns left on columns and top on rows.

        if len(newSize) == 1:
            newSize = [newSize, newSize]
        if len(align) == 1:
            align = [align, align]

        if len(orig.shape) == 1:  # 1D array
            orig = torch.reshape(orig,(-1, 1))  # make it to (mx1) array

        [m1, n1] = orig.shape
        m2 = newSize[0]
        n2 = newSize[1]
        m = np.minimum(m1, m2)
        n = np.minimum(n1, n2)

        result = torch.ones((m2, n2), device=orig.device) * padding

        start1 = np.array([np.floor((m1 - m) / 2 * (1 + align[0])), np.floor((n1 - n) / 2 * (1 + align[1]))]) + 1
        start2 = np.array([np.floor((m2 - m) / 2 * (1 + align[0])), np.floor((n2 - n) / 2 * (1 + align[1]))]) + 1

        t1 = np.int_(np.linspace(start2[0], start2[0] + m - 1, m) - 1)
        t2 = np.int_(np.linspace(start2[1], start2[1] + n - 1, n) - 1)
        t3 = np.int_(np.linspace(start1[0], start1[0] + m - 1, m) - 1)
        t4 = np.int_(np.linspace(start1[1], start1[1] + n - 1, n) - 1)

        result[t1[:,np.newaxis],t2] = orig[t3[:,np.newaxis],t4]
        # trickly to broadcast 2D matrix
        # https://towardsdatascience.com/numpy-indexing-explained-c376abb2440d

        # result[np.int_(np.linspace(start2[0], start2[0] + m - 1, m) - 1), np.int_(
        #     np.linspace(start2[1], start2[1] + n - 1, n) - 1)] = \
        #     orig[np.int_(np.linspace(start1[0], start1[0] + m - 1, m) - 1), np.int_(
        #         np.linspace(start1[1], start1[1] + n - 1, n) - 1)]

        return result
        
    def conv2(self, x, y, mode='full'):
        # While Matlab's conv2 results in artifacts on the bottom and right of an image,
        # scipy.signal.convolve2d has the same artifacts on the top and left of an image.
        return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)
        
    def conv2_torch(self, x, y, mode='full'):
        # While Matlab's conv2 results in artifacts on the bottom and right of an image,
        # scipy.signal.convolve2d has the same artifacts on the top and left of an image.
        # Ignore these differences, they are much smaller than inaccuracy between torch and scipy

        # torch.nn.functional.conv2d actually computes cross-correlation
        # flip the kernel to get convolution
        # Order of errors (w.r.t scipy) is ~5e-7 when the input values are in range(0, 170).
        # TODO: Investigate further if precision is important
        x = x.view(1,1,*x.shape)
        y = y.flip(dims=(0,1)).view(1,1,*y.shape)
        return torch.nn.functional.conv2d(x, y, padding=[dim-1 for dim in y.shape[-2:]]).squeeze()
    
    def separableFilters(self, sampPerDeg):
        # not full implementation but correct for S-CIELAB usage.
        # Please refer to original Matlab version

        # if sampPerDeg is smaller than minSAMPPERDEG, need to upsample image data before filtering.
        # This can be done equivalently by convolving filters with the upsampling matrix, then downsample it.
        minSAMPPERDEG = 224
        dimension=3

        if sampPerDeg < minSAMPPERDEG:
            uprate = int(np.ceil(minSAMPPERDEG / sampPerDeg))
            sampPerDeg = sampPerDeg * uprate
        else:
            uprate = 1
        
        # these are the same filter parameters, except that the weights are normalized to sum to 1 
        # This eliminates the need to normalize after the filters are generated
        x1 = np.array([0.05, 1.00327, 0.225, 0.114416, 7.0, -0.117686])
        x2 = np.array([0.0685, 0.616725, 0.826, 0.383275])
        x3 = np.array([0.0920, 0.567885, 0.6451, 0.432115])

        # Convert the unit of halfwidths from visual angle to pixels.
        x1[[0, 2, 4]] = x1[[0, 2, 4]] * sampPerDeg
        x2[[0, 2]] = x2[[0, 2]] * sampPerDeg
        x3[[0, 2]] = x3[[0, 2]] * sampPerDeg

        # Limit filter width to 1-degree visual angle, and odd number of sampling points
        # (so that the gaussians generated from Rick's gauss routine are symmetric).
        width = int(np.ceil(sampPerDeg / 2) * 2 - 1)

        # Generate the filters
        # These Gaussians are used in the row and col separable convolutions.
        k1 = np.array([self.gauss(x1[0], width) * np.sqrt(np.abs(x1[1])) * np.sign(x1[1]),
                       self.gauss(x1[2], width) * np.sqrt(np.abs(x1[3])) * np.sign(x1[3]),
                       self.gauss(x1[4], width) * np.sqrt(np.abs(x1[5])) * np.sign(x1[5])])

        # These are the two 1-d kernels used by red/green
        k2 = np.array([self.gauss(x2[0], width) * np.sqrt(np.abs(x2[1])) * np.sign(x2[1]),
                       self.gauss(x2[2], width) * np.sqrt(np.abs(x2[3])) * np.sign(x2[3])])

        # These are the two 1-d kernels used by blue/yellow
        k3 = np.array([self.gauss(x3[0], width) * np.sqrt(np.abs(x3[1])) * np.sign(x3[1]),
                       self.gauss(x3[2], width) * np.sqrt(np.abs(x3[3])) * np.sign(x3[3])])

        # upsample and downsample
        if uprate > 1:
            upcol = np.concatenate((np.linspace(1, uprate, uprate), np.linspace(uprate - 1, 1, uprate - 1))) / uprate
            s = len(upcol)
            upcol = upcol.reshape(1, -1)  # 1xm matrix
            upcol = self.resize(upcol, [1, s + width - 1])
            # upcol = resize(upcol, [1 s + width - 1]);

            up1 = self.conv2(k1, upcol, 'same')
            up2 = self.conv2(k2, upcol, 'same')
            up3 = self.conv2(k3, upcol, 'same')

            mid = np.ceil(up1.shape[1] / 2)
            downs = np.int_(np.concatenate(
                (np.flip(np.arange(mid, 0, -uprate)), np.arange(mid + uprate, up1.shape[1] + 1, uprate))
            ) - 1)

            k1 = up1[:, downs]
            k2 = up2[:, downs]
            k3 = up3[:, downs]

        return [k1, k2, k3]
    
    def separableFilters_torch(self, sampPerDeg):
        # not full implementation but correct for S-CIELAB usage.
        # Please refer to original Matlab version

        # if sampPerDeg is smaller than minSAMPPERDEG, need to upsample image data before filtering.
        # This can be done equivalently by convolving filters with the upsampling matrix, then downsample it.
        minSAMPPERDEG = 224
        dimension=3

        if sampPerDeg < minSAMPPERDEG:
            uprate = int(np.ceil(minSAMPPERDEG / sampPerDeg))
            sampPerDeg = sampPerDeg * uprate
        else:
            uprate = 1
        
        # these are the same filter parameters, except that the weights are normalized to sum to 1 
        # This eliminates the need to normalize after the filters are generated
        x1 = torch.as_tensor([0.05, 1.00327, 0.225, 0.114416, 7.0, -0.117686])
        x2 = torch.as_tensor([0.0685, 0.616725, 0.826, 0.383275])
        x3 = torch.as_tensor([0.0920, 0.567885, 0.6451, 0.432115])

        # Convert the unit of halfwidths from visual angle to pixels.
        x1[[0, 2, 4]] = x1[[0, 2, 4]] * sampPerDeg
        x2[[0, 2]] = x2[[0, 2]] * sampPerDeg
        x3[[0, 2]] = x3[[0, 2]] * sampPerDeg

        # Limit filter width to 1-degree visual angle, and odd number of sampling points
        # (so that the gaussians generated from Rick's gauss routine are symmetric).
        width = torch.as_tensor(int(np.ceil(sampPerDeg / 2) * 2 - 1))

        # Generate the filters
        # These Gaussians are used in the row and col separable convolutions.
        
        k1 = torch.stack([self.gauss_torch(x1[0], width) * torch.sqrt(torch.abs(x1[1])) * torch.sign(x1[1]),
                       self.gauss_torch(x1[2], width) * torch.sqrt(torch.abs(x1[3])) * torch.sign(x1[3]),
                       self.gauss_torch(x1[4], width) * torch.sqrt(torch.abs(x1[5])) * torch.sign(x1[5])])
            
        # These are the two 1-d kernels used by red/green
        k2 = torch.stack([self.gauss_torch(x2[0], width) * torch.sqrt(np.abs(x2[1])) * torch.sign(x2[1]),
                       self.gauss_torch(x2[2], width) * torch.sqrt(np.abs(x2[3])) * torch.sign(x2[3])])

        # These are the two 1-d kernels used by blue/yellow
        k3 = torch.stack([self.gauss_torch(x3[0], width) * torch.sqrt(torch.abs(x3[1])) * torch.sign(x3[1]),
                       self.gauss_torch(x3[2], width) * torch.sqrt(torch.abs(x3[3])) * torch.sign(x3[3])])

        # upsample and downsample
        if uprate > 1:
            upcol = torch.concatenate((torch.linspace(1, uprate, uprate), torch.linspace(uprate - 1, 1, uprate - 1))) / uprate
            s = len(upcol)
            upcol = torch.reshape(upcol, (1, -1))  # 1xm matrix
            upcol = torch.as_tensor(self.resize(upcol, [1, s + width - 1]))
            # upcol = resize(upcol, [1 s + width - 1]);

            up1 = self.conv2_torch(k1, upcol, 'same')
            up2 = self.conv2_torch(k2, upcol, 'same')
            up3 = self.conv2_torch(k3, upcol, 'same')
            
            mid = np.ceil(up1.shape[1] / 2)
            downs = np.int_(np.concatenate(
                (np.flip(np.arange(mid, 0, -uprate)), np.arange(mid + uprate, up1.shape[1] + 1, uprate))
            ) - 1)
            
            k1 = up1[:, downs]
            k2 = up2[:, downs]
            k3 = up3[:, downs]

        return [k1, k2, k3]

    def generateSCIELABfiltersParams(self, sampPerDeg):
        # not full implementation but correct for S-CIELAB usage.
        # Please refer to original Matlab version

        # if sampPerDeg is smaller than minSAMPPERDEG, need to upsample image data before filtering.
        # This can be done equivalently by convolving filters with the upsampling matrix, then downsample it.
        dimension=3

        # these are the same filter parameters, except that the weights are normalized to sum to 1 
        # This eliminates the need to normalize after the filters are generated
        x1 = np.array([0.05, 1.00327, 0.225, 0.114416, 7.0, -0.117686])
        x2 = np.array([0.0685, 0.616725, 0.826, 0.383275])
        x3 = np.array([0.0920, 0.567885, 0.6451, 0.432115])

        # Convert the unit of halfwidths from visual angle to pixels.
        x1[[0, 2, 4]] = x1[[0, 2, 4]] * sampPerDeg
        x2[[0, 2]] = x2[[0, 2]] * sampPerDeg
        x3[[0, 2]] = x3[[0, 2]] * sampPerDeg   

        return [x1, x2, x3]
    
    def applyGaussFilter(self, im, width, kernels):
        # Apply pytorch Gaussian blur and sum for each channel  
        
        result = torch.zeros_like(im)
        for j in range(int((kernels.shape[0])/2)):
            p = kernels[j*2 + 1]*gaussFilter.gaussian_blur(im, width, kernels[j*2])

            # result is sum of several separable convolutions
            result = result + p
        
        return result
      
    def separableConv(self, im, xkernels, ykernels):
        # Two-dimensional convolution with X-Y separable kernels.
        #
        # im is the input matric. im is reflected on all sides before convolution.
        # xkernels and ykernels are both row vectors.
        # If xkernels and ykernels are matrices, each row is taken as
        #   one convolution kernel and convolved with the image, and the
        #   sum of the results is returned.

        w1 = self.pad4conv(im, xkernels.shape[1], 2)

        result = np.zeros_like(im)
        for j in range(xkernels.shape[0]):
            # first convovle in the horizontal direction
            p = self.conv2(w1, xkernels[j,:].reshape(1,-1))
            p = self.resize(p, im.shape)

            # then the vertical direction
            w2 = self.pad4conv(p, ykernels.shape[1], 1)
            p = self.conv2(w2, ykernels[j,:].reshape(-1,1))
            p = self.resize(p, im.shape)

            # result is sum of several separable convolutions
            result = result + p
        return result
    
    def separableConv_torch(self, im, xkernels, ykernels):
        # Two-dimensional convolution with X-Y separable kernels.
        #
        # im is the input matric. im is reflected on all sides before convolution.
        # xkernels and ykernels are both row vectors.
        # If xkernels and ykernels are matrices, each row is taken as
        #   one convolution kernel and convolved with the image, and the
        #   sum of the results is returned.

        w1 = self.pad4conv_torch(im, xkernels.shape[1], 2)
        
        result = torch.zeros_like(im)
        for j in range(xkernels.shape[0]):
            # first convovle in the horizontal direction
            p = self.conv2_torch(w1, xkernels[j,:].reshape(1,-1))
            p = self.resize_torch(p, im.shape)

            # then the vertical direction
            w2 = self.pad4conv_torch(p, ykernels.shape[1], 1)
            p = self.conv2_torch(w2, ykernels[j,:].reshape(-1,1))
            p = self.resize_torch(p, im.shape)
            
            # result is sum of several separable convolutions
            result = result + p
        return result
        
    def pad4conv(self, im, kernelsize, dim):
        # Pad the input image ready for convolution. The edges of the image are reflected on all sides.
        # kernelsize -- size of the convolution kernel in the format
        #   [numRows numCol]. If one number is given, assume numRows=numCols.
        # dim -- when set at 1, pad extra rows, but leave number of columns unchanged;
        #        when set at 2, pad extra columns, leave number of rows unchanged;

        newim = np.copy(im)
        [m, n] = np.int_(im.shape)
        if not isinstance(kernelsize, list):
            kernelsize = [kernelsize, kernelsize]

        # If kernel is larger than image, than just pad all side with half
        # the image size, otherwise pad with half the kernel size
        if kernelsize[0] >= m:
            h = int(np.floor(m / 2))
        else:
            h = int(np.floor(kernelsize[0] / 2))

        if kernelsize[1] >= n:
            w = int(np.floor(n / 2))
        else:
            w = int(np.floor(kernelsize[1] / 2))

        # first reflect the upper and lower edges
        if h != 0 and dim != 2:
            im1 = np.flipud(newim[0:h, :])
            im2 = np.flipud(newim[m - h:m, :])
            newim = np.concatenate((im1, newim, im2), axis=0)

        # then reflect the left and right sides
        if w != 0 and dim != 1:
            im1 = np.fliplr(newim[:, 0:w])
            im2 = np.fliplr(newim[:, n - w:n])
            newim = np.concatenate((im1, newim, im2), axis=1)

        return newim
        
    def pad4conv_torch(self, im, kernelsize, dim):
        # Pad the input image ready for convolution. The edges of the image are reflected on all sides.
        # kernelsize -- size of the convolution kernel in the format
        #   [numRows numCol]. If one number is given, assume numRows=numCols.
        # dim -- when set at 1, pad extra rows, but leave number of columns unchanged;
        #        when set at 2, pad extra columns, leave number of rows unchanged;

        newim = torch.clone(im)
        [m, n] = np.int_(im.shape)
        #print('is list or not')
        #print(isinstance(kernelsize, list))
        #print('kernel type before')
        #print(type(kernelsize))
        if not isinstance(kernelsize, list):
            kernelsize = [kernelsize, kernelsize]
        #print('kernel type after')
        #print(type(kernelsize))
        
        # If kernel is larger than image, than just pad all side with half
        # the image size, otherwise pad with half the kernel size
        if kernelsize[0] >= m:
            h = int(np.floor(m / 2))
        else:
            h = int(np.floor(kernelsize[0] / 2))

        if kernelsize[1] >= n:
            w = int(np.floor(n / 2))
        else:
            w = int(np.floor(kernelsize[1] / 2))

        # first reflect the upper and lower edges
        if h != 0 and dim != 2:
            im1 = torch.flipud(newim[0:h, :])
            im2 = torch.flipud(newim[m - h:m, :])
            newim = torch.cat((im1, newim, im2), axis=0)

        # then reflect the left and right sides
        if w != 0 and dim != 1:
            im1 = torch.fliplr(newim[:, 0:w])
            im2 = torch.fliplr(newim[:, n - w:n])
            newim = torch.cat((im1, newim, im2), axis=1)

        return newim


def deltaE00(Lab1, Lab2, paramFctr = [1,1,1]):

    kL = paramFctr[0]; kC = paramFctr[1]; kH = paramFctr[2]

    #a1 = np.power(Lab1[1,:],2)
    #b1 = np.power(Lab1[2,:],2)
    #c1 = a1 + b1
    
    # CIELAB Chroma
    C1 = torch.sqrt( torch.pow(Lab1[1,:],2) + torch.pow(Lab1[2,:],2) )
    C2 = torch.sqrt( torch.pow(Lab2[1,:],2) + torch.pow(Lab2[2,:],2) )

    # Lab Prime
    mC = torch.add(C1,C2)/2
    G = 0.5*( 1 - torch.sqrt(  torch.divide( torch.pow(mC,7) , torch.pow(mC,7)+25**7 ) ))
    LabP1 = torch.vstack( (Lab1[0,:], Lab1[1,:]*(1+G), Lab1[2,:]) )
    LabP2 = torch.vstack( (Lab2[0,:], Lab2[1,:]*(1+G), Lab2[2,:]) )

    # Chroma
    CP1 = torch.sqrt( torch.pow(LabP1[1,:],2) + torch.pow(LabP1[2,:],2) )
    CP2 = torch.sqrt( torch.pow(LabP2[1,:],2) + torch.pow(LabP2[2,:],2) )

    # Hue Angle
    hP1 = torch.arctan2( LabP1[2,:],LabP1[1,:] ) * 180/torch.pi # varies from -180 to +180 degree
    hP2 = torch.arctan2( LabP2[2,:],LabP2[1,:] ) * 180/torch.pi # varies from -180 to +180 degree
    hP1[hP1<0] = hP1[hP1<0] + 360 # varies from 0 to +360 degree
    hP2[hP2<0] = hP2[hP2<0] + 360 # varies from 0 to +360 degree


    # Delta Values
    DLP = LabP1[0,:] - LabP2[0,:]
    DCP = CP1 - CP2
    DhP = hP1 - hP2; DhP[DhP>180] = DhP[DhP>180]-360; DhP[DhP<-180] = DhP[DhP<-180]+360
    DHP = torch.multiply( 2*torch.sqrt(torch.multiply(CP1,CP2)), torch.sin( DhP/2.*torch.pi/180. ) )

    # Arithmetic mean of LCh' values
    mLP = ( LabP1[0,:]+LabP2[0,:] )/2
    mCP = (CP1+CP2)/2
    mhP = torch.zeros_like(mCP)
    # for k in range(0,mhP.numel()):
    #     if abs(hP1[k]-hP2[k])<=180:
    #         mhP[k] = (hP1[k]+hP2[k])/2
    #     elif abs(hP1[k]-hP2[k])>180 and hP1[k]+hP2[k]<360:
    #         mhP[k] = (hP1[k]+hP2[k]+360)/2
    #     elif abs(hP1[k]-hP2[k])>180 and hP1[k]+hP2[k]>=360:
    #         mhP[k] = (hP1[k]+hP2[k]-360)/2
    mask1 = torch.abs(hP1-hP2) <= 180
    mhP[mask1] = (hP1+hP2)[mask1]/2
    mask2 = (hP1+hP2) < 360
    mhP[torch.logical_and(torch.logical_not(mask1), mask2)] = ((hP1+hP2)[torch.logical_and(torch.logical_not(mask1), mask2)]+360)/2
    mhP[torch.logical_and(torch.logical_not(mask1), torch.logical_not(mask2))] = ((hP1+hP2)[torch.logical_and(torch.logical_not(mask1), torch.logical_not(mask2))]-360)/2

    # Weighting Functions
    SL = 1 + torch.divide(  0.015*torch.pow(mLP-50,2), torch.sqrt( 20+torch.pow(mLP-50,2) )  )
    SC = 1+0.045*mCP
    T = 1-0.17*torch.cos((mhP-30)*torch.pi/180.)+0.24*torch.cos((2*mhP)*torch.pi/180.)+0.32*torch.cos((3*mhP+6)*torch.pi/180.)-0.2*torch.cos((4*mhP-63)*torch.pi/180.)
    SH = 1+0.015*torch.multiply(mCP,T)

    # Rotation function
    RC = 2 * torch.sqrt(torch.divide(  torch.pow(mCP,7), torch.pow(mCP,7)+25**7  ))
    # DTheta = 30.*exp(-((mhP-275)./25).^2)
    DTheta = 30 * torch.exp(-torch.pow(  (mhP-275)/25,2  ))
    RT = torch.multiply( -torch.sin(2*DTheta*torch.pi/180.), RC )

    dE00 = torch.sqrt(  torch.pow( torch.divide(DLP,kL*SL) ,2) + torch.pow( torch.divide(DCP,kC*SC) ,2) + torch.pow( torch.divide(DHP,kH*SH) ,2)
                        + torch.multiply(RT, torch.multiply( torch.divide(DCP,kC*SC), torch.divide(DHP,kH*SH) ) ))
    return dE00
