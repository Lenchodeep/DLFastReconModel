import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imsave

def attentionMapVisal():
    att = np.load('E:/DLCode/DLFastReconstruction/PsychoGanCode/checkpoints/IXI/DA/ABLATION/DC/test/attentioni.npy') 
    att = np.average(att, axis=0)
    # att = att[12,:,:]
    image = np.load('E:/DLCode/DLFastReconstruction/PsychoGanCode/checkpoints/IXI/DA/ABLATION/DC/test/image.npy')
    att = (att-att.min())/(att.max()-att.min())
    # image = (image-image.min())/(image.max()-image.min())

    test = (image * att)
    test = (test-test.min())/(test.max()-test.min())
    print(test.min(), test.max())
    camp = plt.cm.jet
    test1 = camp(test/test.max())
    imsave('E:/DLCode/DLFastReconstruction/PsychoGanCode/checkpoints/IXI/DA/ABLATION/DC/test/attentioni.png',test1)
    plt.figure()
    plt.imshow((test), plt.cm.jet, vmax=1, vmin=0)
    plt.figure()
    plt.imshow((att), plt.cm.jet, vmax=1, vmin=0)
    plt.show()

def testKMax():
    att = np.load('E:/DLCode/DLFastReconstruction/PsychoGanCode/checkpoints/IXI/DA/ABLATION/DC/test/testk.npy') 
    image = np.load('E:/DLCode/DLFastReconstruction/PsychoGanCode/checkpoints/IXI/DA/ABLATION/DC/test/image.npy')
    image = (image-image.min())/(image.max()-image.min())

    k = att[23,:,:]
    att = np.mean(att, axis=0)
    # k = np.log10(att)
    recon = np.abs(np.fft.ifft2(k))
    recon = (recon-np.min(recon))/ (np.max(recon)-np.min(recon))
    print(np.max(recon))
    plt.figure()
    plt.imshow(np.fft.fftshift(k), plt.cm.gray)
    plt.figure()
    plt.imshow((image), plt.cm.gray)
    plt.figure()
    plt.imshow((recon), plt.cm.gray)
    plt.show()

attentionMapVisal()