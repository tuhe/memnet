from utils import *
from generate_data import *
from mnist_cnn_base import Mnist_Wrapper
from matplotlib import pyplot as mpl

def scene_image_encoder(network_wrapper, window_size, positions):
    # pick network

    x = 3

def test() :
    scenes = ensure_resource_file('data/object_retrieve_2_1_T10_rendered')
    mw = Mnist_Wrapper()

    # simpler: wrap the shit out of it and extract subregions.
    # since numpy array keep it as such.

    sc = scenes[0]

    tw = 40
    pad = tw//2

    I = sc.rendered
    iz = I.shape

    Mgrid = 8
    pixels_x = np.linspace(0, iz[0],Mgrid + 2)
    pixels_x = pixels_x[1:-1].astype(int)
    pixels_y = pixels_x

    pixels_xx, pixels_yy = np.meshgrid(pixels_x, pixels_y)

    Ipad = np.pad(I, ((pad,pad),(pad,pad)), 'constant')
    mpl.figure(1)
    mpl.imshow(Ipad)

    mpl.figure(2)
    cc = 1
    for ix in range(  Mgrid ):
       for iy in range( Mgrid ):
            xl = pixels_xx[ix, iy]
            yl = pixels_yy[ix, iy]
            mpl.subplot(Mgrid,Mgrid,cc)
            di = Ipad[ (iz[0] - yl)-tw//2 + pad : (iz[0] - yl) + tw//2 + pad, xl-tw//2 + pad : xl + tw//2 + pad ]
            cc = cc+1
            mpl.imshow(di)
            mpl.title('x: ' + str(xl) + ' y: ' + str(yl) )

    cl,prob = mw.predict(di.reshape((1, tw * tw)))

    x = 4

if __name__ == '__main__':
    test()
