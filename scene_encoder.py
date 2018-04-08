from utils import *
#from generate_data import *
#from mnist_cnn_base import Mnist_Wrapper
from matplotlib import pyplot as mpl
import numpy as np
import math

def scene_image_encoder(network_wrapper, window_size, positions):
    # pick network

    x = 3
def scene_add_gridded_encoding(scenes, NNwrapper, grid_N) :
    ## add grid encoding to the scene.
    iz = scenes[0].rendered.shape

    loc_x = np.linspace(0, 1,grid_N + 2)
    loc_x = loc_x[1:-1]
    loc_y = loc_x

    tw = NNwrapper.tw
    pad = tw // 2


    for sdex in range(len(scenes)) :
        print("Patching %i"%(sdex,))
        sc = scenes[sdex]
        Ipad = np.pad(sc.rendered, ((pad, pad), (pad, pad)), 'constant')

        patches = np.zeros( (grid_N*grid_N, tw*tw) )
        ij = []
        cc = 0
        for i in range(grid_N) :
            for j in range(grid_N) :
                xl = math.floor( loc_x[i] * iz[0] )
                yl = math.floor( loc_y[j] * iz[1] )
                di = Ipad[(iz[0] - yl) - tw // 2 + pad: (iz[0] - yl) + tw // 2 + pad, xl - tw // 2 + pad: xl + tw // 2 + pad]

                ij.append([i,j])
                patches[cc,:] = di.reshape(1,tw*tw)

                cc = cc + 1
        cl, prob = NNwrapper.predict(patches)

        scenes[sdex].patch_encoding = [[0] * grid_N for j in range(grid_N)]
        for k in range(cc) :
            i = ij[k][0]
            j = ij[k][1]
            scenes[sdex].patch_encoding[i][j] = ( (i,j), (loc_x[i], loc_y[j]), cl[k], prob[k,:])
    return scenes


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
            mpl.figure(2)
            xl = pixels_xx[ix, iy]
            yl = pixels_yy[ix, iy]
            mpl.subplot(Mgrid,Mgrid,cc)
            di = Ipad[ (iz[0] - yl)-tw//2 + pad : (iz[0] - yl) + tw//2 + pad, xl-tw//2 + pad : xl + tw//2 + pad ]
            mpl.imshow(di)
            mpl.title('x: ' + str(xl) + ' y: ' + str(yl) )

            cl,prob = mw.predict(di.reshape((1, tw * tw)))
            mpl.figure(3)
            mpl.subplot(Mgrid, Mgrid, cc)
            Kmnist = 11
            mpl.bar( np.asarray( range(Kmnist) ), prob.ravel() )
            mpl.show()

            cc = cc+1


    x = 4

if __name__ == '__main__':
    test()
