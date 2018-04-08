import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy as sp
from PIL import Image

from matplotlib import pyplot as plt

from utils import ensure_resource_file
from mnist_cnn_base import Mnist_Wrapper
from scene_encoder import scene_add_gridded_encoding


ddopts = [[-1, 0], [0, -1], [1, 0], [0, 1]]

def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)


class Scene :
    KW_TYPE = "TYPE"
    KW_COMMAND = "COMMAND"
    commands = ['LEFT', 'BELOW', 'RIGHT', 'ABOVE']


    def __init__(self,Ntypes):
        self.Ntypes = Ntypes # Number of unique object types.
        self.points = []
        self.query_target = (None,None)
        self.rendered = None

    def add_element(self, xy, type):
        self.points.append( (xy, type ) )

    def distance_to_nearest(self,nxy):
        xy = self.all_xy()
        return sp.spatial.distance.cdist(xy,nxy).min()


    def all_xy(self):
        ar = np.asarray( [ xy for (xy,_) in self.points] ).squeeze()
        return ar

    def plot(self) :


        cm = plt.cm.Dark2.colors

        for p in self.points :
            xy = p[0].ravel()
            plt.plot( xy[0], xy[1], 'o', color=cm[ p[1] ])
            plt.text( xy[0], xy[1], "  %i"%p[1])

        plt.axis([0, 1, 0, 1])


        def c2s(command) :
            s = ""
            for cl in command :
                if cl[0] == self.KW_TYPE :
                    s = s + "obj:%i"%cl[1]
                if cl[0] == self.KW_COMMAND :
                    s = s + " %s "%self.commands[cl[1]]

            return s

        ss = "query: " + c2s( self.query_target[0] ) + " target: " + c2s( [self.query_target[1] ] )
        plt.title(ss)
        plt.show()
        43



    def render_scene(self,images,pixels):
        #background = Image.new('RGBA', (pixels, pixels), (255, 255, 255, 255))
        #plt.imshow(background)
        background = np.zeros([pixels, pixels])
        for p in self.points :
            dd = p[1]
            ima = images[dd][ np.random.randint(0, len(images[dd]) )]
            xy = p[0]
            #im = Image.fromarray(np.uint8(ima * 255))
            background = safepaste(background, ima, xy[0, 0]*pixels, xy[0, 1] * pixels)
        return background

def safepaste(A,B,x,y) :
    x = np.round(x)
    y = np.round(y)

    zb = B.shape
    za = A.shape

    for i in range(zb[0]) :
        for j in range(zb[1]) :

            ii = int( za[0] - y + i - zb[0] // 2 )
            jj = int( x + j - zb[1] // 2 )

            if ii < za[0] and ii >= 0 and jj < za[1] and jj >= 0:
                # print("ii,jj")
                A[ii,jj] = 1.0 * B[i,j]
    return A

def generate_scenes_object_retrival(T=10,Ntype=4,side_padding=0.05,obj_radius=0.025,obj_adjecancy_shift=0.05, obj_other_shift=0.1,Mconfusers_close=1, Mconfusers_anywhere=0):

    scenes = []
    for t in range(T):
        ds = Scene(Ntype)
        objs = []
        ddt = side_padding + obj_radius * 3 + obj_adjecancy_shift
        lxy_true = (np.random.rand(1,2) * (1-2*ddt) + ddt)
        cat_true = np.random.randint(0, Ntype)

        ds.add_element(lxy_true,cat_true)

        dir_other = np.random.choice(4, size=Mconfusers_close + 1, replace=False)
        cat_other = np.random.choice([i for i in range(Ntype) if i != cat_true], Mconfusers_close + 1,
                                     replace=False)


        for j in range(Mconfusers_close + 1) :
            dxy = lxy_true + np.asanyarray(ddopts[dir_other[j]] ) * ( obj_radius*2 + obj_adjecancy_shift )
            ds.add_element(dxy,cat_other[j])

        for j in range(Mconfusers_anywhere):
            cat = np.random.choice( [j for j in range(Ntype) if not j == cat_true] )

            while True:
                lxy = np.random.rand(1, 2) * (1-side_padding*2) + side_padding

                if ds.distance_to_nearest(lxy) > obj_radius + obj_other_shift: break

            ds.add_element( lxy, cat)

        query = [(ds.KW_TYPE, cat_true), (ds.KW_COMMAND, dir_other[0])]
        target = (ds.KW_TYPE, cat_other[0])

        ds.query_target = (query, target)
        scenes.append(ds)
    return scenes




class KNNEncoder:
    def __init__(self, N, K):
        self.N = N
        self.K = K
        self.points_xy = np.random.rand(self.N, 2)

    # returns a m x N vector of encodings where m is the number of points in xy

    def encode_points(self, xy): # m x 2 numpy array
        m = xy.shape[0]

        out = np.zeros( (m, self.N) )
        dists = sp.spatial.distance.cdist(xy, self.points_xy)
        idx = np.argpartition(dists, self.K, axis=1)[:, :3]
        for j in range(m) :
            out[j,idx[j,:]] = 1

        return out

class RandomPositionEncoder :

    def __init__(self, N, sigma): # define class with N neurons
        self.points_xy = np.random.rand(N,2)
        self.N = N
        self.sigma = sigma / np.sqrt(N)


    def encode_points(self, xy): # n x 2 numpy array
        m = np.exp(
            -np.power(sp.spatial.distance.cdist(xy, self.points_xy), 2) / (2 * self.sigma * self.sigma))
        m = m / m.sum(1)
        return m

if False :
    N = 50
    x = np.random.rand(N)
    y = np.random.rand(N)
    plt.scatter(x, y)
    plt.show()


def encode_scene(scenes,posenc,fout):

    m_scenes = []
    m_queries = []
    m_targets = []

    for scene in scenes:
        Ncommands = len(scene.commands)
        Ntypes = scene.Ntypes

        mscene = np.zeros( (posenc.N, Ntypes) )
        for p in scene.points :
            xy_enc = posenc.encode_points(p[0])
            im_enc = one_hot(p[1], Ntypes)

            mscene = mscene + xy_enc.transpose() @ im_enc.transpose()

        mscene = mscene.clip(0, 1)

        Nenc = {scene.KW_COMMAND : Ncommands, scene.KW_TYPE : Ntypes}
        mquery = cat_seq_encode(scene.query_target[0], Nenc)
        mtarget = cat_seq_encode([scene.query_target[1]], Nenc)[0]

        m_scenes.append(mscene)
        m_targets.append(mtarget)
        m_queries.append(mquery)

        34

    print("Saving scenes to... " + fout)
    np.save(fout, [m_scenes,m_queries,m_targets])

    return fout #mscene, mquery, mtarget


def cat_seq_encode(comseq, Nencoding):
    # for s in comseq :
    es = [one_hot(s[1], Nencoding[s[0]]) for s in comseq]
    return es


def one_hot(k, N):
    a = np.zeros((N, 1))
    a[k] = 1
    return a


import glob

def mnist_split() :
    print("Loading mnist..")
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    # train_data = mnist.train.images  # Returns np.array
    # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # eval_data = mnist.test.images  # Returns np.array
    # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    train_data = np.asarray(mnist.train.images)
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

    train_digits = []
    for k in range(10) :
        ims = train_data[train_labels == k, :]
        dd = []
        for i in range( ims.shape[0] ) :
            ima = np.reshape( ims[i,:], [28, 28])
            dd.append(ima)

        train_digits.append( dd )

    print("Done!")
    return train_digits


def test1(T=1000):
    sp = 0.05
    Mconfusers_close = 2
    Mconfusers_anywhere = 1

    base = 'data/object_retrieve_%i_%i_T%i'%(Mconfusers_close,Mconfusers_anywhere,T)

    gf = lambda : generate_scenes_object_retrival(T=T, Ntype=4, side_padding=sp, obj_radius=sp, obj_adjecancy_shift=sp,
                                    obj_other_shift=sp * 2,
                                    Mconfusers_close=2, Mconfusers_anywhere=1)

    scenes = ensure_resource_file(base + "_scene", gf )

    def render_scenes(scenes):
        digits = mnist_split()
        for i in range(len(scenes)) :
            scenes[i].rendered = scenes[i].render_scene(digits, 200)
        return scenes
    base_rendered = base + "_rendered"
    scenes = ensure_resource_file(base_rendered, lambda : render_scenes(scenes))

    grid_N = 22
    base_gridded = base_rendered + "_gridN%i"%grid_N
    print("Loading nn wrapper...")
    NNwrapper = Mnist_Wrapper()
    print("Done!")
    scenes = ensure_resource_file(base_gridded, lambda: scene_add_gridded_encoding(scenes, NNwrapper, grid_N) )


    dx = 3
    plt.figure(1)
    plt.clf()
    sc = scenes[dx]

    plt.imshow(sc.rendered)
    plt.figure(2)
    plt.clf()
    sc.plot()
    plt.figure(3)
    plt.clf()
    for pr in sc.patch_encoding :
        for patch in pr :
            xy = patch[1]
            cl = patch[2]
            probs = patch[3]
            if cl != len(probs) - 1:
                plt.text(xy[0], xy[1], "%i"%cl)

    plt.show()

    return

    pe = RandomPositionEncoder(40, .5)
    #pe = KNNEncoder(40, 3)

    fout = 'data/object_retrieve_%i_%i_T%i.npy'%(Mconfusers_close,Mconfusers_anywhere,T)
    encode_scene(scenes,pe,fout)
    #fout = generate_data_object_retrival(Nencoding=enc_lengths, Mconfusers_close=2, Mconfusers_anywhere=4, T=T)
    return fout

if __name__ == "__main__":
    print("generating data..")
    test1()

