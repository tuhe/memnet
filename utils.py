import tensorflow as tf
import math
import pickle
import gzip
import glob

def ensure_resource_file(outfile,gf=None) :
    base_zp = outfile + ".pklz"
    if any(glob.glob(base_zp)):
        print("Loading resources from " + base_zp)
        with gzip.open(base_zp, 'rb') as output:
            res = pickle.load(output)
    else:
        print("Re-generating resources; later saving to..." + base_zp)
        if not gf : print("Bad function supplied")
        res = gf()
        print("Done!")
        pickle.dump(res, gzip.open(base_zp, 'wb'))

    return res

def augment(images, labels,
            resize=None,  # (width, height) tuple or None
            horizontal_flip=False,
            vertical_flip=False,
            rotate=0,  # Maximum rotation angle in degrees
            crop_probability=0,  # How often we do crops
            min_crop_percent=0.6,  # Minimum linear dimension of a crop
            max_crop_percent=1.,  # Maximum linear dimension of a crop
            mixup=0, # Mixup coeffecient, see https://arxiv.org/abs/1710.09412.pdf
            pad_to_size=None, # tupple of target size (target_height, target_width)
            class_translation_minmax=None): # each row is an element of form [t_min,t_max]. Images will be translated unifromormly within this range
    if resize is not None:
        images = tf.image.resize_bilinear(images, resize)

    # My experiments showed that casting on GPU improves training performance
    if images.dtype != tf.float32:
        images = tf.image.convert_image_dtype(images, dtype=tf.float32)
        #images = tf.subtract(images, 0.5)
        #images = tf.multiply(images, 2.0)

    if pad_to_size is not None :
        py1 = (pad_to_size[0] - images.shape[1].value)//2
        py2 = pad_to_size[0] - images.shape[1].value - py1

        px1 = (pad_to_size[1] - images.shape[2].value) // 2
        px2 = pad_to_size[1] - images.shape[2].value - px1
        paddings = tf.constant([[0, 0], [py1, py2], [px1, px2], [0,0]])

        images = tf.pad(images, paddings, "CONSTANT")

    if class_translation_minmax is not None :
        dts = tf.gather(class_translation_minmax, labels)

        shp = tf.shape(images)
        batch_size, height, width = shp[0], shp[1], shp[2]

        neg_coin = tf.cast( tf.less(tf.random_uniform([batch_size,2], 0, 1.0), 0.5), dtype=tf.float32)
        neg_coin = tf.add(tf.multiply(neg_coin,tf.constant(2.) ),tf.constant(-1.) )

        pt_size_unif = tf.random_uniform( [batch_size,2] )

        coin_dims = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)

        b = tf.reduce_max(dts, reduction_indices=[1])
        a = tf.reduce_min(dts, reduction_indices=[1])

        ar = tf.where(coin_dims,a,tf.scalar_mul( tf.constant( 0. ),a ) )
        ar = tf.reshape(ar, [batch_size,1])
        ar = tf.concat([ar, tf.subtract( tf.reshape(a, [batch_size,1]) ,ar )], 1)

        m1 = tf.multiply(pt_size_unif, tf.reshape( tf.subtract( tf.reshape(b, [batch_size,1]), ar), [batch_size,2]) )
        RT = tf.multiply( tf.add( m1 , tf.reshape(ar, [batch_size,2])),  neg_coin )

        images = tf.contrib.image.translate(images, RT)

    with tf.name_scope('augmentation'):
        shp = tf.shape(images)
        batch_size, height, width = shp[0], shp[1], shp[2]
        width = tf.cast(width, tf.float32)
        height = tf.cast(height, tf.float32)

        # The list of affine transformations that our image will go under.
        # Every element is Nx8 tensor, where N is a batch size.
        transforms = []
        identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        if horizontal_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
                [-1., 0., width, 0., 1., 0., 0., 0.], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                         tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if vertical_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
                [1, 0, 0, 0, -1, height, 0, 0], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                         tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if rotate > 0:
            angle_rad = rotate / 180 * math.pi
            angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
            transforms.append(
                tf.contrib.image.angles_to_projective_transforms(
                    angles, height, width))

        if crop_probability > 0:
            crop_pct = tf.random_uniform([batch_size], min_crop_percent,
                                         max_crop_percent)
            left = tf.random_uniform([batch_size], 0, width * (1 - crop_pct))
            top = tf.random_uniform([batch_size], 0, height * (1 - crop_pct))
            crop_transform = tf.stack([
                crop_pct,
                tf.zeros([batch_size]), top,
                tf.zeros([batch_size]), crop_pct, left,
                tf.zeros([batch_size]),
                tf.zeros([batch_size])
            ], 1)

            coin = tf.less(
                tf.random_uniform([batch_size], 0, 1.0), crop_probability)
            transforms.append(
                tf.where(coin, crop_transform,
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if transforms:
            images = tf.contrib.image.transform(
                images,
                tf.contrib.image.compose_transforms(*transforms),
                interpolation='BILINEAR')  # or 'NEAREST'

        def cshift(values):  # Circular shift in batch dimension
            return tf.concat([values[-1:, ...], values[:-1, ...]], 0)

        if mixup > 0:
            labels = tf.to_float(labels)
            beta = tf.distributions.Beta(mixup, mixup)
            lam = beta.sample(batch_size)
            ll = tf.expand_dims(tf.expand_dims(tf.expand_dims(lam, -1), -1), -1)
            images = ll * images + (1 - ll) * cshift(images)
            labels = lam * labels + (1 - lam) * cshift(labels)

            labels = tf.to_int32(labels)

    return images, labels