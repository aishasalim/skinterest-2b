import tensorflow as tf
import pandas as pd
import numpy as np

AUTO = tf.data.AUTOTUNE

class PhotoAug(tf.keras.layers.Layer):
    def call(self, x):
        x = tf.image.random_brightness(x, 0.15)
        x = tf.image.random_contrast(x, 0.8, 1.25)
        x = tf.image.random_saturation(x, 0.8, 1.25)
        x = tf.image.random_hue(x, 0.05)
        return x

def center_crop(img, frac=0.6, size=(224,224)):
    h = tf.shape(img)[0]; w = tf.shape(img)[1]
    nh = tf.cast(tf.cast(h, tf.float32) * frac, tf.int32)
    nw = tf.cast(tf.cast(w, tf.float32) * frac, tf.int32)
    off_h = (h - nh) // 2; off_w = (w - nw) // 2
    img = tf.image.crop_to_bounding_box(img, off_h, off_w, nh, nw)
    return tf.image.resize(img, size)

def _decode(path, y_light, y_under, frac, size):
    b = tf.io.read_file(path)
    img = tf.image.decode_image(b, channels=3, expand_animations=False)
    img.set_shape([None,None,3])
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = center_crop(img, frac, size)
    return img, (tf.cast(y_light, tf.float32), tf.cast(y_under, tf.int32))

def make_ds(df, img_size=(224,224), frac=0.6, batch=32, training=False):
    ds = tf.data.Dataset.from_tensor_slices((
        df["image_path"].values,
        df["lighting_label"].astype(int).values,
        df["undertone_id"].astype(int).values,
    ))
    if training:
        ds = ds.shuffle(len(df), reshuffle_each_iteration=True)
    ds = ds.map(lambda p,yl,yu: _decode(p,yl,yu,frac,img_size), num_parallel_calls=AUTO)
    if training:
        aug = PhotoAug()
        ds = ds.map(lambda x,y: (aug(x, training=True), {"lighting_out": y[0], "undertone_out": y[1]}),
                    num_parallel_calls=AUTO)
    else:
        ds = ds.map(lambda x,y: (x, {"lighting_out": y[0], "undertone_out": y[1]}),
                    num_parallel_calls=AUTO)
    return ds.batch(batch).prefetch(AUTO)

def balanced_by(df, mode="undertone_x_tone", seed=42, cap=None):
    rng = np.random.default_rng(seed)
    if mode == "undertone_x_tone" and "tone_bucket" in df.columns:
        groups = list(df.groupby(["undertone_label", "tone_bucket"]))
    else:
        groups = list(df.groupby("undertone_label"))
    sizes = [len(g) for _, g in groups]
    target = max(sizes)
    parts = []
    for _, g in groups:
        need = target if cap is None else min(target, cap)
        if len(g) >= need:
            parts.append(g.sample(need, random_state=seed))
        else:
            parts.append(g.sample(need, replace=True, random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
