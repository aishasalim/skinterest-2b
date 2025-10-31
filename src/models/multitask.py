import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet152V2
from src.layers.color_calibration import ColorCalibration, ResNetV2Preprocess

def build_multitask(img_size=(224,224), drop=0.35, num_classes=10, ccm_reg=1e-4):
    inp = layers.Input(shape=img_size + (3,))
    x = ColorCalibration(name="color_calibration", reg_lambda=ccm_reg)(inp)
    x = ResNetV2Preprocess(name="res_net_v2_preprocess")(x)

    base = ResNet152V2(include_top=False, weights="imagenet",
                       input_shape=img_size + (3,), pooling="avg", name="resnet152v2")
    base.trainable = False
    feats = base(x, training=False)
    feats = layers.Dropout(drop)(feats)

    h_light = layers.Dense(128, activation="relu", name="light_fc")(feats)
    h_light = layers.Dropout(0.2, name="light_drop")(h_light)
    lighting_out = layers.Dense(1, activation="sigmoid", name="lighting_out")(h_light)

    h_tone = layers.Dense(128, activation="relu", name="tone_fc")(feats)
    h_tone = layers.Dropout(0.2, name="tone_drop")(h_tone)
    undertone_out = layers.Dense(num_classes, activation="softmax", name="undertone_out")(h_tone)

    return models.Model(inp, [lighting_out, undertone_out])
