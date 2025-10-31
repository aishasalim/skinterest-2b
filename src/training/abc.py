import tensorflow as tf

LOSSES = {
  "lighting_out": tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
  "undertone_out": tf.keras.losses.SparseCategoricalCrossentropy()
}
METRICS = {
  "lighting_out": [tf.keras.metrics.BinaryAccuracy(name="acc"), tf.keras.metrics.AUC(name="auc")],
  "undertone_out": [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
}
W = {"lighting_out":1.0, "undertone_out":1.0}

def adamw(lr, wd):
    try:
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd, clipnorm=1.0)
    except Exception:
        import tensorflow_addons as tfa
        return tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd, clipnorm=1.0)

def compile_with(model, stage):
    if stage == "A": opt = adamw(1e-3, 1e-4)
    elif stage == "B": opt = adamw(5e-4, 1e-4)
    else: opt = adamw(5e-5, 5e-5)
    model.compile(optimizer=opt, loss=LOSSES, loss_weights=W, metrics=METRICS)

def run_phases(model, train_ds, val_ds, epochs=(4,3,8), callbacks=()):
    # A: heads only
    model.get_layer("color_calibration").trainable = False
    for l in model.layers:
        if "resnet152v2" in l.name.lower():
            l.trainable = False
    compile_with(model, "A")
    hist_A = model.fit(train_ds, validation_data=val_ds, epochs=epochs[0], callbacks=list(callbacks), verbose=1)

    # B: unfreeze CCM
    model.get_layer("color_calibration").trainable = True
    for l in model.layers:
        if "resnet152v2" in l.name.lower():
            l.trainable = False
    compile_with(model, "B")
    hist_B = model.fit(train_ds, validation_data=val_ds, epochs=epochs[1], callbacks=list(callbacks), verbose=1)

    # C: top 40% of backbone
    base = model.get_layer("resnet152v2")
    cut = int(len(base.layers) * 0.6)
    for i, L in enumerate(base.layers):
        L.trainable = (i >= cut)
    compile_with(model, "C")
    hist_C = model.fit(train_ds, validation_data=val_ds, epochs=epochs[2], callbacks=list(callbacks), verbose=1)

    return hist_A, hist_B, hist_C
