"""Model architectures. Models take PREPROCESSED input (see src/preprocessing.py).

All models are linear stacks: [Input, augmentation, (backbone | conv blocks), head...].
Augmentation layers are active only during training. This linear layout is what lets
src/gradcam.py walk the layers generically.
"""
from __future__ import annotations

from pathlib import Path

import keras
from keras import layers


def augmentation_block() -> keras.Sequential:
    # Geometric-only augmentation: safe for any input value range (no clipping).
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.05, 0.05),
    ], name="augmentation")


def build_custom_cnn(input_shape: tuple[int, int, int], num_classes: int, dropout: float = 0.3) -> keras.Model:
    """Model A. Four conv blocks (Conv-BN-ReLU x2 + MaxPool), GAP, small dense head.

    Easy to explain: each block doubles the filters (32->256) while pooling halves the
    spatial size, so early layers learn edges/textures and later layers learn parts/shapes.
    """
    inputs = keras.Input(shape=input_shape, name="image")
    x = augmentation_block()(inputs)
    for i, filters in enumerate([32, 64, 128, 256], start=1):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{i}_a")(x)
        x = layers.BatchNormalization(name=f"bn{i}_a")(x)
        x = layers.ReLU(name=f"relu{i}_a")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{i}_b")(x)
        x = layers.BatchNormalization(name=f"bn{i}_b")(x)
        x = layers.ReLU(name="last_conv_activation" if i == 4 else f"relu{i}_b")(x)
        if i < 4:
            x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
            x = layers.Dropout(0.1, name=f"drop{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="fc")(x)
    x = layers.Dropout(max(dropout, 0.4), name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name="custom_cnn")


class PretrainedWeightsError(RuntimeError):
    """No pretrained ImageNet weights could be loaded and random initialisation was not explicitly allowed."""


# Verified mirror (legacy Keras-Applications release). Only MobileNetV2: the legacy ResNet50 file uses a
# different layer layout/BN convention and is NOT loadable into the Keras 3 architecture, so it is not offered.
GITHUB_MIRRORS = {
    "mobilenetv2": {
        "url": "https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v1.1/"
               "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5",
        "sha256": "f8aff69536bd77a692c594f559c798c19bf7f3f36668fc9fa00b21c6aab4797c",
    },
}


def _weight_sources(kind: str):
    """Ordered list of (label, loader) pairs; each loader returns a keras `weights=` argument."""
    from src.utils import load_config, models_dir, resolve
    out = []
    local = load_config()["models"][kind].get("weights_path")
    if local and resolve(local).is_file():
        out.append((f"local file {local}", lambda: str(resolve(local))))
    out.append(("keras-imagenet (official download)", lambda: "imagenet"))
    if kind in GITHUB_MIRRORS:
        m = GITHUB_MIRRORS[kind]

        def fetch():
            return keras.utils.get_file(fname=Path(m["url"]).name, origin=m["url"], file_hash=m["sha256"],
                                        hash_algorithm="sha256", cache_dir=str(models_dir() / "pretrained"),
                                        cache_subdir="")
        out.append(("github mirror (sha256-verified)", fetch))
    return out


def _load_backbone(factory, kind: str, input_shape, use_pretrained: bool, allow_random_init: bool):
    """Return (backbone, pretrained_loaded, source_description).

    A failed pretrained load is an ERROR unless allow_random_init is set: silently training a "transfer"
    model from random weights gives ~50% accuracy that looks like ordinary under-training.
    """
    if not use_pretrained:
        return factory(weights=None, include_top=False, input_shape=input_shape, name="backbone"), False, \
               "random init (explicit --no-pretrained)"
    errors = []
    for label, get in _weight_sources(kind):
        try:
            w = get()
            return factory(weights=w, include_top=False, input_shape=input_shape, name="backbone"), True, label
        except Exception as exc:
            errors.append(f"  - {label}: {type(exc).__name__}: {str(exc)[:120]}")
    msg = ("Could not load pretrained ImageNet weights for " + kind + ":\n" + "\n".join(errors) +
           "\nFix the network/proxy, or put an official Keras no-top weights file at the path given by "
           f"models.{kind}.weights_path in config.yaml. To train from random weights on purpose, "
           "pass --no-pretrained (or --allow-random-init to continue after a failed download).")
    if not allow_random_init:
        raise PretrainedWeightsError(msg)
    print("\n" + "!" * 78 + "\nWARNING: " + msg + "\nContinuing with RANDOM initialisation (--allow-random-init).\n" + "!" * 78)
    return factory(weights=None, include_top=False, input_shape=input_shape, name="backbone"), False, \
           "random init (FALLBACK after failed weight download)"


def build_transfer_model(kind: str, input_shape: tuple[int, int, int], num_classes: int,
                         dropout: float = 0.3, use_pretrained: bool = True,
                         allow_random_init: bool = False) -> tuple[keras.Model, bool, str]:
    """Models B/C: frozen ImageNet backbone + GAP + Dropout + softmax head. Returns (model, pretrained, source)."""
    factory = {"mobilenetv2": keras.applications.MobileNetV2, "resnet50": keras.applications.ResNet50}[kind]
    backbone, loaded, source = _load_backbone(factory, kind, input_shape, use_pretrained, allow_random_init)
    backbone.trainable = not loaded          # freeze only if pretrained (phase 1: train head)
    inputs = keras.Input(shape=input_shape, name="image")
    x = augmentation_block()(inputs)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name=kind), loaded, source


def unfreeze_for_fine_tuning(model: keras.Model, last_n: int) -> int:
    """Unfreeze the last `last_n` backbone layers, keeping BatchNorm frozen. Returns #unfrozen."""
    backbone = model.get_layer("backbone")
    backbone.trainable = True
    for layer in backbone.layers[:-last_n]:
        layer.trainable = False
    n = 0
    for layer in backbone.layers[-last_n:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True; n += 1
    return n
