"""
The generator half of the pipeline: a convolutional VAE over one-hot
antibody sequences.

The VAE itself was *not* the reason no candidate cleared the dashboard's
threshold -- sampling the trained 200-residue weights produces real kappa
light-chain frameworks ("DIQMTQSPSSLSASVGDRVTITC..."), so the decoder works.
Two things were still limiting it, and both are fixed here:

* `latent_dim` was hardcoded to 2. A 2-D latent can only trace a 2-D manifold
  of sequences, so 800 samples came back as near-copies of a handful of
  frameworks -- few *distinct* candidates, whatever the threshold. It is now
  a constructor argument (default 16). Old 2-D weights still load fine:
  `VAE.reload` replaces the encoder/decoder wholesale, so the value passed
  here only affects freshly trained models, and generation reads the real
  latent width back off the decoder.
NO BatchNorm, deliberately. Adding it here looked harmless and was not: the
decoder trained fine (87.5% held-out residue accuracy) but at INFERENCE, where
BatchNorm switches from batch statistics to its moving averages, the very same
latent points decoded to runs of a single residue ("NCNNNHNNNYNNPNC...") while
the train-mode forward pass on those points gave real antibody frameworks
("QVQLKESGFSLPQPTETLSLTTSVEGG..."). Generation only ever runs in inference
mode, so every candidate came out as junk -- which the interaction classifier
then happily scored at 0.99. The encoder is left without it too: its moving
averages would otherwise distort the latent statistics that sampling relies on.

* The decoder's output was a plain per-cell `sigmoid`, i.e. 22 independent
  Bernoulli draws per position, which is not what a residue is. The output
  head is now a softmax *across the alphabet axis*, so each position is a
  proper distribution over the 22 letters. The tensor shape is unchanged,
  so everything downstream (`decode`'s argmax, the Siamese encoder) is
  unaffected.
"""
from keras import layers
import keras

from intelligent_antibodies.modules.layers.SamplingLayer import SamplingLayer
from intelligent_antibodies.modules.models.VAE import VAE


class VAEFull:

    def __init__(self, vector_size, alphabet_size, latent_dim: int = 16, softmax_output: bool = True):
        self.vector_size = vector_size
        self.alphabet_size = alphabet_size
        self.latent_dim = latent_dim

        # Encoder
        encoder_inputs = keras.Input(shape=(vector_size, alphabet_size, 1))
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        # A third stride-2 block. With only two, a 400-residue window flattens
        # to 100*11*64 = 70,400 units, and the dense layer that follows is 18M
        # parameters on its own -- which does not fit the project's 50% GPU
        # budget (the run spent its time retrying allocations instead of
        # training). Three blocks flatten to 50*3*128 = 19,200 instead.
        x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        # 16 units was a hard bottleneck in front of the latent layer: whatever
        # latent_dim asked for, everything had to squeeze through 16 first.
        x = layers.Dense(256, activation="relu")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = SamplingLayer()([z_mean, z_log_var])
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        # Decoder
        inputs  = keras.Input(shape=(latent_dim,))
        x       = layers.Dense(vector_size // 2 * alphabet_size // 2 * 64, activation="relu")(inputs)
        x       = layers.Reshape((vector_size // 2, alphabet_size // 2, 64))(x)
        x       = layers.Conv2DTranspose(64, 3, strides=1, padding="same", activation="relu")(x)
        x       = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
        logits  = layers.Conv2DTranspose(1, 3, padding="same", name="logits")(x)

        if softmax_output:
            # Softmax over the ALPHABET axis (axis 2 of (pos, alphabet, 1)):
            # exactly one residue per position, which is what the data is and
            # what makes the decoder's max activation a real confidence.
            outputs = layers.Softmax(axis=2, name="residue_distribution")(logits)
        else:
            outputs = layers.Activation("sigmoid", name="residue_sigmoid")(logits)

        self.decoder = keras.Model(inputs, outputs, name="decoder")

        # VAE. `categorical` matches the softmax head above; the loss and the
        # output head have to agree, so it is derived rather than set twice.
        self.vae = VAE(
            self.encoder, self.decoder,
            reconstruction="categorical" if softmax_output else "bernoulli",
        )


if __name__ == "__main__":
    import numpy as np

    vector_size, alphabet_size, latent_dim = 120, 22, 8
    model = VAEFull(vector_size, alphabet_size, latent_dim=latent_dim)

    # Self-check: no layer in either half may behave differently in training
    # and inference. BatchNorm here cost a full training run -- the decoder
    # scored 87.5% held-out residue accuracy and still emitted runs of a
    # single residue at generation time, because generation only ever runs in
    # inference mode where BatchNorm swaps batch statistics for its moving
    # averages. Dropout would do the same to the latent statistics.
    for half in (model.encoder, model.decoder):
        offenders = [
            layer.name for layer in half.layers
            if isinstance(layer, (layers.BatchNormalization, layers.Dropout))
        ]
        assert not offenders, f"{half.name} has train/inference-dependent layers: {offenders}"

    z = np.random.normal(size=(4, latent_dim)).astype("float32")
    train_mode = np.asarray(model.decoder(z, training=True))
    eval_mode = np.asarray(model.decoder(z, training=False))
    assert np.allclose(train_mode, eval_mode, atol=1e-6), (
        "decoder output depends on training mode: generated sequences will not "
        "match what the model was trained to produce"
    )

    # Each position must be a distribution over the alphabet, not 22
    # independent activations.
    assert eval_mode.shape == (4, vector_size, alphabet_size, 1), eval_mode.shape
    sums = eval_mode.squeeze(-1).sum(axis=-1)
    assert np.allclose(sums, 1.0, atol=1e-4), f"positions do not sum to 1: {sums.min()}..{sums.max()}"

    print("VAEFull self-check OK")
