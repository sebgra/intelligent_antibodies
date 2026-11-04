import json
import os

import keras
import numpy as np
import tensorflow as tf

from intelligent_antibodies.modules.layers import SamplingLayer


class VAE(keras.Model):
    """
    Variational autoencoder over one-hot protein sequences.

    Parameters
    ----------
    reconstruction : {"categorical", "bernoulli"}
        How the decoder's output is scored. ``"categorical"`` (the default,
        and what `VAEFull`'s softmax head produces) treats each sequence
        position as one distribution over the alphabet -- which is what a
        residue actually is. ``"bernoulli"`` is the original per-cell binary
        cross-entropy, kept so previously trained models can still be
        reloaded and retrained as they were.
    beta : float
        Weight on the KL term. Generation samples the latent *prior*, so the
        aggregate posterior has to actually resemble that prior; `beta` is the
        knob that trades reconstruction fidelity against that match.
    free_bits : float
        Per-dimension KL floor, in nats. With a wider latent space some
        dimensions otherwise collapse to the prior and carry no information
        (posterior collapse); the KL term below this floor is not penalized,
        which keeps those dimensions alive.
    kl_warmup_epochs : int
        Ramp `beta` linearly from 0 over this many epochs. A full KL penalty
        from step 0 tends to collapse the latent space before the decoder can
        use it.
    pad_class_index : int
        `ProteinOneHotEncoder` pads short sequences with ALL-ZERO rows, which
        are not valid distributions: under a categorical loss they contribute
        nothing, leaving the decoder free to emit anything past the end of a
        sequence. Those rows are mapped to this class -- the alphabet's 'X'
        padding letter (index 20) -- so the model learns where a sequence
        *ends* and `clean_generated_sequence` can strip the tail.
    """

    def __init__(
        self,
        encoder,
        decoder,
        reconstruction: str = "categorical",
        beta: float = 1.0,
        free_bits: float = 0.05,
        kl_warmup_epochs: int = 10,
        pad_class_index: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction = reconstruction
        self.beta = beta
        self.free_bits = free_bits
        self.kl_warmup_epochs = kl_warmup_epochs
        self.pad_class_index = pad_class_index
        # Updated by `KLWarmup` (see scripts/train_vae.py); a plain Python
        # float would be captured as a constant by the compiled train_step.
        self.kl_weight = tf.Variable(1.0 if kl_warmup_epochs <= 0 else 0.0, trainable=False, dtype="float32")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.accuracy_tracker = keras.metrics.Mean(name="residue_accuracy")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.accuracy_tracker,
        ]

    def _targets(self, data):
        """(batch, pos, alphabet) target distributions, padding mapped to 'X'."""
        target = tf.squeeze(data, axis=-1)
        # An all-zero row is padding, not a residue.
        is_pad = tf.reduce_sum(target, axis=-1) < 0.5
        pad_onehot = tf.one_hot(
            tf.fill(tf.shape(is_pad), self.pad_class_index), tf.shape(target)[-1]
        )
        return tf.where(is_pad[..., None], pad_onehot, target)

    def _compute_losses(self, data):
        z_mean, z_log_var, z = self.encoder(data, training=True)
        reconstruction = self.decoder(z, training=True)

        # Keep z_log_var in a range `exp` can represent. Unclipped, one large
        # early step makes exp(z_log_var) overflow to inf, the KL term goes
        # NaN, and every weight follows -- which is exactly what happened at
        # the 400-residue window (loss: nan from the first epoch) while the
        # same code was stable at 60.
        z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
        # z_mean enters the KL term squared, so an unbounded excursion there
        # overflows just as easily.
        z_mean = tf.clip_by_value(z_mean, -1e4, 1e4)

        if self.reconstruction == "categorical":
            target = self._targets(data)
            pred = tf.squeeze(reconstruction, axis=-1)
            # Keras' own categorical cross-entropy rather than a hand-rolled
            # log: it normalizes and epsilon-guards internally.
            per_position = keras.losses.categorical_crossentropy(target, pred, axis=-1)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(per_position, axis=-1))
            accuracy = tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.argmax(target, axis=-1), tf.argmax(pred, axis=-1)),
                    "float32",
                )
            )
        else:
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            accuracy = tf.constant(0.0)

        # Per-dimension KL, with a free-bits floor applied before summing so
        # one informative dimension cannot pay for a collapsed one.
        kl_per_dim = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_per_dim = tf.maximum(kl_per_dim, self.free_bits)
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_per_dim, axis=1))

        total_loss = reconstruction_loss + self.beta * self.kl_weight * kl_loss
        return total_loss, reconstruction_loss, kl_loss, accuracy

    def train_step(self, data):
        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, kl_loss, accuracy = self._compute_losses(data)
        grads = tape.gradient(total_loss, self.trainable_weights)
        # Drop non-finite gradients instead of applying them. A single bad
        # step (the KL term can spike hard early at a long sequence window)
        # otherwise writes NaN into every weight, and the run is dead from
        # then on -- it reports `loss: nan` for all remaining epochs and
        # saves unusable weights. Zeroing the step loses one update instead.
        grads = [
            tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) if g is not None else g
            for g in grads
        ]
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.accuracy_tracker.update_state(accuracy)
        return {m.name.replace("total_loss", "loss"): m.result() for m in self.metrics}

    def test_step(self, data):
        """
        Validation pass.

        Without this, `fit(validation_data=...)` fell back to calling the
        model's `__call__`, which a subclassed VAE with a custom loss does not
        define -- so there was no way to see whether the VAE was overfitting.
        """
        total_loss, reconstruction_loss, kl_loss, accuracy = self._compute_losses(data)
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.accuracy_tracker.update_state(accuracy)
        return {m.name.replace("total_loss", "loss"): m.result() for m in self.metrics}

    def latent_statistics(self, data, batch_size: int = 128, labels=None) -> dict:
        """
        Per-dimension mean/std of `z_mean` over `data` (the aggregate posterior).

        Generation samples the prior, but the aggregate posterior only matches
        that prior if the KL term was strong enough during training. Recording
        the real statistics lets `generate_antibody_sequence` sample where the
        data actually lives instead of guessing, which is the difference
        between decoding plausible sequences and decoding off-manifold noise.
        """
        z_mean, _z_log_var, _z = self.encoder.predict(data, batch_size=batch_size, verbose=0)
        z_mean = np.asarray(z_mean)
        stats = {
            "latent_dim": int(z_mean.shape[1]),
            "mean": z_mean.mean(axis=0).tolist(),
            "std": z_mean.std(axis=0).tolist(),
            "n_samples": int(z_mean.shape[0]),
        }

        # Per-group statistics, when the caller can say which group each
        # sequence belongs to. The VAE is never told that heavy and light
        # chains are different things, but it separates them anyway, so the
        # centre of each family is a usable handle: sampling near one of them
        # generates that family instead of an uncontrolled mixture.
        if labels is not None:
            labels = np.asarray(labels, dtype=object)
            groups = {}
            for name in sorted({l for l in labels if l is not None}):
                selected = z_mean[labels == name]
                if len(selected) < 20:          # too few to trust a centre
                    continue
                groups[str(name)] = {
                    "mean": selected.mean(axis=0).tolist(),
                    "std": selected.std(axis=0).tolist(),
                    "n_samples": int(len(selected)),
                }
            if groups:
                stats["by_chain_type"] = groups
        return stats

    def save(self, filename):
        """Save model in 2 part"""
        filename, extension = os.path.splitext(filename)
        self.encoder.save(f'{filename}-encoder.keras')
        self.decoder.save(f'{filename}-decoder.keras')

    def reload(self, filename):
        """Reload a 2 part saved model."""
        filename, extension = os.path.splitext(filename)
        self.encoder = keras.models.load_model(f'{filename}-encoder.keras', custom_objects={'SamplingLayer': SamplingLayer})
        self.decoder = keras.models.load_model(f'{filename}-decoder.keras')
        print('Reloaded.')
