# `VariationalLossLayer` is deliberately NOT re-exported here. It is a PyTorch
# layer left over from the FIDLE course material this project started from; it
# is used nowhere, and cannot work with this VAE in any case, which computes
# its own loss in `VAE.train_step`. Importing it from the package __init__
# pulled the whole of `torch` (~900 MB installed) into every process that
# touched `modules.layers` -- including the dashboard, which never needs it.
# Import it explicitly from its own module if you ever want it.
from intelligent_antibodies.modules.layers.SamplingLayer import SamplingLayer

__all__ = ["SamplingLayer"]
