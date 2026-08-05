import keras


def build_generator(dim: int = 1, hidden_units: int = 64, n_layers: int = 2) -> keras.Model:
    """g(z): nominal-level events -> per-event weights."""
    inputs = keras.Input(shape=(dim,), dtype="float64")
    x = inputs
    for _ in range(n_layers):
        x = keras.layers.Dense(hidden_units, activation="relu", dtype="float64")(x)
    x = keras.layers.Dense(1, activation="softplus", dtype="float64")(x)
    return keras.Model(inputs, x, name="generator")


def build_discriminator(dim: int = 1, hidden_units: int = 64, n_layers: int = 2) -> keras.Model:
    """d(x): reco-level events -> data vs MC probability."""
    inputs = keras.Input(shape=(dim,), dtype="float64")
    x = inputs
    for _ in range(n_layers):
        x = keras.layers.Dense(hidden_units, activation="relu", dtype="float64")(x)
    x = keras.layers.Dense(1, activation="sigmoid", dtype="float64")(x)
    return keras.Model(inputs, x, name="discriminator")
