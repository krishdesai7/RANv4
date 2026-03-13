import keras


def build_generator(hidden_units: int = 64, n_layers: int = 2) -> keras.Model:
    """g(z): nominal-level events -> per-event weights."""
    inputs = keras.Input(shape=(1,))
    x = inputs
    for _ in range(n_layers):
        x = keras.layers.Dense(hidden_units, activation="relu")(x)
    x = keras.layers.Dense(1, activation="softplus")(x)
    return keras.Model(inputs, x, name="generator")


def build_discriminator(hidden_units: int = 64, n_layers: int = 2) -> keras.Model:
    """d(x): reco-level events -> data vs MC probability."""
    inputs = keras.Input(shape=(1,))
    x = inputs
    for _ in range(n_layers):
        x = keras.layers.Dense(hidden_units, activation="relu")(x)
    x = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, x, name="discriminator")
