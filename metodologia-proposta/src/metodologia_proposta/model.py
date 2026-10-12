import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


@tf.keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(
        self,
        d_model=128,
        num_heads=4,
        mlp_dim=256,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)

        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
        )

        self.drop1 = layers.Dropout(dropout)

        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.mlp = tf.keras.Sequential(
            [
                layers.Dense(mlp_dim, activation="gelu"),
                layers.Dropout(dropout),
                layers.Dense(d_model),
                layers.Dropout(dropout),
            ]
        )

    def call(self, x, training=False):
        h = self.norm1(x)
        h = self.attn(h, h, training=training)

        x = x + self.drop1(h, training=training)

        h = self.norm2(x)
        h = self.mlp(h, training=training)

        return x + h

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "dropout": self.dropout,
            }
        )

        return config


@tf.keras.utils.register_keras_serializable()
class TokenIndex(layers.Layer):
    def __init__(self, num_tokens, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens

    def call(self, x):
        batch_size = tf.shape(x)[0]

        indices = tf.range(self.num_tokens)[tf.newaxis, :]

        return tf.tile(
            indices,
            [batch_size, 1],
        )

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_tokens)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_tokens": self.num_tokens,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable()
class FeatureMapTokenizer(layers.Layer):
    def __init__(self, d_model=128, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.proj = layers.Dense(d_model)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        channels = tf.shape(x)[3]

        tokens = tf.reshape(
            x,
            [batch_size, height * width, channels],
        )

        return self.proj(tokens)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model})

        return config


def create_model(
    patch_size: int,
    num_classes: int,
    d_model: int = 128,
    num_heads: int = 4,
    mlp_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> Model:
    """
    Cria o modelo CNN + Transformer com informação
    da posição espacial global de cada patch.
    """

    # Entradas
    img_input = layers.Input(
        shape=(patch_size, patch_size, 1),
        name="image_input",
    )

    pos_input = layers.Input(
        shape=(2,),
        name="position_input",
    )

    # CNN
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same",
    )(img_input)

    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same",
    )(x)

    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same",
    )(x)

    x = layers.MaxPooling2D((2, 2))(x)

    # Depois de 3 MaxPooling2D:
    # 64 -> 32 -> 16 -> 8
    feature_map_size = patch_size // 8
    num_tokens = feature_map_size * feature_map_size

    # Feature map -> tokens
    tokens = FeatureMapTokenizer(
        d_model=d_model,
        name="featuremap_tokenizer",
    )(x)

    # Embedding da posição local dos tokens
    token_indices = TokenIndex(
        num_tokens,
        name="token_indices",
    )(pos_input)

    token_pos_emb = layers.Embedding(
        input_dim=num_tokens,
        output_dim=d_model,
        name="token_pos_emb",
    )(token_indices)

    # Embedding da posição global do patch
    global_pos_emb = layers.Dense(
        d_model,
        name="global_pos_proj",
    )(pos_input)

    global_pos_emb = layers.Reshape(
        (1, d_model),
        name="global_pos_expand",
    )(global_pos_emb)

    # Feature + posição local + posição global
    x = layers.Add(name="add_pos")(
        [
            tokens,
            token_pos_emb,
            global_pos_emb,
        ]
    )

    x = layers.Dropout(dropout)(x)

    # Transformer Encoder
    for i in range(num_layers):
        x = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            name=f"tx_{i}",
        )(x)

    # Classificação
    x = layers.LayerNormalization(
        epsilon=1e-6,
    )(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(
        128,
        activation="relu",
    )(x)

    x = layers.Dropout(0.5)(x)

    output = layers.Dense(
        num_classes,
        activation="softmax",
    )(x)

    model = Model(
        inputs=[img_input, pos_input],
        outputs=output,
        name="CNN_Transformer_Pos_Model",
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model