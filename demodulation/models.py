from tensorflow import keras


def make_wave_model():
    return keras.Sequential([
        keras.layers.Conv1D(32, 7, activation='relu', padding='same', input_shape=(None, 1)),
        keras.layers.Conv1D(32, 9, activation='relu', padding='same', strides=4),
        keras.layers.Conv1D(32, 9, activation='relu', padding='same', strides=4),
        keras.layers.Conv1D(32, 9, activation='relu', padding='same', strides=4),
        keras.layers.Conv1D(16, 16, padding='same', strides=16),
    ])


def make_sg_model():
    return keras.Sequential([
        keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same', input_shape=(None, 257, 1)),
        keras.layers.Conv2D(2, (3, 3), activation='relu', padding='same'),
        keras.layers.Reshape((-1, 514)),
        keras.layers.Conv1D(128, 9, activation='relu', padding='same', strides=4),
        keras.layers.Conv1D(16, 9, activation='relu', padding='same', strides=4),
        keras.layers.Conv1D(16, 9, activation='relu', padding='same', strides=4),
        keras.layers.Conv1D(16, 16, padding='same', strides=16),
    ])


def make_common_model():
    wave_input = keras.layers.Input((None, 1))
    wave_features = keras.layers.Conv1D(16, 9, padding='same')(wave_input)
    wave_features = keras.layers.LeakyReLU()(wave_features)
    wave_features = keras.layers.Conv1D(64, 9, padding='same')(wave_features)
    wave_features = keras.layers.LeakyReLU()(wave_features)
    wave_features = keras.layers.Conv1D(256, 9, padding='same')(wave_features)
    wave_features = keras.layers.LeakyReLU()(wave_features)
    # wave_features = keras.layers.LayerNormalization()(wave_features)

    spectrogram_input = keras.layers.Input((None, 257, 1))
    spectrogram_features = keras.layers.Conv2D(8, (3, 3), padding='same')(spectrogram_input)
    spectrogram_features = keras.layers.LeakyReLU()(spectrogram_features)
    spectrogram_features = keras.layers.Conv2D(4, (3, 3), padding='same')(spectrogram_features)
    spectrogram_features = keras.layers.LeakyReLU()(spectrogram_features)
    spectrogram_features = keras.layers.Reshape((-1, 1028))(spectrogram_features)
    spectrogram_features = keras.layers.Conv1D(512, 3, padding='same')(spectrogram_features)
    spectrogram_features = keras.layers.LeakyReLU()(spectrogram_features)
    spectrogram_features = keras.layers.Conv1D(256, 3, padding='same')(spectrogram_features)
    spectrogram_features = keras.layers.LeakyReLU()(spectrogram_features)
    # spectrogram_features = keras.layers.LayerNormalization()(spectrogram_features)

    all_features = keras.layers.concatenate([wave_features, spectrogram_features], axis=2)
    all_features = keras.layers.Conv1D(512, 9, padding='same', strides=4)(all_features)
    all_features = keras.layers.LeakyReLU()(all_features)
    all_features = keras.layers.Conv1D(512, 9, padding='same', strides=4)(all_features)
    all_features = keras.layers.LeakyReLU()(all_features)
    # all_features = keras.layers.LayerNormalization()(all_features)
    all_features = keras.layers.Conv1D(512, 9, padding='same', strides=4)(all_features)
    all_features = keras.layers.LeakyReLU()(all_features)
    prediction = keras.layers.Conv1D(16, 16, padding='same', strides=16)(all_features)
    return keras.Model([wave_input, spectrogram_input], prediction)


def make_pooled_model():
    wave_input = keras.layers.Input((None, 1))
    wave_features = keras.layers.Conv1D(16, 9, padding='same')(wave_input)
    wave_features = keras.layers.LeakyReLU()(wave_features)
    wave_features = keras.layers.Conv1D(64, 9, padding='same')(wave_features)
    wave_features = keras.layers.LeakyReLU()(wave_features)
    wave_features = keras.layers.MaxPool1D(2, 2, padding='same')(wave_features)
    wave_features = keras.layers.Conv1D(128, 9, padding='same')(wave_features)
    wave_features = keras.layers.LeakyReLU()(wave_features)
    wave_features = keras.layers.Conv1D(128, 9, padding='same')(wave_features)
    wave_features = keras.layers.LeakyReLU()(wave_features)
    wave_features = keras.layers.MaxPool1D(2, 2, padding='same')(wave_features)

    spectrogram_input = keras.layers.Input((None, 257, 1))
    spectrogram_features = keras.layers.Conv2D(8, (3, 3), padding='same')(spectrogram_input)
    spectrogram_features = keras.layers.LeakyReLU()(spectrogram_features)
    spectrogram_features = keras.layers.Conv2D(8, (3, 3), padding='same')(spectrogram_features)
    spectrogram_features = keras.layers.LeakyReLU()(spectrogram_features)
    spectrogram_features = keras.layers.MaxPool2D(2, 2, padding='same')(spectrogram_features)
    spectrogram_features = keras.layers.Conv2D(16, (3, 3), padding='same')(spectrogram_features)
    spectrogram_features = keras.layers.MaxPool2D(2, 2, padding='same')(spectrogram_features)
    spectrogram_features = keras.layers.LeakyReLU()(spectrogram_features)
    spectrogram_features = keras.layers.Reshape((-1, 1040))(spectrogram_features)
    spectrogram_features = keras.layers.Conv1D(512, 3, padding='same')(spectrogram_features)
    spectrogram_features = keras.layers.LeakyReLU()(spectrogram_features)
    spectrogram_features = keras.layers.Conv1D(128, 3, padding='same')(spectrogram_features)
    spectrogram_features = keras.layers.LeakyReLU()(spectrogram_features)

    all_features = keras.layers.concatenate([wave_features, spectrogram_features], axis=2)
    all_features = keras.layers.Conv1D(128, 9, padding='same')(all_features)
    all_features = keras.layers.LeakyReLU()(all_features)
    all_features = keras.layers.MaxPool1D(2, 2, padding='same')(all_features)
    all_features = keras.layers.Conv1D(128, 9, padding='same')(all_features)
    all_features = keras.layers.LeakyReLU()(all_features)
    all_features = keras.layers.MaxPool1D(2, 2, padding='same')(all_features)
    all_features = keras.layers.Conv1D(128, 9, padding='same', strides=4)(all_features)
    all_features = keras.layers.LeakyReLU()(all_features)
    prediction = keras.layers.Conv1D(16, 16, padding='same', strides=16)(all_features)
    return keras.Model([wave_input, spectrogram_input], prediction)


def residual_block_identity(conv_block, channels, conv_shape, incoming):
    convolved = conv_block(channels, conv_shape, padding='same')(incoming)
    convolved = keras.layers.LeakyReLU()(convolved)
    convolved = conv_block(channels, conv_shape, padding='same')(convolved)
    combined = keras.layers.add([incoming, convolved])
    return keras.layers.LeakyReLU()(combined)


def residual_block_increase(conv_block, channels, conv_shape, incoming):
    convolved = conv_block(channels // 2, conv_shape, padding='same')(incoming)
    convolved = keras.layers.LeakyReLU()(convolved)
    convolved = conv_block(channels, conv_shape, padding='same')(convolved)
    shortcut_projection = conv_block(channels, 1 if conv_block == keras.layers.Conv1D else (1, 1))(incoming)
    combined = keras.layers.add([shortcut_projection, convolved])
    return keras.layers.LeakyReLU()(combined)


def make_residual_model():
    wave_input = keras.layers.Input((None, 1))
    wave_features = keras.layers.Conv1D(4, 13, padding='same')(wave_input)
    wave_features = keras.layers.LeakyReLU()(wave_features)
    wave_features = residual_block_increase(keras.layers.Conv1D, 16, 13, wave_features)
    wave_features = keras.layers.MaxPool1D(2, 2, padding='same')(wave_features)
    wave_features = residual_block_increase(keras.layers.Conv1D, 64, 13, wave_features)
    wave_features = keras.layers.MaxPool1D(2, 2, padding='same')(wave_features)
    wave_features = residual_block_increase(keras.layers.Conv1D, 256, 13, wave_features)

    spectrogram_input = keras.layers.Input((None, 257, 1))
    spectrogram_features = keras.layers.Conv2D(4, (5, 5), padding='same')(spectrogram_input)
    spectrogram_features = keras.layers.LeakyReLU()(spectrogram_features)
    spectrogram_features = residual_block_increase(keras.layers.Conv2D, 16, (5, 5), spectrogram_features)
    spectrogram_features = keras.layers.MaxPool2D(2, 2, padding='same')(spectrogram_features)
    spectrogram_features = residual_block_identity(keras.layers.Conv2D, 16, (5, 5), spectrogram_features)
    spectrogram_features = keras.layers.MaxPool2D(2, 2, padding='same')(spectrogram_features)
    spectrogram_features = keras.layers.Reshape((-1, 1040))(spectrogram_features)
    spectrogram_features = keras.layers.Conv1D(256, 9, padding='same')(spectrogram_features)
    spectrogram_features = keras.layers.LeakyReLU()(spectrogram_features)
    spectrogram_features = residual_block_identity(keras.layers.Conv1D, 256, 13, spectrogram_features)

    all_features = keras.layers.concatenate([wave_features, spectrogram_features], axis=2)
    all_features = residual_block_identity(keras.layers.Conv1D, 512, 13, all_features)
    all_features = keras.layers.MaxPool1D(2, 2, padding='same')(all_features)
    all_features = residual_block_identity(keras.layers.Conv1D, 512, 13, all_features)
    all_features = residual_block_identity(keras.layers.Conv1D, 512, 13, all_features)
    all_features = keras.layers.MaxPool1D(2, 2, padding='same')(all_features)
    all_features = keras.layers.Conv1D(512, 13, padding='same', strides=4)(all_features)
    all_features = keras.layers.LeakyReLU()(all_features)
    prediction = keras.layers.Conv1D(16, 16, padding='same', strides=16)(all_features)
    return keras.Model([wave_input, spectrogram_input], prediction)
