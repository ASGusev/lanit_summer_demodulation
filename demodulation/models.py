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
    wave_features = keras.layers.Conv1D(32, 9, padding='same')(wave_features)
    wave_features = keras.layers.LeakyReLU()(wave_features)
    wave_features = keras.layers.Conv1D(64, 9, padding='same')(wave_features)
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
    spectrogram_features = keras.layers.Conv1D(64, 3, padding='same')(spectrogram_features)
    spectrogram_features = keras.layers.LeakyReLU()(spectrogram_features)
    # spectrogram_features = keras.layers.LayerNormalization()(spectrogram_features)

    all_features = keras.layers.concatenate([wave_features, spectrogram_features], axis=2)
    all_features = keras.layers.Conv1D(64, 9, padding='same', strides=4)(all_features)
    all_features = keras.layers.LeakyReLU()(all_features)
    all_features = keras.layers.Conv1D(64, 9, padding='same', strides=4)(all_features)
    all_features = keras.layers.LeakyReLU()(all_features)
    # all_features = keras.layers.LayerNormalization()(all_features)
    all_features = keras.layers.Conv1D(32, 9, padding='same', strides=4)(all_features)
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
