import numpy as np
import librosa
import soundfile as sf
import pickle

def create_silence_and_extract_features(duration=5, sr=22050, n_mels=128, n_fft=2048,hop_length=512):
    """
    Create a silence audio segment and extract its log-mel features.

    Args:
    duration(float): Duration of the slience in seconds.
    sr(int): Sample rate of the audio.
    n_mels(int): Number of mel bands to generate
    n_fft(int): Lenth of the FFT window.
    hop_lenth(int): Number of samples between successive frames.

    Returns:
    numpy.ndarray: Log-mel spectrogram of the silence.
    """

    slience = np.zeros(int(duration * sr))
    sf.write('slience.wav', slience, sr)

    mel_spectrogram = librosa.feature.melspectrogram(y=slience, sr=sr, n_mels=n_mels, 
                                                    n_fft=n_fft, hop_length=hop_length)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    save_path = 'slience.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(log_mel_spectrogram, f)

    return log_mel_spectrogram

create_silence_and_extract_features()