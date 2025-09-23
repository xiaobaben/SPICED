import os
import pandas as pd
import numpy as np
from scipy import stats, signal
import pywt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.signal import spectrogram
from scipy.signal import butter, filtfilt, find_peaks
import mne
"""
IMPORTANT!!!
You can selectively extract different initial features for different tasks
For example, sleep-related features such as slow-wave power can be included for sleep tasks.
"""


def calculate_hjorth_params(x):
    """ Compute Hjorth"""
    diff1 = np.diff(x)
    var_0 = np.var(x)
    var_1 = np.var(diff1)
    mobility = np.sqrt(var_1 / var_0)
    diff2 = np.diff(diff1)
    var_2 = np.var(diff2)
    complexity = np.sqrt(var_2 / var_1) / mobility
    return mobility, complexity


def calculate_sleep_prototype(data, fs):
    num_channels = data.shape[0]
    all_features = []

    for ch in tqdm(range(num_channels), desc="Calculating Prototype"):
        sig = data[ch]
        """Time Domain"""
        mean = np.mean(sig)
        variance = np.var(sig)
        skewness = stats.skew(sig)
        kurt = stats.kurtosis(sig)
        mobility, complexity = calculate_hjorth_params(sig)

        """Frequency Domain"""
        freq, psd = signal.welch(sig, fs=fs, nperseg=1024)
        bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
        band_powers = []
        for low, high in bands:
            idx = np.logical_and(freq >= low, freq <= high)
            band_powers.append(np.trapz(psd[idx], freq[idx]))

        """Time-Frequency Domain"""
        coefficient = pywt.wavedec(sig, 'db4', level=3)
        wavelet_energy = [np.sum(np.square(c)) for c in coefficient]

        """Sleep Features"""
        delta_bands = [(0.5, 2), (2, 4)]
        swa_powers = [np.trapz(psd[(freq >= low) & (freq <= high)]) for (low, high) in delta_bands]

        f, t, Sxx = spectrogram(sig, fs=fs, nperseg=512, noverlap=256)
        spindle_band = (11, 16)
        idx = np.logical_and(f >= spindle_band[0], f <= spindle_band[1])
        spindle_power = np.sum(Sxx[idx, :], axis=0)
        threshold = np.mean(spindle_power) + 2 * np.std(spindle_power)
        spindle_events = np.sum(spindle_power > threshold)

        b, a = butter(4, [0.5, 4], btype='bandpass', fs=fs)
        filtered = filtfilt(b, a, sig)
        peaks, _ = find_peaks(-filtered, height=np.percentile(-filtered, 95))
        k_complex_count = len(peaks)  # K波检测

        channel_features = [
            mean, variance, mobility, complexity, skewness, kurt,
            *band_powers, *wavelet_energy, *swa_powers, spindle_events, k_complex_count]
        all_features.append(channel_features)

    feature_vector = np.array(all_features)
    print(feature_vector)
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(feature_vector).flatten()
    print(normalized_features)

    return normalized_features


def calculate_emotion_prototype(data, fs):
    num_channels = data.shape[0]
    all_features = []

    for ch in tqdm(range(num_channels), desc="Calculating Prototype"):
        sig = data[ch]
        """Time Domain"""
        mean = np.mean(sig)
        variance = np.var(sig)
        skewness = stats.skew(sig)
        kurt = stats.kurtosis(sig)
        mobility, complexity = calculate_hjorth_params(sig)

        """Frequency Domain"""
        freq, psd = signal.welch(sig, fs=fs, nperseg=1024)
        bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
        band_powers = []
        for low, high in bands:
            idx = np.logical_and(freq >= low, freq <= high)
            band_powers.append(np.trapz(psd[idx], freq[idx]))

        """Time-Frequency Domain"""
        coefficient = pywt.wavedec(sig, 'db4', level=3)
        wavelet_energy = [np.sum(np.square(c)) for c in coefficient]

        channel_features = [
            mean, variance, mobility, complexity, skewness, kurt,
            *band_powers, *wavelet_energy, ]
        all_features.append(channel_features)

    feature_vector = np.array(all_features)
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(feature_vector).flatten()

    return normalized_features


def calculate_motor_prototype(data, fs):
    num_channels = data.shape[0]
    all_features = []

    for ch in tqdm(range(num_channels), desc="Calculating Prototype"):
        sig = data[ch]
        """Time Domain"""
        mean = np.mean(sig)
        variance = np.var(sig)
        skewness = stats.skew(sig)
        kurt = stats.kurtosis(sig)
        mobility, complexity = calculate_hjorth_params(sig)

        """Frequency Domain"""
        freq, psd = signal.welch(sig, fs=fs, nperseg=1024)
        bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
        band_powers = []
        for low, high in bands:
            idx = np.logical_and(freq >= low, freq <= high)
            band_powers.append(np.trapz(psd[idx], freq[idx]))

        """Time-Frequency Domain"""
        coefficient = pywt.wavedec(sig, 'db4', level=3)
        wavelet_energy = [np.sum(np.square(c)) for c in coefficient]

        channel_features = [
            mean, variance, mobility, complexity, skewness, kurt,
            *band_powers, *wavelet_energy, ]
        all_features.append(channel_features)

    feature_vector = np.array(all_features)
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(feature_vector).flatten()

    return normalized_features


def save_sleep_prototype():
    path = "/data/xbb/Synaptic Homeostasis/dataset/ISRUC/raw_data"
    path_list = os.listdir(path)

    for sub in path_list:
        sub_path = path + f'/{sub}'
        print(sub_path)
        folder = f"/data/xbb/Synaptic Homeostasis/dataset/ISRUC/prototype3"
        sub_data = np.load(sub_path)
        print(sub_data.shape)
        sub_prototype = calculate_sleep_prototype(sub_data, fs=200)
        print(sub_prototype.shape)
        save_data_path = folder + f"/{os.path.split(sub_path)[1]}"
        np.save(save_data_path, sub_prototype)


def save_emotion_prototype():
    save_path = "/data/xbb/Synaptic Homeostasis/dataset/FACED/prototype3"
    ori_path = f"/data/cyn/FACED/Processed_data"

    path = [str(i) for i in range(0, 123)]
    path = [i.zfill(3) for i in path]
    print(len(path))
    for idx in path:
        file_path = ori_path + f"/sub{idx}.pkl"
        x_data = pd.read_pickle(file_path)
        x_data = x_data.transpose(1, 0, 2).reshape(32, -1)
        sub_prototype = calculate_emotion_prototype(x_data, fs=250)
        print(sub_prototype.shape)
        save_data_path = save_path + f"/{int(idx)}"
        np.save(save_data_path, sub_prototype)


def save_motor_prototype():
    data_path = []
    save_path = "/data/xbb/Synaptic Homeostasis/dataset/Physionet/prototype3"
    path_ = f"/data/datasets/eeg-motor-movementimagery-dataset-1.0.0/files/"
    path_list = [f"S{str(i).zfill(3)}" for i in range(1, 110)]
    for path in path_list:
        data_path.append(path_ + f"{path}/")
    for idx, subject in enumerate(data_path):
        subject_path_edf = []
        subject_path_list = os.listdir(subject)
        for path in sorted(subject_path_list):
            if path[-3:] == 'edf' and path[-6:-4] in ['04', '06', '08', '10', '12', '14']:
                subject_path_edf.append([subject + f"{path}", subject + f"{path}.event"])
        data = []
        for file in subject_path_edf:
            edf_file, event_file = file[0], file[1]
            raw = mne.io.read_raw_edf(edf_file)
            data.append(raw.get_data())
        data = np.concatenate(data, axis=1)
        sub_prototype = calculate_emotion_prototype(data, fs=160)
        print(sub_prototype.shape)
        save_data_path = save_path + f"/{int(idx)}"
        np.save(save_data_path, sub_prototype)


if __name__ == '__main__':
    save_emotion_prototype()
    save_sleep_prototype()
    save_motor_prototype()

