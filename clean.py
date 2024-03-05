import os
from glob import glob
import numpy as np
import pandas as pd
import librosa
from scipy.io import wavfile
import wavio
import matplotlib.pyplot as plt

class AudioPreprocessor:
    def __init__(self, src_root, dst_root, delta_time=1.0, sampling_rate=16000, threshold=20):
        """
        Initialize the AudioPreprocessor class with paths and audio processing parameters.
        
        :param src_root: Source directory of audio files.
        :param dst_root: Destination directory for processed audio files.
        :param delta_time: Time in seconds to sample audio.
        :param sampling_rate: Sampling rate to downsample audio.
        :param threshold: Threshold magnitude for signal envelope.
        """
        self.src_root = src_root
        self.dst_root = dst_root
        self.delta_time = delta_time
        self.sampling_rate = sampling_rate
        self.threshold = threshold

    @staticmethod
    def envelope(y, rate, threshold):
        """
        Compute the signal envelope and apply a threshold to mask the signal.
        
        :param y: Audio signal.
        :param rate: Sampling rate of the audio signal.
        :param threshold: Threshold value for masking.
        :return: A mask array and the signal envelope.
        """
        mask = []
        y = pd.Series(y).apply(np.abs)
        y_mean = y.rolling(window=int(rate/20), min_periods=1, center=True).max()
        mask = y_mean > threshold
        return mask.values, y_mean

    @staticmethod
    def downsample_mono(path, sr):
        """
        Downsample and convert stereo audio to mono.
        
        :param path: Path to the audio file.
        :param sr: Target sampling rate.
        :return: Downsampled audio signal and its sampling rate.
        """
        obj = wavio.read(path)
        wav = obj.data.astype(np.float32, order='F')
        rate = obj.rate
        wav = librosa.to_mono(wav.T) if wav.ndim > 1 else wav.reshape(-1)
        wav = librosa.resample(wav, orig_sr=rate, target_sr=sr)
        wav = wav.astype(np.int16)
        return sr, wav

    def save_sample(self, sample, rate, target_dir, fn, ix):
        """
        Save a sample of the audio signal.
        
        :param sample: Audio signal sample to save.
        :param rate: Sampling rate of the audio signal.
        :param target_dir: Directory to save the audio signal sample.
        :param fn: Original filename of the audio signal.
        :param ix: Index of the sample in the original audio signal.
        """
        fn = fn.split('.wav')[0]
        dst_path = os.path.join(target_dir, f'{fn}_{ix}.wav')
        if not os.path.exists(dst_path):
            wavfile.write(dst_path, rate, sample)

    @staticmethod
    def check_dir(path):
        """
        Check if a directory exists, and create it if it doesn't.
        
        :param path: Directory path to check.
        """
        os.makedirs(path, exist_ok=True)

    def process(self):
        """
        Process all .wav files in the source directory, applying downsampling, mono conversion,
        signal enveloping, and splitting based on delta time.
        """
        wav_paths = glob(f'{self.src_root}/**/*.wav', recursive=True)
        self.check_dir(self.dst_root)
        
        for src_path in wav_paths:
            cls_dir = os.path.basename(os.path.dirname(src_path))
            target_dir = os.path.join(self.dst_root, cls_dir)
            self.check_dir(target_dir)
            rate, wav = self.downsample_mono(src_path, self.sampling_rate)
            mask, _ = self.envelope(wav, rate, self.threshold)
            wav = wav[mask]
            delta_sample = int(self.delta_time * rate)

            if wav.shape[0] < delta_sample:
                sample = np.zeros(shape=(delta_sample,), dtype=np.int16)
                sample[:wav.shape[0]] = wav
                self.save_sample(sample, rate, target_dir, os.path.basename(src_path), 0)
            else:
                trunc = wav.shape[0] % delta_sample
                for cnt, i in enumerate(range(0, wav.shape[0] - trunc, delta_sample)):
                    start, stop = int(i), int(i + delta_sample)
                    sample = wav[start:stop]
                    self.save_sample(sample, rate, target_dir, os.path.basename(src_path), cnt)


    def plot_envelope(self, path, threshold=None):
        """
        Plot the audio wave against its envelope to visually inspect if the wave fits the envelope.
        
        :param path: Path to the audio file to be plotted.
        :param threshold: Optional threshold to use for this specific plot. If not specified, use the class's threshold.
        """
        if threshold is None:
            threshold = self.threshold
        
        # Downsample and convert to mono as per the class configuration
        rate, wav = self.downsample_mono(path, self.sampling_rate)
        
        # Compute envelope and mask
        mask, env = self.envelope(wav, rate, threshold)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.title('Signal Envelope and Thresholding')
        plt.plot(wav, label='Original Wave', alpha=0.5)
        plt.plot(wav * mask, label='Thresholded Wave', color='green')
        plt.plot(env, label='Envelope', color='red', linestyle='--')
        plt.legend()
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.show()
        

if __name__ == '__main__':
    preprocessor = AudioPreprocessor(src_root='wavfiles', dst_root='clean', delta_time=1.0, sampling_rate=16000, threshold=20)

    # Test and plot the envelope for the specified audio file
    # audio_file_path = 'wavfiles/Hi_hat/3a3d0279.wav'
    # preprocessor.plot_envelope(audio_file_path)
    
    # Clean the audio files
    preprocessor.process()
