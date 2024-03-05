from tensorflow.keras.models import load_model
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from data_generator import DataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


class PredictionHelper:
    def __init__(self, model_fn='models/conv1d.h5', src_dir='clean_noisy', dt=1.0, sr=16000, threshold=20, batch_size=32):
        self.selected_model = model_fn
        self.src_dir = src_dir
        self.delta_time = dt
        self.sample_rate = sr
        self.threshold = sr
        self.batch_size = batch_size
        self.N_CLASSES = len(os.listdir(self.src_dir))
        
        self.loaded_model = self.get_model()
        self.init_data()
  
    def get_model(self):
        return load_model(self.selected_model, {
                'STFT': STFT,
                'Magnitude': Magnitude,
                'ApplyFilterbank': ApplyFilterbank,
                'MagnitudeToDecibel': MagnitudeToDecibel
            })
    
    def init_data(self):
        #  get Data from root
        wav_paths = glob(f'{self.src_dir}/**', recursive=True)
        wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]
        # get labels from folder names
        self.classes = sorted(os.listdir(self.src_dir))
        
        # Create an ecoder for our label
        # i.e change "Label1" to [1, 0, 0, 0, .......]
        encoder = LabelEncoder()
        encoder.fit(self.classes)
        
        labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
        labels = encoder.transform(labels)
        
        # check for the sizes of the test, raise error if it's smaller than number of classes
        assert len(wav_paths) >= self.batch_size, 'Number of samples must be >= batch_size'
        
        # Create data generators
        self.batch = DataGenerator(
                                    wav_paths=wav_paths, 
                                    labels=labels, 
                                    sample_rate=self.sample_rate, 
                                    delta_time=self.delta_time, 
                                    n_classes=self.N_CLASSES, 
                                    batch_size=self.batch_size
                                )
        
    def predict(self):
        results = []
        for cur_batch in tqdm(range(0, self.batch.__len__())):
            X_batch, X_label = self.batch.__getitem__(index=cur_batch)
            y_pred = self.loaded_model.predict(X_batch)
            for actual, pred in zip(X_label, y_pred):
                actual = self.classes[np.argmax(actual)]
                predicted = self.classes[np.argmax(pred)]
                results.append([actual, predicted])
            
        return results
            


if __name__ == '__main__':
    
    helper = PredictionHelper(
            model_fn='models/conv2d.h5',
            src_dir='clean_noisy',
            dt=1.0,
            sr=16000,
            threshold=20,
            batch_size=32
        )

    results = helper.predict()
    
    count = 0
    total = len(results)
    for i in results:
        if i[0] == i[1]:
            count += 1
    
    actual = [pair[0] for pair in results]
    predicted = [pair[1] for pair in results]
    
    # Get the unique classes
    classes = sorted(set(actual + predicted))
    
    # Generate the confusion matrix
    cm = confusion_matrix(actual, predicted, labels=classes)
    
    plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.viridis)
    plt.title('Confusion Matrix without Seaborn')
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Labeling the plot
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()
    
