# -*- coding: utf-8 -*-
import os
from glob import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from data_generator import DataGenerator

class ModelTrainer():
    def __init__(self, model='conv1d', root='clean', batch_size=32, delta_time=1, sample_rate=16000):
        self.model_type = model
        self.root = root
        self.batch_size = batch_size
        self.dt = delta_time
        self.sr = sample_rate
        self.N_CLASSES = len(os.listdir(self.root))
        
        self.model = self.get_model()
        
    def get_model(self):
        params = {
          'N_CLASSES':self.N_CLASSES,
          'SR':self.sr,
          'DT':self.dt
        }
        if self.model_type == 'conv1d':
            from models.conv1d import Conv1D
            return Conv1D(**params)
        elif self.model_type == 'conv2d':
            from models.conv2d import Conv2D
            return Conv2D(**params)
        elif self.model_type == 'lstm':
            from models.lstm import LSTM
            return LSTM(**params)
        else:
            raise Exception(f'Invalid model type - {model}')
            
    def init_data_genrator(self, test_size=0.1, random_state=0):
        #  get Data from root
        wav_paths = glob(f'{self.root}/**', recursive=True)
        wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]
        # get labels from folder names
        classes = sorted(os.listdir(self.root))
        
        # Create an ecoder for our label
        # i.e change "Label1" to [1, 0, 0, 0, .......]
        encoder = LabelEncoder()
        encoder.fit(classes)
        
        labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
        labels = encoder.transform(labels)
        
        # split our data into training and testing set 
        X_train, X_test, Y_train, Y_test = train_test_split(
                                                wav_paths, 
                                                labels, 
                                                test_size=test_size, 
                                                random_state=random_state
                                            )
        
        # check for the sizes of the test, raise error if it's smaller than number of classes
        assert len(Y_train) >= self.batch_size, 'Number of train samples must be >= batch_size'
        
        if len(set(Y_train)) != self.N_CLASSES:
            warnings.warn('Found {}/{} classes in training data. Increase data size or change random_state.'.format(len(set(Y_train)), self.N_CLASSES))
        
        if len(set(Y_test)) != self.N_CLASSES:
            warnings.warn('Found {}/{} classes in validation data. Increase data size or change random_state.'.format(len(set(Y_test)), self.N_CLASSES))
        
        # Create data generators
        self.train_batch_generator = DataGenerator(
                                    wav_paths=X_train, 
                                    labels=Y_train, 
                                    sample_rate=self.sr, 
                                    delta_time=self.dt, 
                                    n_classes=self.N_CLASSES, 
                                    batch_size=self.batch_size
                                )
        
        self.test_batch_generator = DataGenerator(
                                    wav_paths=X_test, 
                                    labels=Y_test, 
                                    sample_rate=self.sr, 
                                    delta_time=self.dt, 
                                    n_classes=self.N_CLASSES, 
                                    batch_size=self.batch_size
                                )
        
    def train(self, epochs=30, verbose=1, checkpoint_path=f'models/model.h5', log_path=f'logs/model_logs.csv'):
        cp = ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True, 
                save_weights_only=False,
                mode='auto', 
                save_freq='epoch', 
                verbose=1
            )
        
        logger = CSVLogger(log_path, append=False)
        
        if self.model:
            self.model.fit(
                    self.train_batch_generator,
                    validation_data=self.test_batch_generator,
                    epochs=epochs,
                    verbose=1,
                    callbacks=[logger, cp]
                )

if __name__ == '__main__':
    trainer = ModelTrainer(
                    model='lstm', 
                    root='clean', 
                    batch_size=32, 
                    delta_time=1, 
                    sample_rate=16000
                )
    
    trainer.init_data_genrator()
    
    trainer.train(
        checkpoint_path=f'models/{trainer.model_type}.h5', 
        log_path=f'logs/{trainer.model_type}_logs.csv'
    )
    
