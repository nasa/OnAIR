"""Curve Characterizer, for fitting curves for association rule mining
   CODE ADAPTED FROM: 
   https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
from keras.utils.vis_utils import plot_model
from keras.models import Model, save_model, load_model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate

from src.data_driven_components.curve_characterizer.gen_data import * 
from src.data_driven_components.curve_characterizer.file_io import * 
from src.data_driven_components.curve_characterizer.util import * 

#TODO Write tests 
class CurveCharacterizer:
    def __init__(self, main_data_path='', prepModel=False):
        # Could have these passed to init or inferred. 
        # Hardcoded for now.
        self.num_frames = 2000
        self.frame_size = 20
        self.root_data_path = main_data_path
        self.model_built = False 
        self.classes = {0:'increase',
                        1:'decrease',
                        2:'sinusoidal',
                        3:'constant'}

        self.model_path = os.path.dirname(os.path.realpath(__file__)) + '/models/'
        
        try:
            self.model = load_model(self.model_path + "my_model")
        except:
            self.build_characterizer()
            self.model.save(self.model_path + "my_model")
            self.model_built = True

        # if prepModel == True: 
            # self.build_characterizer()
            # self.model_built = True

    """ Trains RAISR """
    def apriori_training(self, data_train):
        return

    """ Updates based on RAISR frames """
    def update(self, frame):
        return

    """ Make characterizater """
    def build_characterizer(self, data_path=''):
        tmp_path = data_path + '/curve_characterizer_tmp/' if self.root_data_path == '' else self.root_data_path + '/curve_characterizer_tmp/'
        setup_folders(tmp_path)
        write_to_file(*gen_data(self.num_frames, self.frame_size), tmp_path, 'train')
        write_to_file(*gen_data(self.num_frames, self.frame_size), tmp_path, 'test')
        self.model = self.train_model(tmp_path)
        breakdown_folders(tmp_path)

    """ Render curve characterization """
    def predict(self, sample):
        # if self.model_built == False:
        #     self.build_characterizer()
        #     self.model_built = True
        prediction = self.model.predict([sample, sample, sample])
        return self.classes[np.argmax(prediction)]

    """ Run an experiment """
    def train_model(self, data_path, repeats=2):
        trainX, trainy, testX, testy = load_dataset(data_path)
        scores = list()
        for r in range(repeats):
            score, model = self.evaluate_model(trainX, trainy, testX, testy)
            score = score * 100.0
            print('>#%d: %.3f' % (r+1, score))
            scores.append(score)
        self.summarize_results(scores)
        return model

    """ Fit and evaluate a model"""
    def evaluate_model(self, trainX, trainy, testX, testy):
        verbose, epochs, batch_size = 0, 10, 32
        n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
        # head 1
        inputs1 = Input(shape=(n_timesteps,n_features))
        conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs1)
        drop1 = Dropout(0.5)(conv1)
        pool1 = MaxPooling1D(pool_size=2)(drop1)
        flat1 = Flatten()(pool1)
        # head 2
        inputs2 = Input(shape=(n_timesteps,n_features))
        conv2 = Conv1D(filters=64, kernel_size=5, activation='relu')(inputs2)
        drop2 = Dropout(0.5)(conv2)
        pool2 = MaxPooling1D(pool_size=2)(drop2)
        flat2 = Flatten()(pool2)
        # head 3
        inputs3 = Input(shape=(n_timesteps,n_features))
        conv3 = Conv1D(filters=64, kernel_size=11, activation='relu')(inputs3)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling1D(pool_size=2)(drop3)
        flat3 = Flatten()(pool3)

        merged = concatenate([flat1, flat2, flat3])
        dense1 = Dense(100, activation='relu')(merged)
        outputs = Dense(n_outputs, activation='softmax')(dense1)
        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit([trainX,trainX,trainX], trainy, epochs=epochs, batch_size=batch_size, verbose=0)
        _, accuracy = model.evaluate([testX,testX,testX], testy, batch_size=batch_size, verbose=0)

        return accuracy, model

    """ Summarize scores"""
    def summarize_results(self, scores):
        print(scores)
        m, s = np.mean(scores), np.std(scores)
        print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))






