
import random
import pickle
import sys

import os
from collections import defaultdict

import scipy.io.wavfile

sys.path.append("EmotionDetection/api")
import Analysis.Vokaturi as Vokaturi

import numpy as np
import xlsxwriter
from sklearn.neural_network import MLPRegressor

class NeuralNetworkModelEmotion:
    def __init__(self, modelDict=None):
        self.modelDict = modelDict

    def predict_single_sample(self, filename):

        if self.modelDict is None:
            self.modelDict = pickle.load(open("Analysis/ModelStorage/emotionnn.pkl", "rb"))

        print("Loading library...")
        Vokaturi.load("Analysis/OpenVokaturi-3-0-win64.dll")
        print("Analyzed by: %s" % Vokaturi.versionAndLicense())

        print("Reading sound file...")
        (sample_rate, samples) = scipy.io.wavfile.read(filename)
        print("   sample rate %.3f Hz" % sample_rate)

        print("Allocating Vokaturi sample array...")
        buffer_length = len(samples)
        print("   %d samples, %d channels" % (buffer_length, samples.ndim))
        c_buffer = Vokaturi.SampleArrayC(buffer_length)
        if samples.ndim == 1:  # mono
            c_buffer[:] = samples[:] / 32768.0
        else:  # stereo
            c_buffer[:] = 0.5 * (samples[:, 0] + 0.0 + samples[:, 1]) / 32768.0

        print("Creating VokaturiVoice...")
        voice = Vokaturi.Voice(sample_rate, buffer_length)

        print("Filling VokaturiVoice with samples...")
        voice.fill(buffer_length, c_buffer)

        print("Extracting emotions from VokaturiVoice...")
        quality = Vokaturi.Quality()
        emotionProbabilities = Vokaturi.EmotionProbabilities()
        voice.extract(quality, emotionProbabilities)

        if quality.valid:
            print("Neutral: %.3f" % emotionProbabilities.neutrality)
            print("Happy: %.3f" % emotionProbabilities.happiness)
            print("Sad: %.3f" % emotionProbabilities.sadness)
            print("Angry: %.3f" % emotionProbabilities.anger)
            print("Fear: %.3f" % emotionProbabilities.fear)

            neutral = emotionProbabilities.neutrality
            happy = emotionProbabilities.happiness
            sad = emotionProbabilities.sadness
            angry = emotionProbabilities.anger
            fear = emotionProbabilities.fear

            feature = []

            feature.append(neutral)

            feature.append(happy)

            feature.append(sad)

            feature.append(angry)

            feature.append(fear)

            features = np.array(feature)
            features = features.reshape(1, -1)

            openness = self.modelDict['openness'].predict(features)

            extraversion = self.modelDict['extraversion'].predict(features)

            neuroticism = self.modelDict['neuroticism'].predict(features)

            agreeableness = self.modelDict['agreeableness'].predict(features)

            conscientiousness = self.modelDict['conscientiousness'].predict(features)

            resultsDict =  defaultdict(float)

            resultsDict['openness'] = openness
            resultsDict['extraversion'] = extraversion
            resultsDict['neuroticism'] = neuroticism
            resultsDict['agreeableness'] = agreeableness
            resultsDict['conscientiousness'] = conscientiousness

            voice.destroy()
            return resultsDict
        else:
            print("Not enough sonorancy to determine emotions")

        voice.destroy()

# NeuralNetwork = NeuralNetworkModelEmotion()
#
# params = {
#     'hidden_layer_sizes': (190, 190, 190, 190,),
#     'activation': 'relu',
#     'solver': 'adam',
#     'learning_rate': 'adaptive',
#     'alpha': 0.00001
# }
#
# NeuralNetwork.test()
#