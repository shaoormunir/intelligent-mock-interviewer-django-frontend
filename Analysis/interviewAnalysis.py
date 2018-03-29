import pickle
import subprocess
from statistics import mean

import pathlib

from Analysis.EmotionNeuralNetwork import NeuralNetworkModelEmotion
from Analysis.neuralnetworkmodel import NeuralNetworkModel


class InterviewAnalysis:
    def __init__(self, videoName = None):
        self.videoName = videoName

    def setVideoName(self, videoName):
        self.videoName = videoName

    def analyzeVideo(self):

        if self.videoName is not None:
            neuralNetworkModelPickle = open("Analysis/ModelStorage/nnlivetest.pickle", "rb")
            NNModelDict = pickle.load(neuralNetworkModelPickle)

            if NNModelDict is not None:
                nnetwork = NeuralNetworkModel(NNModelDict)
                openness, extraversion, neuroticism, agreeableness, conscientiousness = nnetwork.predict_single_video(self.videoName)
                openness_audio, extraversion_audio, conscientiousness_audio, neuroticism_audio, agreeableness_audio = self.analyzeAudio()

                if openness_audio is not None:
                    openness["average"] = (openness["average"] + openness_audio)/2
                    extraversion["average"] = (extraversion["average"] + extraversion_audio)/2
                    neuroticism["average"] = (neuroticism["average"] + neuroticism_audio)/2
                    conscientiousness["average"] = (conscientiousness["average"]+conscientiousness_audio)/2
                    agreeableness["average"] = (agreeableness["average"]+agreeableness_audio)/2

                    openness["audio"] = openness_audio
                    extraversion["audio"] = extraversion_audio
                    conscientiousness["audio"] = conscientiousness_audio
                    neuroticism["audio"] = neuroticism_audio
                    agreeableness["audio"] = agreeableness_audio

                return openness, extraversion, neuroticism, agreeableness, conscientiousness
        else:
            print("There was no video file found.")


    def analyzeAudio(self):
        audioFileName = self.extractAudio()

        emotionalNN = NeuralNetworkModelEmotion()

        resultsDict = emotionalNN.predict_single_sample(audioFileName)

        if resultsDict is not None:
            openness_audio = resultsDict['openness']
            print("Openness value (audio): {}".format(openness_audio))

            extraversion_audio = resultsDict['extraversion']
            print("Extraversion value (audio): {}".format(extraversion_audio))

            conscientiousness_audio = resultsDict['conscientiousness']
            print("Conscientiousness value (audio): {}".format(conscientiousness_audio))

            neuroticism_audio = resultsDict['neuroticism']
            print("Neuroticism value (audio): {}".format(neuroticism_audio))

            agreeableness_audio = resultsDict['agreeableness']
            print("Agreeableness value (audio): {}".format(agreeableness_audio))

            return openness_audio, extraversion_audio, conscientiousness_audio, neuroticism_audio, agreeableness_audio

    def extractAudio(self):
        filePath = 'media/'

        fileName = pathlib.PurePosixPath(self.videoName).stem
        audioFileName = filePath + fileName + '.wav'

        print(audioFileName)

        command = "ffmpeg -i C:/Users/shaoo/PycharmProjects/FYPWebsite/{} -ab 160k -ac 2 -ar 44100 -vn {}".format(
            filePath + self.videoName, audioFileName)

        subprocess.call(command, shell=True)

        return audioFileName