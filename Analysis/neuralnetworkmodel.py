from collections import defaultdict

import pickle

import numpy as np
import time
import xlsxwriter
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from statistics import mean

from Analysis.VideoProcessing.featureextraction import FeatureExtraction
from Analysis.VideoProcessing.frameextraction import FrameExtraction


class NeuralNetworkModel:
    def __init__(self, modelDict=None):
        self.modelDict = modelDict

    def predict_single_video(self, videoName, model=None):
        if model is not None:
            self.modelDict = model

        print("Trying to predict a single frame value")

        neuroticismListFace = []
        neuroticismListLeftEye = []
        neuroticismListRightEye = []
        agreeablenessListFace = []
        agreeablenessListLeftEye = []
        neuroticismListSmile = []
        agreeablenessListSmile = []
        agreeablenessListRightEye = []
        conscientiousnessListFace = []
        conscientiousnessListLeftEye = []
        conscientiousnessListRightEye = []
        conscientiousnessListSmile = []
        extraversionListFace = []
        extraversionListLeftEye = []
        extraversionListRightEye = []
        extraversionListSmile = []
        opennessListFace = []
        opennessListLeftEye = []
        opennessListRightEye = []
        opennessListSmile = []

        extractor = FrameExtraction()
        features = FeatureExtraction(24, 8)

        croppedFaces, smileFrames, leftEyeFrames, rightEyeFrames = extractor.extract_single_video(videoName)
        featuresDict = features.extract_single_video(croppedFaces, smileFrames, leftEyeFrames, rightEyeFrames)

        if featuresDict['face']:
            opennessListFace = self.modelDict['face']['openness'].predict(featuresDict['face'])
            extraversionListFace = self.modelDict['face']['extraversion'].predict(featuresDict['face'])
            neuroticismListFace = self.modelDict['face']['neuroticism'].predict(featuresDict['face'])
            agreeablenessListFace = self.modelDict['face']['agreeableness'].predict(featuresDict['face'])
            conscientiousnessListFace = self.modelDict['face']['conscientiousness'].predict(featuresDict['face'])

        if featuresDict['righteye']:
            opennessListRightEye = self.modelDict['righteye']['openness'].predict(featuresDict['righteye'])
            extraversionListRightEye = self.modelDict['righteye']['extraversion'].predict(featuresDict['righteye'])
            neuroticismListRightEye = self.modelDict['righteye']['neuroticism'].predict(featuresDict['righteye'])
            agreeablenessListRightEye = self.modelDict['righteye']['agreeableness'].predict(
                featuresDict['righteye'])
            conscientiousnessListRightEye = self.modelDict['righteye']['conscientiousness'].predict(
                featuresDict['righteye'])

        if featuresDict['lefteye']:
            opennessListLeftEye = self.modelDict['lefteye']['openness'].predict(featuresDict['lefteye'])
            extraversionListLeftEye = self.modelDict['lefteye']['extraversion'].predict(featuresDict['lefteye'])
            neuroticismListLeftEye = self.modelDict['lefteye']['neuroticism'].predict(featuresDict['lefteye'])
            agreeablenessListLeftEye = self.modelDict['lefteye']['agreeableness'].predict(featuresDict['lefteye'])
            conscientiousnessListLeftEye = self.modelDict['lefteye']['conscientiousness'].predict(
                featuresDict['lefteye'])

        if featuresDict['smile']:
            opennessListSmile = self.modelDict['smile']['openness'].predict(featuresDict['smile'])
            extraversionListSmile = self.modelDict['smile']['extraversion'].predict(featuresDict['smile'])
            neuroticismListSmile = self.modelDict['smile']['neuroticism'].predict(featuresDict['smile'])
            agreeablenessListSmile = self.modelDict['smile']['agreeableness'].predict(featuresDict['smile'])
            conscientiousnessListSmile = self.modelDict['smile']['conscientiousness'].predict(featuresDict['smile'])

        openness = defaultdict(float)
        extraversion = defaultdict(float)
        neuroticism = defaultdict(float)
        conscientiousness = defaultdict(float)
        agreeableness = defaultdict(float)

        openness["average"] = -1
        extraversion["average"] = -1
        neuroticism["average"] = -1
        conscientiousness["average"] = -1
        agreeableness["average"] = -1

        opennessList = None
        agreeablenessList = None
        conscientiousnessList = None
        extraversionList = None
        neuroticismList = None

        if opennessListFace is not None or opennessListRightEye is not None or opennessListLeftEye is not None or opennessListSmile is not None:
            opennessList = np.array(
                (mean(opennessListFace), mean(opennessListLeftEye), mean(opennessListRightEye), mean(opennessListSmile)))
            openness["average"] = np.average(opennessList)
            openness["min"] = np.min(opennessList)
            openness["max"] = np.max(opennessList)

        if extraversionListFace is not None or extraversionListLeftEye is not None or extraversionListRightEye is not None or extraversionListSmile is not None:
            extraversionList = np.array(
                (mean(extraversionListFace), mean(extraversionListLeftEye), mean(extraversionListRightEye), mean(extraversionListSmile)))
            extraversion["average"] = np.average(extraversionList)
            extraversion["min"] = np.min(extraversionList)
            extraversion["max"] = np.max(extraversionList)

        if neuroticismListFace is not None or neuroticismListLeftEye is not None or neuroticismListRightEye is not None or neuroticismListSmile is not None:
            neuroticismList = np.array(
                (mean(neuroticismListFace), mean(neuroticismListLeftEye), mean(neuroticismListRightEye), mean(neuroticismListSmile)))
            neuroticism["average"] = np.average(neuroticismList)
            neuroticism["min"] = np.min(neuroticismList)
            neuroticism["max"] = np.max(neuroticismList)

        if agreeablenessListFace is not None or agreeablenessListLeftEye is not None or agreeablenessListRightEye is not None or agreeablenessListLeftEye is not None or agreeablenessListSmile is not None:
            agreeablenessList = np.array(
                (mean(agreeablenessListFace), mean(agreeablenessListLeftEye), mean(agreeablenessListRightEye), mean(agreeablenessListSmile)))
            agreeableness["average"] = np.average(agreeablenessList)
            agreeableness["min"] = np.min(agreeablenessList)
            agreeableness["max"] = np.max(agreeablenessList)

        if conscientiousnessListFace is not None or conscientiousnessListLeftEye is not None or conscientiousnessListRightEye is not None or conscientiousnessListSmile is not None:
            conscientiousnessList = np.array((mean(conscientiousnessListFace), mean(conscientiousnessListLeftEye),
                                                    mean(conscientiousnessListRightEye), mean(conscientiousnessListSmile)))
            conscientiousness["average"] = np.average(conscientiousnessList)
            conscientiousness["min"] = np.min(conscientiousnessList)
            conscientiousness["max"] = np.max(conscientiousnessList)

        return openness, extraversion, neuroticism, agreeableness, conscientiousness