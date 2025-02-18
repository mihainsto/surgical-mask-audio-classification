# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# !pip install librosa
# !apt-get install libsndfile1 -y

# + _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19"
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#import IPython.display as ipd # to display audio inside jupyter
import librosa # Audio parsing
import librosa.display
import matplotlib.pyplot as plt # to make graphs
#import sklearn # Ml
from tqdm import tqdm_notebook as tqdm # progress bar
import multiprocessing # going faster
from multiprocessing import Pool
import time
import random
from pathlib import Path

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# + _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a" _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0"
def f_path(fileName, tpe = None):
    """
    Adds full path to a filename recived
    tpe = None / validation / train / test
    """
    if tpe == None:
        return "ml-fmi-23-2020/" + fileName
    else:
        return "ml-fmi-23-2020/" + tpe + "/" + tpe + "/" + fileName
    
def split_into_files(file):
    """
    Splits the recived input into files
    """
    file = file.split("\n")
    file = [x.split(",") for x in file]
    return file
# Reading the file names
with open(f_path("train.txt"), "r") as f:
    trainFileNames = split_into_files(f.read())
with open(f_path("validation.txt"), "r") as f:
    validationFileNames = split_into_files(f.read())
with open(f_path("test.txt"), "r") as f:
    testFileNames = split_into_files(f.read())
# -

#Lisen to a file
fileIndex = 100
curentFilePath = f_path(trainFileNames[fileIndex][0], "train")
print(trainFileNames[fileIndex][1])
#ipd.Audio(curentFilePath)


def display_waveform(path):
    y, sr = librosa.load(curentFilePath, duration=10)
    plt.figure()
    plt.subplot(3, 1, 1)
    librosa.display.waveplot(y, sr=sr)
def display_colorbar(path):
    wave , sr = librosa.load(path)
    st = librosa.stft(wave)
    srD = librosa.amplitude_to_db(abs(st))
    plt.figure(figsize=(20, 5))
    librosa.display.specshow(srD, sr=sr, x_axis='time', y_axis='hz') 
    plt.colorbar()


# +
def spectral_centroids(path):
    # spectral centroids
    # weighter mean of the frequencies
    wave , sr = librosa.load(path)
    spectral_centroids = librosa.feature.spectral_centroid(wave, sr=sr)[0]
    return np.array(spectral_centroids)

def mfcc(path):
    wave, sr = librosa.load(path)
    mfccs = librosa.feature.mfcc(wave, sr=sr)
    return mfccs


# -

def load_file(path):
    y, sr = librosa.load(path)
    return (path, y, sr)
def load_file_list(path_list):
    p = Pool(multiprocessing.cpu_count())
    with p:
        files = p.map(load_file, path_list)
    
    p.close()
    p.join()
    return files


# +
startTime = time.time()

trainFileNames = trainFileNames[:-1]
trainOnlyFilesNames = [f_path(x[0], "train") for x in trainFileNames]
trainLabels = [int(x[1]) for x in trainFileNames]

validationFileNames = validationFileNames[:-1]
validationOnlyFilesNames = [f_path(x[0], "validation") for x in validationFileNames]
validationLabels = [int(x[1]) for x in validationFileNames]

testFileNames = testFileNames[:-1]
testOnlyFilesNames = [f_path(x[0], "test") for x in testFileNames]

#trainFiles = load_file_list(trainOnlyFilesNames)
validationFiles = load_file_list(validationOnlyFilesNames)
#testFiles = load_file_list(testOnlyFilesNames)

print("Elapsed " + str(time.time() - startTime))
# -

print(len(validationFiles))

# +
startTime = time.time()
def random_augmentation_factor():
    random1 = random.randint(7,10) / 10
    random2 = random.randint(10, 13) / 10
    random3 = random.randint(1,2)
    
    if random3 == 1:
        return random2
    else:
        return random1
def manipulate_speed(data, speed_factor):
    return librosa.effects.time_stretch(data, speed_factor)
def pitch_augment(data, sample_rate):
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change =  pitch_pm * 2*(np.random.uniform())  
    data = librosa.effects.pitch_shift(data.astype('float64'), 
                                      sample_rate, n_steps=pitch_change, 
                                      bins_per_octave=bins_per_octave)
    return data

def time_shift_augment(data):
    timeshift_fac = 0.2 *2*(np.random.uniform()-0.5)  # up to 20% of length
    start = int(data.shape[0] * timeshift_fac)
    if (start > 0):
        data = np.pad(data,(start,0),mode='constant')[0:data.shape[0]]
    else:
        data = np.pad(data,(0,-start),mode='constant')[0:data.shape[0]]
    return data


augmentedValidationFiles = []

def file_name_for_augment(path):
    return path.split('.')[0] + 'a.' + path.split('.')[1]
def file_name_double_for_augment(path):
    return path.split('.')[0] + 'aa.' + path.split('.')[1]
# speeding the files
for file in validationFiles:
    augmentedValidationFiles.append(file)
    augmentedValidationFiles.append((file_name_for_augment(file[0]), manipulate_speed(file[1], random_augmentation_factor()), file[2]))

validationFiles = augmentedValidationFiles
augmentedValidationFiles = []

# time shifting
for file in validationFiles:
    augmentedValidationFiles.append((file_name_double_for_augment(file[0]), time_shift_augment(file[1]), file[2]))
    
print(len(validationFiles))
for file in augmentedValidationFiles:
    validationFiles.append(file)
print(len(validationFiles))
print("Elapsed " + str(time.time() - startTime))
# -



# +
def create_fold_spectogram(file, saveDirectory):
    path, y, sr = file
    savePathImage = saveDirectory + '/images/' + path.split('/')[-1].replace('.wav', '.png')
    savePathNpy = saveDirectory + '/text/' + path.split('/')[-1].replace('.wav', '.npy')
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.set_frame_on(False)
    feature = librosa.feature.melspectrogram(y, sr = sr)
    librosa.display.specshow(librosa.power_to_db(feature, ref=np.max))
    np.save(savePathNpy, feature)
    plt.savefig(savePathImage, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()
    return True

def create_mfcc_spectogram(file, saveDirectory):
    path, y, sr = file
    savePathImage = saveDirectory + '/images/' + path.split('/')[-1].replace('.wav', '.png')
    savePathNpy = saveDirectory + '/text/' + path.split('/')[-1].replace('.wav', '.npy')    
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.set_frame_on(False)
    feature = librosa.feature.mfcc(y, sr = sr)
    librosa.display.specshow(librosa.power_to_db(feature, ref=np.max))
    np.save(savePathNpy, feature)
    plt.savefig(savePathImage, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()
    return True
    
def create_crf_spectogram(file, saveDirectory):
    path, y, sr = file
    savePathImage = saveDirectory + '/images/' + path.split('/')[-1].replace('.wav', '.png')
    savePathNpy = saveDirectory + '/text/' + path.split('/')[-1].replace('.wav', '.npy')    
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.set_frame_on(False)
    feature = librosa.feature.chroma_stft(y, sr = sr)
    librosa.display.specshow(librosa.power_to_db(feature, ref=np.max))
    np.save(savePathNpy, feature)
    plt.savefig(savePathImage, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()
    return True


# +
def create_folders():
    def create_subfolders_for(folder_name):
        Path(folder_name).mkdir(parents=True, exist_ok=True)
        Path(folder_name + "/images").mkdir(parents=True, exist_ok=True)
        Path(folder_name + "/text").mkdir(parents=True, exist_ok=True)
    def create_folders_for(name):
        Path(name + "/fold_spectogram").mkdir(parents=True, exist_ok=True)
        create_subfolders_for(name + "/fold_spectogram")
        Path(name + "/mfcc_spectogram").mkdir(parents=True, exist_ok=True)
        create_subfolders_for(name + "/mfcc_spectogram")
        Path(name + "/crf_spectogram").mkdir(parents=True, exist_ok=True)
        create_subfolders_for(name + "/crf_spectogram")
    
    
    Path("kfold").mkdir(parents=True, exist_ok=True)
    Path("kfold/validation").mkdir(parents=True, exist_ok=True)
    create_folders_for("kfold/validation")#


def process_and_save_train(file):

    #create_fold_spectogram(file, "train/fold_spectogram")
    #create_mfcc_spectogram(file, "train/mfcc_spectogram")
    #create_crf_spectogram(file, "train/crf_spectogram")
    
    return True

def process_and_save_validation(file):

    create_fold_spectogram(file, "kfold/validation/fold_spectogram")
    #create_mfcc_spectogram(file, "validation/mfcc_spectogram")
    #create_crf_spectogram(file, "validation/crf_spectogram")
    
    return True
    
def process_and_save_test(file):

    #create_fold_spectogram(file, "test/fold_spectogram")
    #create_mfcc_spectogram(file, "test/mfcc_spectogram")
    #create_crf_spectogram(file, "test/crf_spectogram")
    
    return True



# -

create_folders()

# # +
# #processing train data

# startTime = time.time()
# p = Pool(multiprocessing.cpu_count())
# with p:
#     p.map(process_and_save_train, trainFiles)
# p.close()
# p.join()

# print("Processed train data in " + str(time.time() - startTime))
# startTime = time.time()

#processing validation data

startTime = time.time()
p = Pool(multiprocessing.cpu_count())
with p:
    p.map(process_and_save_validation, validationFiles)
p.close()
p.join()

print("Processed validation data in " + str(time.time() - startTime))
startTime = time.time()

#processing test data

# startTime = time.time()
# p = Pool(multiprocessing.cpu_count())
# with p:
#     p.map(process_and_save_test, testFiles)
# p.close()
# p.join()

# print("Processed test data in " + str(time.time() - startTime))
# startTime = time.time()
# # -






