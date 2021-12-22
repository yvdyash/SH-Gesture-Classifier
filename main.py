import cv2
import numpy as np
import os
import handshape_feature_extractor
import frameextractor
import scipy.spatial as sp
from numpy import genfromtxt

# import the handfeature extractor class

# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video


train_directory = os.path.join("traindata")
gestureCount = 0
# index_name_map = {}

def find_gesture1(filename):
    filename=filename.lower()
    filename = filename[5:]
    if filename.startswith("0"):
        return 0
    if filename.startswith("1"):
        return 1
    if filename.startswith("2"):
        return 2
    if filename.startswith("3"):
        return 3
    if filename.startswith("4"):
        return 4
    if filename.startswith("5"):
        return 5
    if filename.startswith("6"):
        return 6
    if filename.startswith("7"):
        return 7
    if filename.startswith("8"):
        return 8
    if filename.startswith("9"):
        return 9
    if filename.startswith("decrease"):
        return 10
    if filename.startswith("fan") and "on" in filename:
        return 11
    if filename.startswith("fan") and "off" in filename:
        return 12
    if filename.startswith("increase"):
        return 13
    if filename.startswith("lightoff"):
        return 14
    if filename.startswith("lighton"):
        return 15
    if filename.startswith("setthermo"):
        return 16


def find_gesture(filename):
    filename = filename.lower()
    if "fan" in filename and "on" in filename:
        return 11
    if "fan" in filename and "off" in filename:
        return 12
    if "increase" in filename and "fan" in filename:
        return 13
    if "decrease" in filename and "fan" in filename:
        return 10
    if "light" in filename and "off" in filename:
        return 14
    if "fan" in filename and "on" in filename:
        return 15
    if "set" in filename and "thermo" in filename:
        return 16
    if "0" in filename:
        return 0
    if "1" in filename:
        return 1
    if "2" in filename:
        return 2
    if "3" in filename:
        return 3
    if "4" in filename:
        return 4
    if "5" in filename:
        return 5
    if "6" in filename:
        return 6
    if "7" in filename:
        return 7
    if "8" in filename:
        return 8
    if "9" in filename:
        return 9
    else:
        return 17


def train_and_get_penultimateLayer(train_directory, instance, store_feature):
    for filename in os.listdir(train_directory):
        if filename.endswith(".mp4"):
            gesture_ind = find_gesture(filename)
            full_filename = os.path.join(train_directory, filename)
            # print(full_filename)
            frameextractor.frameExtractor(
                full_filename,
                os.getcwd() + "/extractedFrames", gesture_ind)
            # index_name_map[gestureCount] = filename[:-25]
            # print(filename[:-25])
        else:
            continue

    for filename in os.listdir(os.getcwd() + "/extractedFrames"):
        if filename.endswith(".png"):
            full_filename = os.path.join(os.getcwd() + "/extractedFrames", filename)
            img = cv2.imread(full_filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ind = int(filename[:-4])
            if ind!=17:
                store_feature[ind - 1] = np.squeeze(instance.extract_feature(img))

    np.savetxt("training.csv", store_feature, delimiter=',')

    print(len(store_feature))


instance = handshape_feature_extractor.HandShapeFeatureExtractor.get_instance()

store_feature = genfromtxt("training.csv", delimiter=',')

# commenting the below penultimate layer finding as it was found already locally and result csv is uploaded.
train_and_get_penultimateLayer(train_directory, instance, store_feature)


# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video

print("starting test")



test_directory = os.path.join('test')
gestureCount = 0
for filename in os.listdir(test_directory):
    if filename.endswith(".mp4"):
        full_filename = os.path.join(test_directory, filename)
        # print(full_filename)
        name = frameextractor.frameExtractor(
            full_filename,
            os.getcwd()+"/extractedFramesTest", gestureCount)
        # os.rename(name, os.getcwd()+"/extractedFramesTest/"+filename[:-4]+".png")
        gestureCount+=1
    else:
        continue


# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================

training_csv = "training.csv"
store_feature_new = genfromtxt(training_csv, delimiter=',')

res = []

for filename in os.listdir(os.getcwd() + "/extractedFramesTest"):
    if filename.endswith(".png"):
        full_filename = os.path.join(os.getcwd() + "/extractedFramesTest", filename)
        img = cv2.imread(full_filename, 0)
        test_vector = instance.extract_feature(img)
        lst = []
        for each in store_feature_new:
            lst.append(sp.distance.cosine(np.squeeze(test_vector), each))
        gesture_num = lst.index(min(lst))
        res.append(gesture_num)
        # print("org: "+filename + " -- pred: "+index_name_map[gesture_num])

print(res)

np.savetxt("Results.csv", res, delimiter=',', fmt="%d")