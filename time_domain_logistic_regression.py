import numpy as np
from scipy import signal
from scipy.signal import resample_poly #resample signals
import os
import mne
from pymatreader import read_mat
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from scipy.signal import welch
from sklearn.decomposition import PCA

import antropy as ant

def read_files(root):
    raw = []
    path = root
    file_list = os.listdir(path)
    for file in file_list:
        full_path = os.path.join(path,file)
        try:
            raw.append(read_mat(full_path))
        except Exception as e:
            print(f"Error reading file '{file}': {e}")
            continue
    return raw

#loaded data
ictal_file = np.array(read_files('/Users/anusha/QBIO499_Project/Ictal'))
interictal_file = np.array(read_files('/Users/anusha/QBIO499_Project/Interictal'))


#isolating the data:
def isolate_data(data):
    eeg_data = [d['data']for d in data]
    filtered_data = []
    for arr in eeg_data:
        if arr.shape[1] == 5000:
            filtered_data.append(arr)
    return filtered_data

ictal_data = isolate_data(ictal_file) #have timexchannel data for 922 subjects - at each index, is another array
ictal_data = np.concatenate(list(ictal_data), axis=0) #concatenating all the arrays into one big array
ictal_data = np.delete(ictal_data, [10000, 37088], 0)
interictal_data = isolate_data(interictal_file)
interictal_data = np.concatenate(list(interictal_data), axis=0)

#so now I have ictal and interictal data for channels X time for 922 subjects
#define bandpass filter
def bandpass_filter(data, low_freq, high_freq, fs, order = 4):
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def notch_filter(data, sr, notch_freq=60, Q = 30):
    nyquist = 0.5 * sr
    freq = notch_freq / nyquist
    b,a = butter(2,[freq-1/Q, freq+1/Q], btype='bandstop')
    filtered_data = filtfilt(b,a,data)
    return filtered_data

ictal_data = bandpass_filter(ictal_data, 0.5, 30, 300)
interictal_data = bandpass_filter(interictal_data, 0.5, 30, 300)
ictal_data = notch_filter(ictal_data, 300)
interictal_data = notch_filter(interictal_data, 300)


def z_score_normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data

ictal_data = z_score_normalize(ictal_data)
interictal_data = z_score_normalize(interictal_data)

#applying hjorth parameters
ictal_data = ant.hjorth_params(ictal_data, axis = 0)
ictal_features = np.array(ictal_data)
ictal_features = np.transpose(ictal_features)

interictal_data = ant.hjorth_params(interictal_data, axis = 0)
interictal_data = np.array(interictal_data)
interictal_features = np.transpose(interictal_data)
print(ictal_features.shape)

ictal_labels = np.ones(ictal_features.shape[0])
interictal_labels = np.zeros(interictal_features.shape[0])

X = np.concatenate((ictal_features, interictal_features), axis=0)
y = np.concatenate((ictal_labels, interictal_labels), axis=0)

random_indices = np.random.permutation(len(X))
X = X[random_indices]
y = y[random_indices]

X_train, X_test = np.split(X, [int(.8*len(X))])
y_train, y_test = np.split(y, [int(.8*len(y))])

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train,y_train)
y_prob = log_reg_model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr,tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()





