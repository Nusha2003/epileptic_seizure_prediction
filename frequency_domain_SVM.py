#function that reads through the files
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


#isolating theta and delta bands
def extract_features(data, fs): #calculates power spectral density for each trial
    
    freqs, welch_power= welch(data, fs, nperseg=256)
    # welch_power = welch(trial, fs, nperseg=256)[1]  # Frequency array - frequency values at which the PSD is estimated
    #PSD: 2D array represeting the power spectral density estimates
    band_powers = extract_frequency_bands(freqs, welch_power, bands)
    return np.array(band_powers)

def extract_frequency_bands(freqs, psd, bands): #
    band_powers = []
    for band_name, (low, high) in bands.items():
        band_indices = np.where((freqs >=low)& (freqs<high))[0]
        band_power = np.mean(psd[... , band_indices], axis = -1)
        band_powers.append(band_power)
    return np.array(band_powers)

bands = {'delta': (0.5, 4), 'theta': (4, 8)}

#extracting features for ictal and interictal data
fs = 300
ictal_features = extract_features(ictal_data, fs)
ictal_features = np.transpose(ictal_features)
interictal_features = extract_features(interictal_data, fs)
interictal_features = np.transpose(interictal_features)

#reducing the number of rows to make the computations easier:
#now, we have the delta and theta band powers for ictal and interictal data
#this will be the train data

#creating labels for the data:
ictal_labels = np.ones(ictal_features.shape[0])
interictal_labels = np.zeros(interictal_features.shape[0])

#combining the data and labels
X = np.concatenate((ictal_features, interictal_features), axis=0)
y = np.concatenate((ictal_labels, interictal_labels), axis=0)

random_indices = np.random.permutation(len(X))
X = X[random_indices]
y = y[random_indices]

X = np.delete(X,slice(10000,69087), axis = 0)
y = np.delete(y, slice(10000,69087), axis = 0)
#split into test, train, predict - 80% for train, 10% for validate, 10% for test
X_train, X_val, X_test = np.split(X, [int(.8*len(X)), int(.9*len(X))])
y_train, y_val, y_test = np.split(y, [int(.8*len(y)), int(.9*len(y))])

#initializing PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_val = pca.transform(X_val)
X_test = pca.transform(X_test)

#training the model:
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svm_model = SVC() #creates SVM object using SVC class
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} #parameters to be tested
grid_search = GridSearchCV(svm_model, param_grid, cv = 5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
#train SVM model with best hyperparameters
svm_model = SVC(C = best_params['C'], gamma = best_params['gamma'], kernel = best_params['kernel'])
svm_model.fit(X_train, y_train)

#make predictions on validation set
y_pred = svm_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation accuracy: {accuracy}")

#make predictions on test set
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy}")


#evaluating performance:
import matplotlib.pyplot as plt
plt.figure(figsize = (8,6))
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='viridis', alpha=0.5)
plt.colorbar(label='Class')
plt.title('PCA Components Scatter Plot')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)








