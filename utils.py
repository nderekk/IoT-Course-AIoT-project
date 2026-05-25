from os import listdir
from os.path import isfile, join
import os

# basic data engineering
import pandas as pd
import numpy as np
import scipy

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# db
import pymongo
from bson import ObjectId

# configs & other
import yaml
from tqdm.notebook import tqdm_notebook
from datetime import datetime
from time import time

from psynlig import pca_explained_variance_bar

# utils processing
from utils import sliding_window_pd
from utils import apply_filter
from utils import filter_instances
from utils import flatten_instances_df
from utils import df_rebase
from utils import rename_df_column_values
from utils import subtract_mean

# utils visualization
from utils_visual import plot_instance_time_domain
from utils_visual import plot_instance_3d
from utils_visual import plot_np_instance
from utils_visual import plot_heatmap
from utils_visual import plot_scatter_pca
from utils_visual import plot_bar
from utils_visual import plot_distribution_analysis



def sliding_window_pd(
        df,
        ws=500,
        overlap=250,
        w_type="hann",
        w_center=True,
        print_stats=False
) -> list:
    """Applies the sliding window algorithm to the DataFrame rows.

    Args:
        df: The DataFrame with all the values that will be inserted to the
            sliding window algorithm.
        ws: The window size in number of samples.
        overlap: The hop length in number of samples.
        w_type: The windowing function.
        w_center: If False, set the window labels as the right edge of the
            window index. If True, set the window labels as the center of the
            window index.
        print_stats: Print statistical inferences from the process. Defaults
            to False.

    Returns:
        A list of DataFrames each one corresponding to a produced window.
    """
    counter = 0
    windows_list = list()
    # min_periods: Minimum number of observations in window required to have
    # a value;
    # For a window that is specified by an integer, min_periods will default
    # to the size of the window.
    for window in df.rolling(window=ws, step=overlap, min_periods=ws,
                             win_type=w_type, center=w_center):
        if (window[window.columns[0]].count() >= ws) and (not window.isnull().values.any()):           
            if print_stats:
                print("Print Window:", counter)
                print("Number of samples:", window[window.columns[0]].count())
            if 'user' in window.columns and window['user'].nunique() != 1:
                continue
            windows_list.append(window)
        counter += 1
    if print_stats:
        print("List number of window instances:", len(windows_list))

    return windows_list


def apply_filter(
        arr,
        order=5,
        wn=0.1,
        filter_type="lowpass"
) -> np.ndarray:
    """Applies filter to the multi-axis signal.

    Args:
        arr: The initial NumPy signal array values.
        order: The order of the filter.
        wn: The critical frequency or frequencies.
        filter_type: The type of filter. {‘lowpass’, ‘highpass’, ‘bandpass’,
            ‘bandstop’}

    Returns:
        NumPy Array with the filtered signal.
    """
    fbd_filter = scipy.signal.butter(N=order, Wn=wn, btype=filter_type,
                                     output="sos")
    filtered_signal = scipy.signal.sosfiltfilt(sos=fbd_filter, x=arr, padlen=0)

    return filtered_signal


def filter_instances(
        instances_list,
        order,
        wn,
        filter_type
) -> list:
    """Applies filter to a list of windows (each window is a DataFrame).

    Args:
        instances_list: List of DataFrames.
        order: The order of the filter.
        wn: The critical frequency or frequencies.
        filter_type: The type of filter. {‘lowpass’, ‘highpass’, ‘bandpass’,
            ‘bandstop’}

    Returns:

    """
    filtered_instances_list = list()
    for item in instances_list:
        filtered_instance = item.apply(apply_filter,
                                       args=(order, wn, filter_type)
                                       )
        filtered_instances_list.append(filtered_instance)
    print("Number of filtered instances in the list:",
          len(filtered_instances_list)
          )

    return filtered_instances_list


def flatten_instances_df(instances_list: list) -> pd.DataFrame:
    """Flattens each instance and create a DataFrame with the whole flattened
        instances.

    Args:
        instances_list: The list of DataFrames to be flattened

    Returns:
        A DataFrame that includes the whole flattened DataFrames
    """
    flattened_instances_list = list()
    for item in instances_list:
        instance = item.to_numpy().flatten()
        flattened_instances_list.append(instance)
    df = pd.DataFrame(flattened_instances_list)

    return df


def df_rebase(
        df: pd.DataFrame,
        target_list: list,
        ref_list: list
) -> pd.DataFrame:
    """Changes the order and name of DataFrame columns to the project's needs
        for readability.

    Args:
        df: The pandas DataFrame.
        order_list: List object that contains the proper order of the default
             column names.
        ref_list: List object that contains the renaming list based
            on the project needs.

    Returns:
        A DataFrame with the new columns order and names.
    """
    print("Initial columns:", list(df.columns))

    if are_lists_equal(list(df.columns), ref_list):
        pass

    else:
        if len(target_list) == len(ref_list): 
            # keep and re-order only the necessary columns of the initial DataFrame
            df = df[target_list]
            rename_dict = dict(zip(target_list, ref_list))
            df = df.rename(columns=rename_dict)  # rename the columns
        else:
            print("The length of the target list and the reference list is not equal.")

    print("Processed columns:", list(df.columns))

    return df


def rename_df_column_values(
    np_array: np.ndarray, 
    y: list, 
    columns_names: tuple = ("acc_x", "acc_y", "acc_z")
):
    """Creates a DataFrame with a "y" label column and replaces the values of the y with the index
    of the unique values of y.

    Args:
        np_array: 2D NumPy array.
        y: List with the y labels
        columns_names: List with the DF columns names.

    Returns:
        DataFrame with the multi-axes values and the target labels column.
    """
    arr_y = np.array(y)  # list to numpy array
    unique_values_list = np.unique(arr_y)  # unique list of values

    df = pd.DataFrame(np_array, columns=columns_names)
    df["y"] = y

    # replace the row item value in the y column of the df, with its index in the unique list
    for idx, x in enumerate(unique_values_list):
        df["y"] = np.where(df["y"] == x, idx, df["y"])

    return df


def are_lists_equal(
    list1: list, 
    list2: list
) -> bool:
    return set(list1) == set(list2)


def encode_labels(instances_list) -> np.ndarray:
    """Encodes target labels.

    Args:
        instances_list: List of instances to be encoded.

    Returns:
        The encoded array.
    """
    le = preprocessing.LabelEncoder()
    le.fit(instances_list)
    instances_arr = le.transform(instances_list)

    return instances_arr


def list_files_in_folder(folder_path) -> list:
    """Returns a list of all CSV files within the specified folder.

    Args:
        folder_path (str): The directory path to search for files.

    Returns:
        list: A list containing the filenames (strings) of all files
              in the directory that end with the '.csv' extension.
    """
    files_list = list()
    for f in listdir(folder_path):
        if isfile(join(folder_path, f)):
            if f.endswith(".csv"):
                files_list.append(f)

    return files_list

def subtract_mean(df, columns):
    """
    Subtracts the mean from specified columns in a
    DataFrame.
    """
    df_centered = df.copy()
    for col in columns:
        df_centered[col] = df_centered[col] - df_centered[col].mean()
    return df_centered

def chop_edges_and_concat(results: list) -> dict:
    """This fuction takes the list of results from the database query, chops off the first and last 500 rows of data for each gesture instance, and concatenates the remaining data for each gesture_id into a single DataFrame. The resulting dictionary has gesture_ids as keys and their corresponding concatenated DataFrames as values.

    Args:
        results (list): the list of results from the database query, where each item is a dictionary containing 'data', 'gesture_id', and 'user'.

    Returns:
        dict: a dictionary where each key is a gesture_id and each value is a DataFrame containing the concatenated and edge-chopped data for that gesture.
    """
    all_chunks = pd.DataFrame(results)
    # print(len(all_chunks))

    ids = all_chunks['gesture_id'].unique().tolist()
    # print(ids)

    gestures = {}
    for id in ids:
        gesture_chunks = all_chunks[all_chunks['gesture_id'] == id]
        print(gesture_chunks.columns)
        current_data = gesture_chunks[['user', 'data']]
        
        chunks = []
        for chunk in current_data.itertuples():
            # print(type(chunk))
            # print(chunk)
            df = pd.DataFrame(chunk.data)
            df.drop(df.tail(500).index, inplace=True)
            df.drop(df.head(500).index, inplace=True)
            df['user'] = chunk.user
            chunks.append((id, df))

        joined_gesture = pd.concat([df for _, df in chunks], ignore_index=True)
        gestures[id] = joined_gesture

        # print(len(chunks[0][1]))
        # print(chunks[0][1].head())

        # print(len(joined_gesture))
        # print(joined_gesture.head())
    return gestures

def clean_sensor_outliers(df, window_size=50, threshold=3.0, axes = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']):
    """
    Cleans extreme hardware spikes using a Rolling Z-Score and Linear Interpolation.
    Assumes 100Hz data (window_size=50 is 0.5 seconds).
    """
    # copy of the og so we dont overwrite raw data
    df_clean = df.copy()
    
    sensor_columns = axes
    
    for col in sensor_columns:
        # calucalate rolling statistics, rolling just means the mean and std are calculated based on a moving window of data instead of the whole data, this is useful for time series data bvecause gestures are really short and sudden and most of the time there is no movement so essentially our outliers are the info itself. Standard z-score would flag the whole gesture as an outlier, but rolling z-score can catch the sudden spikes within the gesture without flagging the whole thing.
        rolling_mean = df_clean[col].rolling(window=window_size, center=True).mean()
        rolling_std = df_clean[col].rolling(window=window_size, center=True).std()
        
        rolling_z_scores = np.abs((df_clean[col] - rolling_mean) / rolling_std)
        
        # find how many outliers there are
        outlier_count = (rolling_z_scores > threshold).sum()
        # print(f"Cleaned {outlier_count} glitches in {col}")
        
        # replace them with nan so they can be fixed by interpolation
        df_clean.loc[rolling_z_scores > threshold, col] = np.nan
        
        # limit_direction='both' ensures it fixes glitches even if they happen on the very first or last row
        df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both')
        
    return df_clean


def purge_outliers_all_gestures(gestures, axes=['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z'], window_size=50, threshold=3.0):
    cleaned_gestures = {}
    for gesture_id, gesture_df in gestures.items():
        # print(f'Cleaning gesture: {gesture_id}')
        gesture_cleaned = clean_sensor_outliers(gesture_df, window_size, threshold, axes)
        cleaned_gestures[gesture_id] = gesture_cleaned
    return cleaned_gestures

def scale_gestures(gestures_train: dict, gestures_val: dict) -> None:
    """This function scales the accelerometer and gyroscope data in the training and validation gesture datasets using StandardScaler for accelerometer data and RobustScaler for gyroscope data. The scalers are fit on the training data and then applied to both the training and validation datasets to ensure that the scaling is consistent.  

    Args:
        gestures_train (dict): A dictionary containing the training gesture DataFrames.
        gestures_val (dict): A dictionary containing the validation gesture DataFrames.
        
    Returns:
        None: This function modifies the input dictionaries in place, scaling the relevant columns of the DataFrames contained within the dictionaries. It does not return any value.
    """
    std_scaler = StandardScaler()
    robust_scaler = RobustScaler()

    all_train_data = pd.concat(gestures_train.values())
    std_scaler.fit(all_train_data[['acc_x', 'acc_y', 'acc_z']])
    robust_scaler.fit(all_train_data[['gyr_x', 'gyr_y', 'gyr_z']])

    for id in gestures_train:
        gestures_train[id][['acc_x', 'acc_y','acc_z']] = std_scaler.transform(gestures_train[id][['acc_x', 'acc_y', 'acc_z']])
        gestures_train[id][['gyr_x', 'gyr_y', 'gyr_z']] = robust_scaler.transform(gestures_train[id][['gyr_x', 'gyr_y', 'gyr_z']])
        
    for id in gestures_val:
        gestures_val[id][['acc_x', 'acc_y', 'acc_z']] = std_scaler.transform(gestures_val[id][['acc_x', 'acc_y', 'acc_z']])
        gestures_val[id][['gyr_x', 'gyr_y', 'gyr_z']] = robust_scaler.transform(gestures_val[id][['gyr_x', 'gyr_y', 'gyr_z']])

def window_and_filter_gestures(cleaned_gestures: dict, ws=500, overlap=100, order=5, wn=0.4, filter_type='lowpass') -> tuple:
    """The function takes in a dictionary of cleaned gesture DataFrames, applies the specified filter to each DataFrame, and then uses a sliding window approach to create clips of the data. The resulting clipped gestures are stored in a new dictionary, and a distribution of the number of clips per gesture is also created.
    
    *Note on sliding_window_pd: This function was modified so that if a certain window contains data from more that one user, it will discard that window. This is because the suddent switch in swipping/posture style from user to user would trip up the model.

    Args:
        cleaned_gestures (dict): A dictionary containing the cleaned gesture DataFrames.
        ws (int, optional): The window size. Defaults to 500.
        overlap (int, optional): The overlap between consecutive windows. Defaults to 100.
        order (int, optional): The order of the filter. Defaults to 5.
        wn (float, optional): The normalized frequency of the filter. Defaults to 0.4.
        filter_type (str, optional): The type of the filter. Defaults to 'lowpass'.

    Returns:
        tuple: A tuple containing the clipped gestures and the distribution of clips.
    """
    clips_arr = []
    clipped_gestures = {}
    for id, data in cleaned_gestures.items():
        # slice data into windows
        filtered_data = data.drop(columns='user').apply(apply_filter, args=(order, wn, filter_type))
        filtered_data['user'] = data['user'] # reattach user so windowing can discard multiple window users
        
        clips = sliding_window_pd(filtered_data, ws=ws, overlap=overlap, print_stats=False)
        total_clips = len(clips)
        # print(f'Total clips: {total_clips}')
        clips_arr.append({'Gesture': id, 'Total Clips': total_clips})
        clipped_gestures[id] = clips
        # print(f'GESTURE_ID: {id}')
        # plot_instance_time_domain(fclips[1], id)
        # plot_instance_3d(fclips[1], axes_list=['gyr_x', 'gyr_y', 'gyr_z'])
        # plot_instance_3d(fclips[1], axes_list=['acc_x', 'acc_y', 'acc_z'])
    return (clipped_gestures, clips_arr)

def flatten_gestures(gesture_clips: dict) -> dict:
    """Flattens the clipped gesture data into a single DataFrame for each gesture. Is needed for traditional ML models that expect a single DataFrame per class instead of a list of DataFrames.

    Args:
        gesture_clips (dict): The clipped gesture data in dictionary form, where each key is a gesture_id and each value is a list of DataFrames representing the clips for that gesture.

    Returns:
        dict: A dictionary containing the flattened gesture data, where each key is a gesture_id and each value is a single DataFrame representing all the clips for that gesture.
    """
    flattened_clips = {}
    for id, clips in gesture_clips.items():
        flattened_clips[id] = flatten_instances_df(clips)
    return flattened_clips

def prepare_xy(flattened_dict: dict) -> tuple:
    """ Turns the flattened gesture DataFrames into a feature matrix X and target vector y in numpy array form, ready to be fed into a machine learning model.

    Args:
        flattened_dict (dict): The flattened gesture data in dictionary form, where each key is a gesture_id and each value is a single DataFrame representing all the clips for that gesture.

    Returns:
        tuple: A tuple containing the feature matrix X and the target vector y in numpy array form, ready to be fed into a machine learning model.
    """
    X = pd.concat(flattened_dict.values(), ignore_index=True).to_numpy()
    
    y = []
    for gesture_name, df in flattened_dict.items():
        y.extend([gesture_name] * len(df))
        
    return X, np.array(y)

def prepare_xy_3d(gesture_clips_dict: dict) -> tuple:
    """
    Extracts windowed data into a 3D NumPy array suitable for
    sequence-based models such as 1D CNNs or RNNs, without flattening
    the time dimension.

    Unlike the flattened approach used for traditional classifiers, this
    function preserves the structure of each window as a 2D matrix
    (samples * axes), stacking all windows into a 3D array of shape
    (n_windows, window_size, n_axes). This is the required input format
    for nn models like our 1d cnn.

    Args:
        gesture_clips_dict (dict): Dictionary mapping gesture_ids to lists of DataFrames, where each DataFrame represents a windowed clip of sensor data for that gesture.

    Returns:
        tuple: A pair (X_3d, y) where:
            - X_3d (np.ndarray): 3D array of shape (n_windows, window_size, 6),
              containing the raw axes (acc_x, acc_y, acc_z,
              gyr_x, gyr_y, gyr_z) for each window.
            - y (np.ndarray): 1D array of shape (n_windows,) containing the
              corresponding gesture class label for each window.
    """
    X_3d = []
    y = []
    for gesture_name, clips in gesture_clips_dict.items():
        for clip in clips:
            if isinstance(clip, pd.DataFrame):
                X_3d.append(clip[['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']].values)
            else:
                X_3d.append(clip)
        y.extend([gesture_name] * len(clips))
    return np.array(X_3d), np.array(y)

def train_and_predict_cnn(X_train_3d, X_val_3d, y_train, y_val, epochs=20, batch_size=16): 
    """
    Builds, trains, and evaluates a 1D CNN classifier on windowed data using torch.

    The network consists of two Conv1D blocks — each with batch
    normalisation, ReLU activation, max-pooling, and dropout — followed by a
    fully connected head. Input axes are treated as channels (shape: Batch * 6 * Length).
    Labels are integer-encoded internally
    and decoded back to class-name strings before returning, so predictions are
    directly comparable to those of the sklearn classifiers in this pipeline.

    Args:
        X_train_3d (np.ndarray): Training windows of shape (n_windows, window_size, 6).
        X_val_3d   (np.ndarray): Validation windows of shape (n_windows, window_size, 6).
        y_train    (np.ndarray): Gesture class labels for the training set.
        y_val      (np.ndarray): Gesture class labels for the validation set (unused
            during training; used only to size the DataLoader).
        epochs     (int): Number of full passes over the training set. Default 20.
        batch_size (int): Mini-batch size for both loaders. Default 16.

    Returns:
        np.ndarray: Predicted gesture class name strings for each validation window,
            in the same order as X_val_3d.
    """
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)

    ws = X_train_3d.shape[1]
    
    # PyTorch Conv1d expects shape: (Batch, Channels, Length)
    X_train_reshaped = np.transpose(X_train_3d, (0, 2, 1))
    X_val_reshaped = np.transpose(X_val_3d, (0, 2, 1))

    # Convert to Tensors
    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train_reshaped, dtype=torch.float32), 
        torch.tensor(y_train_encoded, dtype=torch.long)
    ), batch_size=batch_size, shuffle=True)
    
    val_loader = DataLoader(TensorDataset(
        torch.tensor(X_val_reshaped, dtype=torch.float32), 
        torch.tensor(y_val_encoded, dtype=torch.long)
    ), batch_size=batch_size, shuffle=False)

    # Model Definition
    class GestureCNN(nn.Module):
        def __init__(self, num_classes, ws):
            super(GestureCNN, self).__init__()
            self.conv_block = nn.Sequential(
                nn.Conv1d(6, 16, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4), nn.Dropout(0.2),
                nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(5), nn.Dropout(0.2)
            )
            self.fc_block = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * (ws // 4 // 5), 64),
                nn.ReLU(), nn.Dropout(0.6),
                nn.Linear(64, num_classes)
            )

        def forward(self, x):
            return self.fc_block(self.conv_block(x))

    model = GestureCNN(len(le.classes_), ws)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)

    # train
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            
    # predict
    model.eval() # turn the model to eval mode to disable dropout an dother training specific features
    all_preds = []
    with torch.no_grad():
        for val_X, _ in val_loader:
            _, predicted = torch.max(model(val_X), 1)
            all_preds.extend(predicted.cpu().numpy())

    # turn integer predictions back to class name strings to match other models
    return le.inverse_transform(all_preds)

def process_fold(gestures_train: dict, gestures_val: dict, ws = 300) -> tuple:
    """
    Executes a single fold of the cross-validation loop, running both
    the flattened RF and 1D CNN pipelines on the same train/val split
    so their predictions are directly comparable.

    Applies the full preprocessing chain in sequence — scaling,
    low-pass filtering and windowing, then branches into two parallel paths:
    the RF receives flattened 1D vectors while the CNN receives the raw
    3D windowed arrays. Both models are
    trained from scratch on each call, ensuring no state leaks across folds.

    Args:
        gestures_train (dict): Training gesture clips, mapping class name
            to a list of raw DataFrames for this fold's training subjects.
        gestures_val (dict): Validation gesture clips in the same format,
            for this fold's held-out subject or split.
        ws (int): Window size in samples applied during segmentation. Default 300.

    Returns:
        tuple: A pair (y_val, fold_predictions) where:
            - y_val (np.ndarray): Ground-truth class labels for the validation set.
            - fold_predictions (dict): Maps model name (str) to its predicted
              label array (np.ndarray), keyed as 'Random Forest (Flattened Base)'
              and '1D CNN (Raw Windows)'.
    """
    # Scaling
    scale_gestures(gestures_train, gestures_val)
    
    # Windowing and Filtering
    clipped_train, _ = window_and_filter_gestures(gestures_train, ws=ws, overlap=50, order=5, wn=0.4, filter_type='lowpass')
    clipped_val, _ = window_and_filter_gestures(gestures_val, ws=ws, overlap=50, order=5, wn=0.4, filter_type='lowpass')
    
    # -----------------------------------------------
    # random forest on flattened data
    # -----------------------------------------------
    train_clips_flat = flatten_gestures(clipped_train)
    val_clips_flat = flatten_gestures(clipped_val)
    
    X_train_flat, y_train_flat = prepare_xy(train_clips_flat)
    X_val_flat, y_val = prepare_xy(val_clips_flat) # y_val is the same for both models
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_flat, y_train_flat)
    y_preds_rf = rf_model.predict(X_val_flat)
    
    # -----------------------------------------------
    # 1d CNN on raw 3d data
    # -----------------------------------------------
    X_train_3d, y_train_3d = prepare_xy_3d(clipped_train)
    X_val_3d, y_val_3d = prepare_xy_3d(clipped_val)
    
    y_preds_cnn = train_and_predict_cnn(X_train_3d, X_val_3d, y_train_3d, y_val_3d)
    
    # Return both predictions
    fold_predictions = {
        "Random Forest (Flattened Base)": y_preds_rf,
        "1D CNN (Raw Windows)": y_preds_cnn
    }
    
    return y_val, fold_predictions