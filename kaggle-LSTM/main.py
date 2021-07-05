import os, sys
sys.path.append("..")

from crema import CremaDataset
import pandas as pd
import numpy as np

import os
import sys

CremaDataset()
# import librosa
# import librosa.display
# import seaborn as sns
# import matplotlib.pyplot as plt

# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.model_selection import train_test_split

# from IPython.display import Audio

# import warnings
# if not sys.warnoptions:
#     warnings.simplefilter("ignore")
# warnings.filterwarnings("ignore", category=DeprecationWarning) 

# # Paths for data.
# Crema = "/kaggle/input/cremad/AudioWAV/"

# crema_directory_list = os.listdir(Crema)

# file_emotion = []
# file_path = []

# for file in crema_directory_list:
#     # storing file paths
#     file_path.append(Crema + file)
#     # storing file emotions
#     part=file.split('_')
#     if part[2] == 'SAD':
#         file_emotion.append('sad')
#     elif part[2] == 'ANG':
#         file_emotion.append('angry')
#     elif part[2] == 'DIS':
#         file_emotion.append('disgust')
#     elif part[2] == 'FEA':
#         file_emotion.append('fear')
#     elif part[2] == 'HAP':
#         file_emotion.append('happy')
#     elif part[2] == 'NEU':
#         file_emotion.append('neutral')
#     else:
#         file_emotion.append('Unknown')
        
# # dataframe for emotion of files
# emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# # dataframe for path of files.
# path_df = pd.DataFrame(file_path, columns=['Path'])
# Crema_df = pd.concat([emotion_df, path_df], axis=1)
# Crema_df.head()