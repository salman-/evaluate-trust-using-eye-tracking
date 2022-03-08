import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

class Dataset:

    def __init__(self,dataset_path):
        self.dt = pd.read_csv(dataset_path)
        self.dt['Label'] = self.dt['Label'].apply(lambda x: 1 if x > 3 else 0)

        self.features = self.dt.iloc[:,1:-1] # Exclude Participant and Label
        self.features = minmax_scale(self.features)

    def get_train_test_data(self):
        train_x,test_x,train_label,test_label = train_test_split(self.features,self.dt['Label'],train_size=0.8,shuffle=True)
        return train_x,test_x,train_label,test_label