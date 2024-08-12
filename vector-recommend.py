import pandas as pd
import numpy as np
def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    return data

class vector:
    def __init__(self,vector_columnID=None,init_vector=None, final_vector=None,vectorID=None):

        self.vector_columnID = vector_columnID
        self.init_vector = init_vector
        self.final_vector = final_vector
        self.vectorID = vectorID

    def linearDiff(self):
        difference = self.init_vector - self.final_vector
        distance = np.linalg.norm(difference)
        return distance
    
    def csvVector(self,file_path):
        data = pd.read_csv(file_path)
        vector = data[data[self.vector_columnID] == self.vectorID]
        vector = vector.drop(columns=[self.vector_columnID])
        vector = vector.iloc[0].to_numpy()
        return vector
    
    def dfVector(self, data):
        vector = data[data[self.vector_columnID] == self.vectorID]
        vector = vector.drop(columns=[self.vector_columnID])
        vector = vector.iloc[0].to_numpy()
        return vector
    
class dimension:
    def __init__(self,*vectors):
        self.vectors = vectors

    def vectorVariance(self,ref_vector,realtiveID=False):
        variances = []
        realitive_vectorID = 0
        for comp_vec in self.vectors:
            variance_vector = ref_vector - comp_vec
            variance = np.linalg.norm(variance_vector)
            if realtiveID:
                vectorData = [variance,realitive_vectorID]
                variances.append(vectorData)
                realitive_vectorID += 1
            else:
                variances.append(variance)   
        return variances
    
    def dimReduct(self,tar_featureID,reduction=1):
        data = {'vector': self.vectors}
        df = pd.DataFrame(data)
        df['feature_value'] = df['vector'].apply(lambda x: x[tar_featureID])
        df = df.nlargest(reduction,'feature_value')
        return df['vector']

class featureData:
    def __init__(self, *featureIDs):
        self.featureIDs = featureIDs

    def updateTraits(self,lr=0.001,*vectors):
        new_vectors = []
        for vector in vectors:
            for trait in self.featureIDs:
                vector[trait] = vector[trait] *lr
            new_vectors.append(vector)
        return new_vectors
        
vector1 = vector(vector_columnID='Movie ID',vectorID=5).csvVector('test2.csv')
vector2 = vector(vector_columnID='Movie ID',vectorID=3).csvVector('test2.csv')
vector3 = vector(vector_columnID='Movie ID',vectorID=4).csvVector('test2.csv')

ref_vector = vector(vector_columnID='Movie ID',vectorID=1).csvVector('test2.csv')
data = preprocess_data('test2.csv')

vector4 = vector(vector_columnID='Movie ID',vectorID=6).dfVector(data)
print(vector4)
variances = dimension(vector1, vector2).vectorVariance(ref_vector,realtiveID=True)

new_vectors = featureData(1,2).updateTraits(0.01,vector1,vector2,vector3,)
reduced = dimension(vector1,vector2).dimReduct(2,2)
print(reduced)
