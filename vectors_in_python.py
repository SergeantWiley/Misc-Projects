import math
import numpy as np
import pandas as pd
# Import libraries

example_vector = [8,6,10,25,76] #5D Vector
#Create an example format

def component(initial, final):
    component = final - initial
    return component
#Find the component

def magnitude(vector):
    squared_values = []
    for component in vector:
        squared_values.append(component**2)
    length = math.sqrt(sum(squared_values))
    return length


def numpy_vector(vector):
    return np.array(vector)

def csv_vector(file_path,vector_columnID,vectorID):
    data = pd.read_csv(file_path)
    vector = data[data[vector_columnID] == vectorID]
    vector = vector.drop(columns=vector_columnID)
    vector = vector.iloc[0].to_numpy()
    return vector

def transform_vector(vector,lr):
    transformation_vector =  []
    for score in vector:
        new_score = score * lr
        transformation_vector.append(new_score)
    transform_vector = np.array(transformation_vector)
    return transform_vector

x_comp = component(10,15)
y_comp = component(8,15)

user_vector = [0.25, 0.25, 0.25, 0.25]
user_vector = numpy_vector(user_vector)
movie_vector = [0.009,0.356,0.110,0.525]
movie_vector = numpy_vector(movie_vector)
transfomer_vector = transform_vector(movie_vector,0.1)
new_user_vector = user_vector + transfomer_vector
print("Original User Vector:", user_vector)
print("Movie Vector:",movie_vector)
print("Transformation Vector:",transfomer_vector)
print("New User Vector:",new_user_vector)


file_path = 'test2.csv'
vector_columnID = 'Movie ID'
vectorID = 4

movie_vector = csv_vector(file_path,vector_columnID,4)