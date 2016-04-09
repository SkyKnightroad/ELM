from ELM import ELM
from util import normalizeData
from loadData import loadData

""" 
# Train on synthetic data
train_data , train_label, test_data, test_label = loadData('synthesis\synthesis', percent = 0.01)

feature_dim = train_data.shape[1]
label_dim = train_label.shape[1]
 
train_data = normalizeData(train_data)
test_data = normalizeData(test_data)

elm = ELM(feature_dim, feature_dim*10, label_dim, 'lite', 'dec')

elm.trainModel(train_data, train_label)
elm.save(r"D:\workspace\Data\Data Synthesis\synthesis\synthesis\weights\elm1")
elm.testModel(test_data, test_label)
"""

# Train on real data
train_data , train_label, test_data, test_label = loadData('REAL\greyscale', percent = 1)

feature_dim = train_data.shape[1]
label_dim = train_label.shape[1]

train_data = normalizeData(train_data)
test_data = normalizeData(test_data)

elm = ELM(feature_dim, feature_dim*10, label_dim, 'lite', 'dec')
elm.trainModel(train_data, train_label)
elm.save(r"D:\workspace\Data\Data Synthesis\synthesis\synthesis\weights\elmReal")
elm.testModel(test_data, test_label)


"""
# Train on synthetic data and fun-tune on real data
from numpy import concatenate
train_data , train_label, test_data, test_label = loadData('Real',  percent = 1)
data = concatenate((train_data, test_data), axis=0)
label = concatenate((train_label, test_label), axis=0)

feature_dim = data.shape[1]
label_dim = label.shape[1]
data = normalizeData(data)

elm = ELM(feature_dim, feature_dim*10, label_dim, 'lite', 'dec')
elm.load(r"D:\workspace\Data\Data Synthesis\synthesis\synthesis\weights\elm") #different results each time load??
elm.trainModel(data, label)
elm.testModel(data, label)
"""
