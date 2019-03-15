from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from DataUtils import DataUtils
import pickle

data = DataUtils()
dataloaders = data.load_data(validation_split = 0)
train_size, val_size = data.get_dataset_sizes()
print("DataSet Size (Train: {}, Validation: {})".format(train_size, val_size))

model_names = ["results/Resnet50/lr_0.0003118464108103618_Mon Feb 25 20:01:50 2019/epochs/epoch_15.pt", 
                      "results/Resnet101/lr_0.0003118464108103618_Fri Feb 22 14:07:07 2019/epochs/epoch_5.pt",
                      "results/Resnet101/lr_0.0003118464108103618_Thu Feb 21 11:01:00 2019/epochs/epoch_3.pt"]

def get_features_layer(model, data):
    if model == None:
        model = torch.load(model_names[0])
    
    model = model.module
    feature_layer_model = nn.Sequential(*list(model.children())[:-1])
    #Using more than one GPU
    ######################################
    if torch.cuda.device_count() > 1:
        #print("Let's use", torch.cuda.device_count(), "GPUs!")
        #dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        feature_layer_model = nn.DataParallel(feature_layer_model)

    feature_layer_model = feature_layer_model.to(device)
    #####################################
    data = torch.from_numpy(data)
    feature_layer_output = feature_layer_model(data)
    feature_layer_output = torch.squeeze(feature_layer_output)
    return feature_layer_output

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
model_1 = torch.load(model_names[0])
model_1 = model_1.to(device)
model_2 = torch.load(model_names[1])
model_2 = model_2.to(device)

xg_reg = xgb.XGBClassifier(silent=True, 
                      scale_pos_weight=1,
                      learning_rate=0.001,  
                      colsample_bytree = 0.4,
                      subsample = 0.8,
                      objective='binary:logistic', 
                      n_estimators=100, 
                      reg_alpha = 0.3,
                      max_depth=3, gamma=10)


data_out_X_train=[]
data_out_y_train=[]
data_out_X_test=[]
data_out_y_test=[]
for i, data in enumerate(dataloaders['train']):
    X, Y, _, _ = data
    X = X.to(device)
    Y = Y.to(device)
    X = X.repeat(1,3,1,1)
    
    X_train, X_test, y_train, y_test = train_test_split(np.asarray(X), np.asarray(Y), test_size=0.2, random_state=42)
    feat_out_1 = get_features_layer(model_1, X_train)
    feat_out_2 = get_features_layer(model_2, X_train)
    feat_out = torch.cat((feat_out_1, feat_out_2), 1)    
    X_train_ = feat_out.cpu().detach().numpy()
    y_train_ = y_train

    feat_out_1 = get_features_layer(model_1, X_test)
    feat_out_2 = get_features_layer(model_2, X_test)
    feat_out = torch.cat((feat_out_1, feat_out_2), 1)
    #print(len(feat_out))
    X_test_ = feat_out.cpu().detach().numpy()  
    #y_test_ = np.concatenate((y_test, y_test), 0)
    y_test_ = y_test
    
    for x in X_train_: 
        data_out_X_train.append(x)
    
    for y in y_train_:
        data_out_y_train.append(y)
    
    for x in X_test_:
        data_out_X_test.append(x)
    
    for y in y_test_:
        data_out_y_test.append(y)
        
#torch.cuda.empty_cache()
print(len(data_out_X_train), len(data_out_X_test), len(data_out_y_train), len(data_out_y_test))

xg_reg.fit(torch.FloatTensor(data_out_X_train), data_out_y_train, verbose=True)  
preds = xg_reg.predict(data_out_X_test)
print(len(preds))
rmse = np.sqrt(mean_squared_error(data_out_y_test, preds))
print("RMSE: %f" % (rmse))
accuracy = accuracy_score(data_out_y_test, preds)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
    #import matplotlib.pyplot as plt
    #xgb.plot_importance(xg_reg)
    #plt.rcParams['figure.figsize'] = [5, 5]
    #plt.show()
pickle.dump(xg_reg, open("pima.pickle.dat", "wb"))
    
    