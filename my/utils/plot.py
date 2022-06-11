import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def drawLossChange(train_losses,val_losses):
    '''
    画出训练过程中的loss变化情况
    '''
    fig, axes = plt.subplots()
    axes.plot(train_losses, label='train loss')
    axes.plot(val_losses, label='validation loss')
    axes.legend()
    plt.show()

def getReport(test_loader, model, device):
    '''
    载入训练好的model，在测试集上测试，打印分类报告并返回all_targets和all_predictions
    '''
    all_targets = []
    all_predictions = []

    for inputs, targets in test_loader:
        # Move to GPU
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)

        # Forward pass
        outputs = model(inputs)
        
        # Get prediction
        # torch.max returns both max and argmax
        _, predictions = torch.max(outputs, 1)

        all_targets.append(targets.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

    all_targets = np.concatenate(all_targets)    
    all_predictions = np.concatenate(all_predictions)  
    print(classification_report(all_targets, all_predictions, digits=4))
    return all_targets, all_predictions

def seeConfusionMatrix(all_targets, all_predictions):
    fig, axes = plt.subplots()
    conf_mat = confusion_matrix(all_targets, all_predictions)
    df_cm = pd.DataFrame(conf_mat, index=['down','flat','up'], columns=['down','flat','up'])
    heatmap = sns.heatmap(df_cm, annot=True, fmt='d', cmap='YlGnBu')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45)
    axes.set_ylabel("true label")
    axes.set_xlabel("predict label")
    plt.show()
