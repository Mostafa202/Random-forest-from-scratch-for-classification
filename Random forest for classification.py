import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sps

dataset=pd.read_csv('r_f.csv')

dataset.columns=['target','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing',
             'gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring',
             'stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population',
             'habitat']






def entropy(data,target_name):
    elements,counts=np.unique(data[target_name],return_counts=True)
    h=np.sum([(counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return h*-1


def info_gain(data,split_attribute,target_name):
    H_data=entropy(data,target_name)
    elements,counts=np.unique(data[split_attribute],return_counts=True)
    H_f=np.sum([(counts[i]/np.sum(counts))*entropy(data.where(elements[i]==data[split_attribute]).dropna(),target_name)for i in range(len(elements))])
    
    return H_data-H_f


def classification(data,original_data,features,target_name,parent_node=None):
    
    if len(np.unique(data[target_name]))<=1:
        return np.unique(data[target_name])[0]
    elif len(data)==0:
        return np.unique(original_data[target_name])[np.argmax(np.unique(original_data[target_name],return_counts=True)[1])]
    elif len(features)==0:
        return parent_node
    else:
        parent_node=np.unique(data[target_name])[np.argmax(np.unique(data[target_name],return_counts=True)[1])]
        features=np.random.choice(features,size=np.int(np.sqrt(len(features))),replace=False)
        best_feature_index=np.argmax([info_gain(data,feature,target_name) for feature in features])
        best_feature=features[best_feature_index]
        tree={best_feature:{}}
        features=[feature for feature in features if feature!=best_feature]
        for val in np.unique(data[best_feature]):
            sub_data=data.where(data[best_feature]==val).dropna()
            sub_tree=classification(sub_data,original_data,features,target_name,parent_node)
            tree[best_feature][val]=sub_tree
        return tree

            
    
def predict(tree,query,default='p'):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                res=tree[key][query[key]]
            except:
                return default
            res=tree[key][query[key]]
            if isinstance(res,dict):
                return predict(res,query,default)
            else:
                return res
        
        
        
def random_forest_train(data,number_trees):
    random_forest_trees=[]
    for tree in range(number_trees):
        random_forest_trees.append(classification(data,data,data.columns[1:],data.columns[0]))
        
    return random_forest_trees

    

def random_forest_predict(random_forest,query,default='p'):
    predictions=[]
    for tree in random_forest:
        predictions.append(predict(tree,query,default))
    return sps.mode(predictions)[0][0]
#    
from sklearn.model_selection import *


train,test=train_test_split(dataset,test_size=0.2,random_state=0)  


def random_forest_test(test,random_forest):
    
    queries=test.iloc[:,1:].to_dict(orient='records')

    predictions=[]
    for q in queries:
        pred=random_forest_predict(random_forest,q)
        predictions.append(pred)
    return (np.sum(np.array(predictions)==np.array(test[test.columns[0]]))/len(test))*100
    
random_forest=random_forest_train(train,10)
print('Accuracy:',random_forest_test(test,random_forest))