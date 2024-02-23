import numpy as np
import pandas as pd
import random


def train_test_split(df,test_size):
    if isinstance(test_size, float):
        test_size= round(test_size*len(df))
    indexList=df.index.tolist()
    testIndexes=random.sample(population=indexList,k=test_size)
    trainIndexes=filter(lambda i: i not in testIndexes, indexList)
    test_df=df.loc[testIndexes]
    train_df=df.loc[trainIndexes]
    return train_df,test_df

def check_purity(data):
    unique_classes = np.unique(data[:,-1])
    if len(unique_classes) == 1:
        return True
    else:
        return False
    
def classify_data(data):
    #from data sent here find the majority element
    
    unique_classes,count_unique_class=np.unique(data[:,-1],return_counts=True)
    index_in_unique_classes=np.argmax(count_unique_class)
    return unique_classes[index_in_unique_classes]

def get_potential_splits(data):
    potential_splits={}
    _,cols = data.shape
    potential_splits={}
    for i in range(cols-1):
        unique_column_values= np.unique(data[:,i])
        
        split_values=[]
        for j in range(len(unique_column_values)):
            if(j!=0):
                split_value=(unique_column_values[j-1]+unique_column_values[j])/2
                split_values.append(split_value)
                
        potential_splits[i]=split_values
    return potential_splits

def split_data(data,split_column,split_value):
    split_column_values=data[:,split_column]
    data_below=data[split_column_values<=split_value]
    data_above=data[split_column_values>split_value]
    return data_below,data_above

def calculate_entropy(data):
    _,counts=np.unique(data[:,-1],return_counts=True)
    probabilities = counts / counts.sum()
    entropy=sum(probabilities*-np.log2(probabilities))
    return entropy

def calculate_overall_entropy(data_below,data_above):
    l_data_below=len(data_below)
    l_data_above=len(data_above)

    p_data_below=l_data_below/(l_data_below+l_data_above)
    p_data_above=l_data_above/(l_data_below+l_data_above)

    overall_entropy=p_data_below*calculate_entropy(data_below) + p_data_above*calculate_entropy(data_above)

    return overall_entropy

def determine_best_split(data,potential_splits):
    overall_entropy=999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below,data_above=split_data(data,column_index,value)
            current_overall_entropy=calculate_overall_entropy(data_below,data_above)
            if current_overall_entropy<=overall_entropy:
                overall_entropy=current_overall_entropy
                best_split_column=column_index
                best_split_value=value
                
    return best_split_column,best_split_value

def decision_tree_algorithm(df, counter=0, min_samples=2,max_depth=5):
    
    #data prepartions
    if counter==0: 
        global column_headers
        column_headers=df.columns
        data=df.values
    else:
        data=df

    #base case
    if (check_purity(data)) or (len(data) < min_samples) or (counter==max_depth) :
        classification = classify_data(data)
        return classification
    #recursion
    else:
        counter+=1
        
        #helper functions
        potential_splits=get_potential_splits(data)
        split_column,split_value=determine_best_split(data,potential_splits)
        data_below, data_above=split_data(data,split_column,split_value)
        
        #instantiate sub tree
        feature_name=column_headers[split_column]
        question="{} <= {}".format(feature_name,split_value)
        sub_tree={question:[]}
        
        #find answers (recursion)
        yes_answer=decision_tree_algorithm(data_below,counter,min_samples,max_depth)
        no_answer=decision_tree_algorithm(data_above,counter,min_samples,max_depth)

        if yes_answer == no_answer:
            sub_tree=yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)

        return sub_tree
    
def classify_example(example,tree):
    question = list(tree.keys())[0]
    feature_name, _, value = question.split()
    
    #ask question
    if example[feature_name] <= float(value):
        answer = tree[question][0]
    else:
        answer = tree[question][1]
    
    #base case
    if not isinstance(answer, dict):
        return answer
    
    #recursive part
    else:
        residual_tree=answer
        return classify_example(example,residual_tree)
    
def calculate_Accuracy(df,tree):
    df["classification"]=df.apply(classify_example,axis=1,args=(tree,))
    df["classification_correct"]=df.classification==df.label
    accuracy=df.classification_correct.mean()
    return accuracy