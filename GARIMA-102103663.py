#!/usr/bin/env python
# coding: utf-8

# ## UCS654 - Predictive Analytics Using Statistics

# ### Assignment 4 - TOPSIS

#  We need to develop a command line python program to implement the Topsis on a given dataset and then also create a Python Package for the same
#  
# The Result file should contain all the columns of input file and two additional columns having Topsis Score and Rank

# In[1]:


import pandas as pd 
import numpy as np
import sys


# #### Step 1: Checking that command line input is correct

# In[2]:

def topsis():
    if len(sys.argv) != 5:
        print("Incorrect number of command line parameters entered! Try again")
        sys.exit(1)

    try:
        inputFile = sys.argv[1]
        df = pd.read_csv(inputFile)
    except FileNotFoundError:
        print("File not Found! Try again")
        sys.exit(1)
        
    try:    
        weights = list(map(float, sys.argv[2].strip().split(',')))
        impacts = list(sys.argv[3].strip().split(','))
    except:
        print("Separate weights and impacts with commas! Try again")
        sys.exit(1)
        
    resultFile = sys.argv[4]

    if(df.shape[1]< 3):
        print("Input file must contain three or more columns! Try again")
        sys.exit(1)
            
    if(( len(weights) == len(impacts) == (df.shape[1]-1)) == False):
        print("Number of weights, number of impacts and number of columns must be same! Try again")
        sys.exit(1)

    for imp in impacts:
        if(imp=='+' or imp=='-'):
            continue
        else:
            print("Impacts must be either positive or negative! Try again")
            sys.exit(1)

    cat_features=[i for i in df.columns[1:] if df.dtypes[i]=='object']
    if(len(cat_features)!=0):
        print("Second to last columns must contain numeric values only! Try again")
        sys.exit(1)


    # In[4]:


    df


    # In[5]:


    feature_values = df.iloc[:,1:].values
    feature_values


    # In[6]:


    options = df.iloc[:,0].values
    options


    # #### Step 2: Convert Categorical to Numerical

    # We have handled this issue earlier by making sure we don't have categorical values in our dataframe

    # #### Step 3 : Vector Normalization

    # In[7]:


    sum_cols=[0]*len(feature_values[0])
    sum_cols   


    # In[8]:


    #Calculating root of sum of squares
    for i in range(len(feature_values)):
        for j in range(len(feature_values[i])):
            sum_cols[j]+=np.square(feature_values[i][j])
            
    for i in range(len(sum_cols)):
        sum_cols[i]=np.sqrt(sum_cols[i])


    # In[9]:


    sum_cols


    # In[10]:


    #Normalized Decision Matrix
    for i in range(len(feature_values)):
        for j in range(len(feature_values[i])):
            feature_values[i][j]=feature_values[i][j]/sum_cols[j]


    # In[11]:


    feature_values


    # #### Step 4: Weight Assignment

    # In[12]:


    weighted_feature_values=[]
    for i in range(len(feature_values)):
        temp=[]
        for j in range(len(feature_values[i])):
            temp.append(feature_values[i][j]*weights[j])
        weighted_feature_values.append(temp)


    # In[ ]:


    weighted_feature_values = np.array(weighted_feature_values)
    weighted_feature_values


    # #### Step 5: Find Ideal Best and Ideal Worst

    # In[ ]:


    #Inititalizing the values of Vj+ and Vj-
    VjPos=[]
    VjNeg=[]
    for i in range(len(weighted_feature_values[0])):
        VjPos.append(weighted_feature_values[0][i])
        VjNeg.append(weighted_feature_values[0][i])


    # In[ ]:


    #Calculating values of Ideal worst and Ideal best arrays
    for i in range(1,len(weighted_feature_values)):
        for j in range(len(weighted_feature_values[i])):
            if impacts[j]=='+':
                if weighted_feature_values[i][j]>VjPos[j]:
                    VjPos[j]=weighted_feature_values[i][j]
                elif weighted_feature_values[i][j]<VjNeg[j]:
                    VjNeg[j]=weighted_feature_values[i][j]
            elif impacts[j]=='-':
                if weighted_feature_values[i][j]<VjPos[j]:
                    VjPos[j]=weighted_feature_values[i][j]
                elif weighted_feature_values[i][j]>VjNeg[j]:
                    VjNeg[j]=weighted_feature_values[i][j]


    # In[ ]:


    VjPos


    # In[ ]:


    VjNeg


    # #### Step 6 : Calculate Euclidean distance
    # 
    # Calculate Euclidean distance from ideal best value and ideal worst value

    # In[ ]:


    Sjpositive=[0]*len(weighted_feature_values)
    Sjnegative=[0]*len(weighted_feature_values)
    for i in range(len(weighted_feature_values)):
        for j in range(len(weighted_feature_values[i])):
            Sjpositive[i]+=np.square(weighted_feature_values[i][j]-VjPos[j])
            Sjnegative[i]+=np.square(weighted_feature_values[i][j]-VjNeg[j])


    # In[ ]:


    for i in range(len(Sjpositive)):
        Sjpositive[i]=np.sqrt(Sjpositive[i])
        Sjnegative[i]=np.sqrt(Sjnegative[i])


    # In[ ]:


    Sjpositive


    # In[ ]:


    Sjnegative


    # #### Step 7: Calculate Performance Score

    # In[ ]:


    performance_score=[0]*len(weighted_feature_values)
    for i in range(len(weighted_feature_values)):
        performance_score[i]=Sjnegative[i]/(Sjnegative[i]+Sjpositive[i])


    # In[ ]:


    performance_score


    # #### Step 8 : TOPSIS Score and Rank

    # In[ ]:


    final_scores_sorted = np.argsort(performance_score) # this returns indices of elements in sorted order
    max_index = len(final_scores_sorted)
    rank = []
    for i in range(len(final_scores_sorted)):
            rank.append(max_index - np.where(final_scores_sorted==i)[0][0])# since we know final_scores_sorted is already sorted, so it will need ranking from back side, so we need to subtract from maximum and get first value of tuple returned by np.where function
    rank_df = pd.DataFrame({"TOPSIS Score" : performance_score, "Ranks": np.array(rank)})


    # In[ ]:


    rank_df


    # In[ ]:


    df = pd.concat([df,rank_df],axis=1)


    # In[ ]:


    df


    # In[ ]:


    df.to_csv(resultFile, index=False)
    
if __name__ == "__main__":
   topsis()
