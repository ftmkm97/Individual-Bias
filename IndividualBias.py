from tabulate import tabulate
import pandas as pd
import numpy as np
import math
import warnings as w
from scipy.stats import percentileofscore

w.filterwarnings('ignore')
def Sorting(df,Label):
    df = df.sort_values(by=Label ,ascending=False).reset_index().drop('index',axis=1)
    return df

def Normalizing(feature):
    feature = feature.apply(lambda x: math.log(x+1))
    return feature

def calculate_Sq(cutOff,feature,dfs):
    listCut = list(dfs[feature])[:cutOff]
    Sq = sum(val / np.log2(indx + 2) for indx, val in enumerate(listCut))
    
    return Sq

def ScoringFunction(cutOff,feature,dfs):
    Score = sum(calculate_Sq(cutOff,feature,df) for df in dfs)
    Score /= len(dfs)
    return Score

def IB(dfs,GroundTruth,cutOff,Feature):
    Score = ScoringFunction(cutOff,Feature,dfs)

    ScoreGT = ScoringFunction(cutOff,Feature,GroundTruth)
    IBk = Score - ScoreGT

    return IBk,Score,ScoreGT

def RunIB(Database,Method,Feature,cutOffs,df):
    Answerer = pd.read_csv('Data/{}/Data/Answerer.csv'.format(Database))
    Test = pd.read_csv('Data/{}/Data/test.csv'.format(Database))
    Test = pd.merge(Test,Answerer,left_on='OwnerUserId',right_on='userId',how='left')
    # df = pd.read_csv('Data/{}/BaselineResults/{}.csv'.format(Database,Method))
    df = pd.merge(df,Answerer,left_on='AnswererId',right_on='userId',how='left')

    qId = df.QuestionId.unique()
    Test.drop(Test[-Test['ParentId'].isin(qId)].index, inplace = True)

    df[Feature] = Normalizing(df[Feature])
    Test[Feature] = Normalizing(Test[Feature])

    dfs = [x for _, x in df.groupby('QuestionId')]
    dfs = [ Sorting(df,['Score']) for df in dfs]

    GroundTruth = [x for _, x in Test.groupby('ParentId')]
    GroundTruth = [Sorting(df,['AcceptedAnswer']) for df in GroundTruth]

    print('\nIB of ',Feature,'.....')
    data=[]
    for i in cutOffs:
        Bias,_,_ = IB(dfs,GroundTruth,i,Feature)
        data.append([Database,Method,Feature, i , Bias])

    heads = ['Dataset','Method','Feature','CutOff','IB']
    print(tabulate(data, headers=heads, tablefmt='fancy_grid'))

    #df = pd.DataFrame(data=data,columns=heads)
    #df.to_csv('Results/IB_{}_{}_{}.csv'.format(Database,Method,Feature),index=False)
    df = pd.read_csv('Results/IB.csv')
    column = ['Dataset','Method','Feature','CutOff','IB']
    df = df._append(pd.DataFrame(data, columns=column), ignore_index=True)
    df.to_csv('Results/IB.csv',index=False)



def RunSQ(Database,Method,Feature,cut,df):
    Answerer = pd.read_csv('Data/{}/Data/Answerer.csv'.format(Database))
    # df = pd.read_csv('Data/{}/BaselineResults/{}.csv'.format(Database,Method))
    df = pd.merge(df,Answerer,left_on='AnswererId',right_on='userId',how='left')


    df[Feature] = Normalizing(df[Feature])

    dfs = [x for _, x in df.groupby('QuestionId')]
    dfs = [Sorting(df,['Score']) for df in dfs]


    print('\nS(Q) of ',Feature,'.....')
    data=[]
    for i in cut:
        Score = ScoringFunction(i,Feature,dfs)
        data.append([Database,Method,Feature,i , Score])

    heads = ['Dataset','Method','Feature','CutOff','S(Q)']
    print(tabulate(data, headers=heads, tablefmt='fancy_grid'))

    #df = pd.DataFrame(data=data,columns=heads)
    #df.to_csv('Results/Score_{}_{}_{}.csv'.format(Database,Method,Feature),index=False)

    df = pd.read_csv('Results/Score.csv')
    column = ['Dataset','Method','Feature','CutOff','Score']
    df = df._append(pd.DataFrame(data, columns=column), ignore_index=True)
    df.to_csv('Results/Score.csv',index=False)


def BiasImpactScore(df,Q_A,A_F):
    qId = df['QuestionId'][0]
    aIdFirstAnswer = df['AnswererId'][0]
    acceptedAnswerId = Q_A[qId]
    # Check if the first answer is not accepted
    if(aIdFirstAnswer != acceptedAnswerId):
        acceptedA_feature = A_F[acceptedAnswerId]
        firstA_feature = A_F[aIdFirstAnswer]
        # Return a tuple of (1, 1) if the first activity is greater than accepted activity, otherwise (1, 0)
        return (1, 1 if (firstA_feature > acceptedA_feature) else 0)
    else:
        # Return (0, 0) if the first answer is accepted
        return (0, 0)

def RunBIS(Database,Method,Feature,df):
    Answerer = pd.read_csv('Data/{}/Data/Answerer.csv'.format(Database))
    # df = pd.read_csv('Data/{}/BaselineResults/{}.csv'.format(Database,Method))
    Test = pd.read_csv('Data/{}/Data/test.csv'.format(Database))

    Test = Test[Test['AcceptedAnswer']==1]
    Q_A = dict(zip(Test.ParentId,Test.OwnerUserId))
    A_F = dict(zip(Answerer['userId'],Answerer[Feature]))

    dfs = [x for _, x in df.groupby('QuestionId')]
    dfs = [Sorting(df,['Score']) for df in dfs]

    results = list(BiasImpactScore(df,Q_A,A_F) for df in dfs)
    # print(results)
    total_false_answers = sum(result[0] for result in results)
    firstA_is_grather = sum(result[1] for result in results)

    bis = firstA_is_grather/total_false_answers

    print('\nBIS of ',Feature,'.....')
    data = [[Database,Method,Feature, bis]]
    heads = ['Dataset','Method','Feature','BIS']
    print(tabulate(data, headers=heads, tablefmt='fancy_grid'))

    df = pd.read_csv('Results/BIS.csv')
    column = ['Dataset','Method','Feature','BIS']
    df = df._append(pd.DataFrame(data, columns=column), ignore_index=True)
    df.to_csv('Results/BIS.csv',index=False)


    return

def RunADS(Database,Feature):
    Answerer = pd.read_csv('Data/{}/Data/Answerer.csv'.format(Database))
    df = pd.read_csv('Data/{}/Data/Dataset.csv'.format(Database))
    #df = pd.read_csv('Data/{}/Data/Test.csv'.format(Database))
    #df = pd.read_csv('Data/{}/Train.csv'.format(Database))
    df = pd.merge(df,Answerer,left_on='OwnerUserId',right_on='userId',how='left')
    df = df[df['PostTypeId']==2].reset_index(drop=True)

    dfs = [x for _, x in df.groupby('ParentId')]

    fscore_acc = 0 
    for i in range(len(dfs)):
        AcceptedAnswerer = dfs[i][Feature][dfs[i]['AcceptedAnswer']==1].values[0]
        f = percentileofscore(list(Answerer[Feature]), AcceptedAnswerer)
        fscore_acc += f
    fscore_acc = fscore_acc/len(dfs)  


    print('\nADS of ',Feature,'.....')
    data = [[Database,Feature, fscore_acc]]
    heads = ['Dataset','Method','Feature','BIS']
    print(tabulate(data, headers=heads, tablefmt='fancy_grid'))


    df = pd.read_csv('Results/ADS.csv')
    column = ['Dataset','Feature','ADS']
    df = df._append(pd.DataFrame(data, columns=column), ignore_index=True)
    df.to_csv('Results/ADS.csv',index=False)
    return

