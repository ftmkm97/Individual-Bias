import pandas as pd
import numpy as np
from csv import writer

import Evaluation as ev
import IndividualBias as ib

def CheckExist(filePath,columns):
    try:
        with open(filePath, 'x') as file:
            writer_object = writer(file)
            writer_object.writerow(columns)
            file.close()
    except FileExistsError:
        print('The {} file already exists'.format(filePath))


if __name__ == "__main__":

    Database = '3DPrinting' # '3DPrinting', 'AI', 'Bioinformatics', 'Biology', 'History'
    Method = 'BGER' # 'BM25', , 'DSSM' ,  'PMEF', 'NeRank' , 'TUEF', 'BGER'
    Feature = 'Activity' # 'Activity', 'Reputation', 'BadgeScore'

    file = 'Results/Evaluation.csv'
    columns = ['Dataset','Method','mean_mrr', 'P@1', 'P@3', 'P@5', 'P@10', 'ndcg@5', 'ndcg@10']
    CheckExist(file,columns)

    file = 'Results/IB.csv'
    columns = ['Dataset','Method','Feature','CutOff','IB']
    CheckExist(file,columns)

    file = 'Results/Score.csv'
    columns = ['Dataset','Method','Feature','CutOff','Score']
    CheckExist(file,columns)

    file = 'Results/BIS.csv'
    columns = ['Dataset','Method','Feature','BIS']
    CheckExist(file,columns)

    file = 'Results/ADS.csv'
    columns = ['Dataset','Feature','ADS']
    CheckExist(file,columns)

    df = pd.read_csv('Data/{}/BaselineResults/{}.csv'.format(Database,Method))
    if Method == 'BGER':
        df.Score = df.Score * -1


    ev.run(Database,Method,df)

    cutOffs = [1 ] #, 2, 3, 4, 5, 10]
    ib.RunIB(Database,Method,Feature,cutOffs,df)

    cut = [5, 10]
    ib.RunSQ(Database,Method,Feature,cut,df)

    ib.RunBIS(Database,Method,Feature,df)


    ib.RunADS(Database,Feature)