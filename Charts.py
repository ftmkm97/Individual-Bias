import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from collections import Counter

def Distribution_of_users(Feature): 

    df = pd.read_csv('Data/History/Data/Dataset.csv')
    Answerer = pd.read_csv('Data/History/Data/Answerer.csv')
    Answerer.columns = ['OwnerUserId','Reputation','Views','UpVotes','DownVotes','BadgeScore','Activity']
    df = pd.merge(df,Answerer,on='OwnerUserId',how='left')
    accepted_answers = df["AcceptedAnswerId"].dropna().tolist()
    accepted_answers = [int(x) for x in accepted_answers]
    df = df[(df['PostTypeId'] == 2) & (df['Id'].isin(accepted_answers))]
    list1 = (df[Feature]+1).tolist()

    df = pd.read_csv('Data/AI/Data/Dataset.csv')
    Answerer = pd.read_csv('Data/AI/Data/Answerer.csv')
    Answerer.columns = ['OwnerUserId','Reputation','Views','UpVotes','DownVotes','BadgeScore','Activity']
    df = pd.merge(df,Answerer,on='OwnerUserId',how='left')
    accepted_answers = df["AcceptedAnswerId"].dropna().tolist()
    accepted_answers = [int(x) for x in accepted_answers]
    df = df[(df['PostTypeId'] == 2) & (df['Id'].isin(accepted_answers))]
    list2 = (df[Feature]+1).tolist()

    df = pd.read_csv('Data/3DPrinting/Data/Dataset.csv')
    Answerer = pd.read_csv('Data/3DPrinting/Data/Answerer.csv')
    Answerer.columns = ['OwnerUserId','Reputation','Views','UpVotes','DownVotes','BadgeScore','Activity']
    df = pd.merge(df,Answerer,on='OwnerUserId',how='left')
    accepted_answers = df["AcceptedAnswerId"].dropna().tolist()
    accepted_answers = [int(x) for x in accepted_answers]
    df = df[(df['PostTypeId'] == 2) & (df['Id'].isin(accepted_answers))]
    list3 = (df[Feature]+1).tolist()

    df = pd.read_csv('Data/Bioinformatics/Data/Dataset.csv')
    Answerer = pd.read_csv('Data/Bioinformatics/Data/Answerer.csv')
    Answerer.columns = ['OwnerUserId','Reputation','Views','UpVotes','DownVotes','BadgeScore','Activity']
    df = pd.merge(df,Answerer,on='OwnerUserId',how='left')
    accepted_answers = df["AcceptedAnswerId"].dropna().tolist()
    accepted_answers = [int(x) for x in accepted_answers]
    df = df[(df['PostTypeId'] == 2) & (df['Id'].isin(accepted_answers))]
    list4 = (df[Feature]+1).tolist()

    df = pd.read_csv('Data/Biology/Data/Dataset.csv')
    Answerer = pd.read_csv('Data/Biology/Data/Answerer.csv')
    Answerer.columns = ['OwnerUserId','Reputation','Views','UpVotes','DownVotes','BadgeScore','Activity']
    df = pd.merge(df,Answerer,on='OwnerUserId',how='left')
    accepted_answers = df["AcceptedAnswerId"].dropna().tolist()
    accepted_answers = [int(x) for x in accepted_answers]
    df = df[(df['PostTypeId'] == 2) & (df['Id'].isin(accepted_answers))]
    list5 = (df[Feature]+1).tolist()

    list1 = [math.log(x) for x in list1]
    list2 = [math.log(x) for x in list2]
    list3 = [math.log(x) for x in list3]
    list4 = [math.log(x) for x in list4]
    list5 = [math.log(x) for x in list5]

    lists = [list3, list2, list4, list5, list1]
    titles = ['3DPrinting', 'AI', 'Bioinformatics', 'Biology','History']

    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(5, 6), sharex=True)
    fig.patch.set_facecolor('white')
    plot_color = 'navy'

    for i, (data, title) in enumerate(zip(lists, titles)):
        freq = Counter(data)
        x, y = zip(*sorted(freq.items()))
        min_y = min(y)
        max_y = max(y)
        axes[i].plot(x, y, color=plot_color, linestyle='-', linewidth=2)
        axes[i].set_facecolor('white')
        axes[i].text(0.05, 0.95, title, transform=axes[i].transAxes, fontsize=14, weight='bold', ha='left', va='top')
        axes[i].set_yticks([min_y, max_y])
        axes[i].set_yticklabels([f'{min_y}', f'{max_y}'], fontsize=10)
        axes[i].grid(True, linestyle='--', alpha=0.7)
        axes[i].tick_params(axis='both', labelsize=10)

    axes[-1].set_xlabel(Feature, fontsize=20)
    axes[2].set_ylabel('Numeber of Answerer (Log Scale)', fontsize=20)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('Charts/{}.png'.format(Feature+'Distribution'), dpi=300, bbox_inches='tight')
    plt.show()

def BarChart(color,title,Answers,percent_accepted,Dataset):
    bar_width = 0.5
    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.barh(Dataset, Answers, bar_width, color=color, label='Number of Answers')

    bars2 = ax.barh(Dataset, np.array(Answers) * (np.array(percent_accepted) / 100),
                    bar_width, color='none', edgecolor='black', hatch='//', label='Accepted Answer Percentage')

    ax.set_xlabel('Number of Answers')
    ax.set_title(title+' Active Answerer')

    for i, (answers, percent) in enumerate(zip(Answers, percent_accepted)):
        ax.text(answers * (percent / 100) + 100, i, f'{percent}%', va='center', ha='left', color='black', fontsize=10)

    plt.savefig('Charts/{}.png'.format(title+'Distribution'), dpi=300, bbox_inches='tight')
    plt.show()

def Distribution_of_Most_Least_users(Feature,Datasets):
    number_of_answers_low =[]
    percent_accepted_low = []

    number_of_answers_high=[]
    percent_accepted_high = []

    for i in Datasets:
        Answerer = pd.read_csv('Data/{}/Data/Answerer.csv'.format(i))
        Answerer.columns = ['OwnerUserId','Reputation','Views','UpVotes','DownVotes','BadgeScore','Activity']
        df = pd.read_csv('Data/{}/Data/Dataset.csv'.format(i))
        df = pd.merge(df,Answerer,on='OwnerUserId',how='left')
        AcceptedAnswer = pd.DataFrame(df[['OwnerUserId']][df.AcceptedAnswer == 1].value_counts()).reset_index()
        AcceptedAnswer.columns = ['OwnerUserId','countAccepted']

        Answer = pd.DataFrame(df[['OwnerUserId']][df.PostTypeId == 2].value_counts()).reset_index()
        Answer.columns = ['OwnerUserId','countAnswers']

        user = df[['OwnerUserId',Feature]].drop_duplicates().sort_values(by=[Feature], ascending=False).reset_index(drop=True)

        user = pd.merge(Answer,user,on='OwnerUserId',how='left')
        user = pd.merge(user,AcceptedAnswer,on='OwnerUserId',how='left')

        user['countAccepted'] = user['countAccepted'].replace(np.nan,0)

        user = user.sort_values(by=[Feature],ascending=False).reset_index(drop=True)

        # user['group']=''
        # user['group'].iloc[:int(len(user)*0.25)] = 1
        # user['group'].iloc[int(len(user)*0.75):] = 2

        thr = user.Activity.mean()
        user['group']=''
        user['group'][user[Feature]>=thr] = 1
        user['group'][user[Feature]<thr] = 2

        number_of_answers_high.append(user['countAnswers'][user['group']==1].sum())
        percent_accepted_high.append(int(user['countAccepted'][user['group']==1].sum() / user['countAnswers'][user['group']==1].sum() *100))
        
        number_of_answers_low.append(user['countAnswers'][user['group']==2].sum())
        percent_accepted_low.append(int(user['countAccepted'][user['group']==2].sum() / user['countAnswers'][user['group']==2].sum() *100))
        
        
    titleHigh = 'Most'
    colorHigh = 'orange'
    BarChart(colorHigh,titleHigh,number_of_answers_high,percent_accepted_high,Datasets)

        
    titleLow = 'Least'
    colorLow = 'lightblue'
    BarChart(colorLow,titleLow,number_of_answers_low,percent_accepted_low,Datasets)

def Chart(CutOff,Feature,data,datasets,algorithms):
    data = np.array(data)

    bar_width = 0.15
    index = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, algorithm in enumerate(algorithms):
        ax.bar(index + i * bar_width, data[i], bar_width, label=algorithm)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('S {}, {} (Q)'.format(CutOff,Feature))
    ax.set_title('Score(Q,{}) across different datasets for various algorithms'.format(Feature))
    ax.set_xticks(index + bar_width * (len(algorithms) - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.legend()

    plt.tight_layout()

    plt.savefig('Charts/Score_{}_{}.png'.format(Feature,CutOff), dpi=300)
    plt.show()


def ScoreChart(CutOff,Feature):
    df = pd.read_csv('Results/Score.csv')
    df = df[(df['CutOff']==CutOff)&(df['Feature']==Feature)].reset_index(drop=True)

    datasets = ['3DPrinting', 'AI', 'Bioinformatics', 'Biology', 'History']
    algorithms = ['BM25','DSSM', 'PMEF','NeRank', 'BGER', 'TUEF']

    data = [[float('%.1f' % df[(df['Method'] == i) & (df['Dataset'] == j)].Score.values[0]) for j in datasets ] for i in algorithms]
    
    Chart(CutOff,Feature,data,datasets,algorithms)





if __name__ == "__main__":

    Datasets = ['History','Biology','Bioinformatics', 'AI','3DPrinting']
    Feature = 'Activity'
    Distribution_of_Most_Least_users(Feature,Datasets)


    Distribution_of_users('Activity')
    Distribution_of_users('Reputation')
    Distribution_of_users('BadgeScore')

    ScoreChart(5,'Activity')
    ScoreChart(5,'Reputation')
    ScoreChart(5,'BadgeScore')

    ScoreChart(10,'Activity')
    ScoreChart(10,'Reputation')
    ScoreChart(10,'BadgeScore')
