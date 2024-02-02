import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import copy
from scipy import stats

dpi=1200
stepsize_roc=200
stepsize_pr=200
stepsize_conf=100

max_neg = 7037
max_pos = 2029
tot_cases = 9066

XGB_index_data = pd.read_csv('Data/Full curves/curve_xgb_index_calibrated.csv', index_col=None, header=None)
XGB_agg_data = pd.read_csv('Data/Full curves/curve_xgb_agg_calibrated.csv', index_col=None, header=None)
RFC_index_data = pd.read_csv('Data/Full curves/curve_rfc_index_calibrated.csv', index_col=None, header=None)
RFC_agg_data = pd.read_csv('Data/Full curves/curve_rfc_agg_calibrated.csv', index_col=None, header=None)
MM_data = pd.read_csv('Data/Full curves/curve_mm.csv', index_col=None, header=None)
LSTM_data = pd.read_csv('Data/Full curves/curve_lstm.csv', index_col=None, header=None)

XGB_index_data.columns = ['run_%s'%(i) for i in range (1,21)]
XGB_agg_data.columns= ['run_%s'%(i) for i in range (1,21)]
RFC_index_data.columns= ['run_%s'%(i) for i in range (1,21)]
RFC_agg_data.columns= ['run_%s'%(i) for i in range (1,21)]
MM_data.columns= ['run_%s'%(i) for i in range (1,21)]
LSTM_data.columns= ['run_%s'%(i) for i in range (1,21)]

XGB_index_data_clean=pd.DataFrame(data=None, columns=XGB_index_data.columns, index=XGB_index_data.index)
XGB_agg_data_clean=pd.DataFrame(data=None, columns=XGB_agg_data.columns, index=XGB_agg_data.index)
RFC_index_data_clean=pd.DataFrame(data=None, columns=RFC_index_data.columns, index=RFC_index_data.index)
RFC_agg_data_clean=pd.DataFrame(data=None, columns=RFC_agg_data.columns, index=RFC_agg_data.index)
MM_data_clean=pd.DataFrame(data=None, columns=MM_data.columns, index=MM_data.index)
LSTM_data_clean=pd.DataFrame(data=None, columns=LSTM_data.columns, index=LSTM_data.index)

data_list=[XGB_index_data,XGB_agg_data,RFC_index_data,RFC_agg_data,MM_data,LSTM_data]
data_clean_list=[XGB_index_data_clean,XGB_agg_data_clean,RFC_index_data_clean,RFC_agg_data_clean,MM_data_clean,LSTM_data_clean]
clean_table=pd.DataFrame(data=None, columns=XGB_index_data_clean.columns, index=XGB_index_data_clean.index)

for m in range(6):
    data = data_list[m]
    data_clean = data_clean_list[m]

    for i in range (20):
        for j in range (3):
            temp_string=data.iloc[j,i].replace('\n ', ' ')
            temp_string=temp_string.replace('    ',' ')
            temp_string=temp_string.replace('   ',' ')
            temp_string=temp_string.replace('  ',' ')
            temp_string=temp_string.replace(' ',', ')
            data_clean.iloc[j,i]=ast.literal_eval(temp_string)
            
            temp_string2=data.iloc[j+4,i].replace('\n ', ' ')
            temp_string2=temp_string2.replace('    ',' ')
            temp_string2=temp_string2.replace('   ',' ')
            temp_string2=temp_string2.replace('  ',' ')
            temp_string2=temp_string2.replace(' ',', ')
            data_clean.iloc[j+4,i]=ast.literal_eval(temp_string2)
            
    data_clean.drop([3,7],inplace=True)
    data_clean.reset_index(drop=True, inplace=True)

XGB_idx_chart = pd.DataFrame(data=None, index=None, columns=['fpr','tpr'])
XGB_agg_chart = pd.DataFrame(data=None, index=None, columns=['fpr','tpr'])
RFC_idx_chart = pd.DataFrame(data=None, index=None, columns=['fpr','tpr'])
RFC_agg_chart = pd.DataFrame(data=None, index=None, columns=['fpr','tpr'])
MM_chart = pd.DataFrame(data=None, index=None, columns=['fpr','tpr'])
LSTM_chart = pd.DataFrame(data=None, index=None, columns=['fpr','tpr'])
All_chart = pd.DataFrame(data=None, index=None, columns=['fpr','tpr','Algorithm'])

chart_list_roc=[XGB_idx_chart,XGB_agg_chart,RFC_idx_chart,RFC_agg_chart,MM_chart,LSTM_chart]
data_clean_list=[XGB_index_data_clean,XGB_agg_data_clean,RFC_index_data_clean,RFC_agg_data_clean,MM_data_clean,LSTM_data_clean]
algorithm_list=['XGB_idx','XGB_agg','RFC_idx','RFC_agg','MM','LSTM']

#ROC curves

stepsize=stepsize_roc

for d in range(6):
    data_clean=data_clean_list[d]
    data_chart=chart_list_roc[d]
    algorithm=algorithm_list[d]
    
    print(algorithm)
    
    for r in range (1,21):
        
        print('run_%s'%(r))

        temp_df = pd.DataFrame(data=data_clean.loc[0,'run_%s'%r], index=range(len(data_clean.loc[0,'run_%s'%r])), columns=['tpr'])
        temp_df['fpr']=data_clean.loc[1,'run_%s'%r]


        smooth_df = pd.DataFrame(data=None, index=range(stepsize), columns=temp_df.columns)
        smooth_df.loc[0]=1

        for i in range(1,stepsize):

            value_df=temp_df[((temp_df['fpr']>=(1-((i+1)/stepsize))) & (temp_df['fpr']<(1-((i-1)/stepsize))))]
            smooth_df.loc[i,'tpr']=np.mean(value_df['tpr'])
            smooth_df.loc[i,'fpr']=(1-(i/stepsize))

        smooth_df.fillna(method='bfill', inplace=True)

        data_chart = pd.concat([data_chart, smooth_df], axis=0)

    data_chart['Algorithm']=algorithm
    chart_list_roc[d]=data_chart
    
XGB_idx_chart=chart_list_roc[0]
XGB_agg_chart=chart_list_roc[1]
RFC_idx_chart=chart_list_roc[2]
RFC_agg_chart=chart_list_roc[3]
MM_chart=chart_list_roc[4]
LSTM_chart=chart_list_roc[5]

XGB_idx_chart.reset_index(drop=True, inplace=True)
XGB_agg_chart.reset_index(drop=True, inplace=True)
RFC_idx_chart.reset_index(drop=True, inplace=True)
RFC_agg_chart.reset_index(drop=True, inplace=True)
MM_chart.reset_index(drop=True, inplace=True)
LSTM_chart.reset_index(drop=True, inplace=True)

chart_list_roc=[XGB_idx_chart,XGB_agg_chart,RFC_idx_chart,RFC_agg_chart,MM_chart,LSTM_chart]
    
All_chart = pd.concat(chart_list_roc,axis=0)
All_chart.reset_index(drop=True, inplace=True)

sns.set(font_scale=1.2,rc={"lines.linewidth": 1, 'figure.figsize':(10,10), 'figure.dpi':dpi, 'xtick.top' : False, 'markers.fillstyle': 'full', "lines.markersize":10  })
sns.set_style("white")
s=sns.lineplot(x='fpr', y='tpr', data=All_chart, hue='Algorithm', hue_order=algorithm_list, style='Algorithm', 
             dashes=False, markers=True, ci=1, markevery=20)

s.set_ylim(0, 1.05)
plt.title('Receiver-Operator Curve', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=12)
plt.xlabel(None, fontsize=12)




# PR curves

XGB_idx_chart_pr = pd.DataFrame(data=None, index=None, columns=['recall','precision','thres_pr'])
XGB_agg_chart_pr = pd.DataFrame(data=None, index=None, columns=['recall','precision','thres_pr'])
RFC_idx_chart_pr = pd.DataFrame(data=None, index=None, columns=['recall','precision','thres_pr'])
RFC_agg_chart_pr = pd.DataFrame(data=None, index=None, columns=['recall','precision','thres_pr'])
MM_chart_pr = pd.DataFrame(data=None, index=None, columns=['recall','precision','thres_pr'])
LSTM_chart_pr = pd.DataFrame(data=None, index=None, columns=['recall','precision','thres_pr'])
All_chart_pr = pd.DataFrame(data=None, index=None, columns=['recall','precision','thres_pr','Algorithm'])

chart_list_pr=[XGB_idx_chart_pr,XGB_agg_chart_pr,RFC_idx_chart_pr,RFC_agg_chart_pr,MM_chart_pr,LSTM_chart_pr]
data_clean_list=[XGB_index_data_clean,XGB_agg_data_clean,RFC_index_data_clean,RFC_agg_data_clean,MM_data_clean,LSTM_data_clean]
algorithm_list=['XGB_idx','XGB_agg','RFC_idx','RFC_agg','MM','LSTM']

stepsize=stepsize_pr

for d in range(6):
    data_clean=data_clean_list[d]
    data_chart_pr=chart_list_pr[d]
    algorithm=algorithm_list[d]
    
    print(algorithm)
    
    for r in range (1,21):
        
        print('run_%s'%(r))

        temp_df = pd.DataFrame(data=data_clean.loc[3,'run_%s'%r], index=range(len(data_clean.loc[3,'run_%s'%r])), columns=['precision'])
        temp_df['recall']=data_clean.loc[4,'run_%s'%r]
        thres_list=data_clean.loc[5,'run_%s'%r]
        thres_list.append(1)
        temp_df['thres_pr']=thres_list

        smooth_df = pd.DataFrame(data=None, index=range(stepsize), columns=temp_df.columns)

        for i in range(1,stepsize):

            value_df=temp_df[((temp_df['recall']>=(1-((i+1)/stepsize))) & (temp_df['recall']<(1-((i-1)/stepsize))))]
            smooth_df.loc[i,'precision']=np.mean(value_df['precision'])
            smooth_df.loc[i,'recall']=(1-(i/stepsize))
            smooth_df.loc[i,'thres_pr']=np.mean(value_df['thres_pr'])

        smooth_df.fillna(method='bfill', inplace=True)

        data_chart_pr = pd.concat([data_chart_pr, smooth_df], axis=0)

    data_chart_pr['Algorithm']=algorithm
    chart_list_pr[d]=data_chart_pr
    
XGB_idx_chart_pr=chart_list_pr[0]
XGB_agg_chart_pr=chart_list_pr[1]
RFC_idx_chart_pr=chart_list_pr[2]
RFC_agg_chart_pr=chart_list_pr[3]
MM_chart_pr=chart_list_pr[4]
LSTM_chart_pr=chart_list_pr[5]

XGB_idx_chart_pr.reset_index(drop=True, inplace=True)
XGB_agg_chart_pr.reset_index(drop=True, inplace=True)
RFC_idx_chart_pr.reset_index(drop=True, inplace=True)
RFC_agg_chart_pr.reset_index(drop=True, inplace=True)
MM_chart_pr.reset_index(drop=True, inplace=True)
LSTM_chart_pr.reset_index(drop=True, inplace=True)

chart_list_pr=[XGB_idx_chart_pr,XGB_agg_chart_pr,RFC_idx_chart_pr,RFC_agg_chart_pr,MM_chart_pr,LSTM_chart_pr]
    
All_chart_pr = pd.concat(chart_list_pr,axis=0)
All_chart_pr.reset_index(drop=True, inplace=True)

sns.set(font_scale=1.2,rc={"lines.linewidth": 3, 'figure.figsize':(10,10), 'figure.dpi':dpi, 'xtick.top' : False, 'markers.fillstyle': 'full', "lines.markersize":10  })
sns.set_style("white")
s=sns.lineplot(x='recall', y='precision', data=All_chart_pr, hue='Algorithm', hue_order=algorithm_list, style='Algorithm', 
             dashes=False, markers=True, ci=1, markevery=stepsize_pr/10)
 s.set_ylim(0, 1)
plt.title('Precision-Recall Curve', fontsize=16)
plt.ylabel('Precision', fontsize=12)
plt.xlabel(None, fontsize=16)

XGB_idx_conf = pd.DataFrame(data=None, index=None, columns=['TP','FP','TN','FN'])
XGB_agg_conf = pd.DataFrame(data=None, index=None, columns=['TP','FP','TN','FN'])
RFC_idx_conf = pd.DataFrame(data=None, index=None, columns=['TP','FP','TN','FN'])
RFC_agg_conf = pd.DataFrame(data=None, index=None, columns=['TP','FP','TN','FN'])
MM_conf = pd.DataFrame(data=None, index=None, columns=['TP','FP','TN','FN'])
LSTM_conf = pd.DataFrame(data=None, index=None, columns=['TP','FP','TN','FN'])
All_conf = pd.DataFrame(data=None, index=None, columns=['TP','FP','TN','FN','Algorithm'])

conf_list=[XGB_idx_conf,XGB_agg_conf,RFC_idx_conf,RFC_agg_conf,MM_conf,LSTM_conf]
chart_list=[XGB_idx_chart,XGB_agg_chart,RFC_idx_chart,RFC_agg_chart,MM_chart,LSTM_chart]
data_clean_list=[XGB_index_data_clean,XGB_agg_data_clean,RFC_index_data_clean,RFC_agg_data_clean,MM_data_clean,LSTM_data_clean]
algorithm_list=['XGB_idx','XGB_agg','RFC_idx','RFC_agg','MM','LSTM']


#Result Table

stepsize = stepsize_conf

for d in range(6):
    data_clean=data_clean_list[d]
    data_conf=conf_list[d]
    algorithm=algorithm_list[d]
    
    print(algorithm)
    
    for r in range (1,21):
        
        print('run_%s'%(r))

        temp_df = pd.DataFrame(data=data_clean.loc[2,'run_%s'%r], index=range(len(data_clean.loc[2,'run_%s'%r])), columns=['thres'])
        temp_df['fpr']=data_clean.loc[1,'run_%s'%r]
        temp_df['tpr']=data_clean.loc[0,'run_%s'%r]
        
        temp_df2 = pd.DataFrame(data=data_clean.loc[5,'run_%s'%r], index=range(len(data_clean.loc[5,'run_%s'%r])), columns=['thres_pr'])
        temp_df2['recall']=data_clean.loc[4,'run_%s'%r]
        temp_df2['precision']=data_clean.loc[3,'run_%s'%r]

        smooth_df = pd.DataFrame(data=None, index=range(stepsize), columns=temp_df.columns)       

        for i in range(1,stepsize):

            value_df=temp_df[((temp_df['thres']>=(1-((i+1)/stepsize))) & (temp_df['thres']<(1-((i-1)/stepsize))))]
            value_df2=temp_df2[((temp_df2['thres_pr']>=(1-((i+1)/stepsize))) & (temp_df2['thres_pr']<(1-((i-1)/stepsize))))]
            smooth_df.loc[i,'tpr']=np.mean(value_df['tpr'])
            smooth_df.loc[i,'fpr']=np.mean(value_df['fpr'])
            smooth_df.loc[i,'precision']=np.mean(value_df2['precision'])
            smooth_df.loc[i,'recall']=np.mean(value_df2['recall'])
            smooth_df.loc[i,'thres']=(1-(i/stepsize))

        smooth_df.loc[0,'thres']=1
        smooth_df.loc[0,'tpr']=0
        smooth_df.loc[0,'fpr']=0
        smooth_df.loc[0,'precision']=np.nan
        smooth_df.loc[0,'recall']=0
        smooth_df.fillna(method='ffill', inplace=True)
        smooth_df.fillna(method='bfill', inplace=True)
        data_conf = pd.concat([data_conf, smooth_df], axis=0)

    data_conf['Algorithm']=algorithm
   
    
    data_conf['TP']=data_conf['tpr']*max_pos
    data_conf['FP']=data_conf['fpr']*max_neg
    data_conf['TN']=(1-data_conf['fpr'])*max_neg
    data_conf['FN']=(1-data_conf['tpr'])*max_pos
   
    conf_list[d]=data_conf
    
XGB_idx_conf=conf_list[0]
XGB_agg_conf=conf_list[1]
RFC_idx_conf=conf_list[2]
RFC_agg_conf=conf_list[3]
MM_conf=conf_list[4]
LSTM_conf=conf_list[5]

XGB_idx_conf.reset_index(drop=True, inplace=True)
XGB_agg_conf.reset_index(drop=True, inplace=True)
RFC_idx_conf.reset_index(drop=True, inplace=True)
RFC_agg_conf.reset_index(drop=True, inplace=True)
MM_conf.reset_index(drop=True, inplace=True)
LSTM_conf.reset_index(drop=True, inplace=True)

conf_list=[XGB_idx_conf,XGB_agg_conf,RFC_idx_conf,RFC_agg_conf,MM_conf,LSTM_conf]
    
All_conf = pd.concat(conf_list,axis=0)
All_conf.reset_index(drop=True, inplace=True)


# C(-|+)
cost_fn = 4
# C(+|-)
cost_fp = 1

for d in range(6):
    
    data_pr=chart_list_pr[d]
    data=data_list[d]
    conf=conf_list[d]
    
    data_pr['F1_score']=2*(data_pr['precision']*data_pr['recall'])/(data_pr['precision']+data_pr['recall'])
    data_pr['F2_score']=5*(data_pr['precision']*data_pr['recall'])/((data_pr['precision']*4)+data_pr['recall'])
    data_pr['F0.5_score']=1.25*(data_pr['precision']*data_pr['recall'])/((data_pr['precision']*0.25)+data_pr['recall'])
    conf['MCC']=((conf['TP']*conf['TN'])-(conf['FP']*conf['FN']))/(((conf['TP']+conf['FP'])*(conf['TP']+conf['FN'])*(conf['TN']+conf['FP'])*(conf['TN']+conf['FN']))**(1/2))
   
    conf['Expected Cost']=(cost_fp*conf['fpr']*(max_neg/tot_cases)+cost_fn*(1-conf['tpr'])*(max_pos/tot_cases))
    conf['Expected Normalized Cost']=conf['Expected Cost'] / (cost_fp*(max_neg/tot_cases)+cost_fn*(max_pos/tot_cases))
    
    data.loc[8]=None
    data.loc[9]=None
    data.loc[10]=None
    data.loc[11]=None
    data.loc[12]=None
    data.loc[13]=None
    data.loc[14]=None
    data.loc[15]=None
    data.loc[16]=None
    data.loc[17]=None

    
    for r in range (20):
        
        conf_short=conf.iloc[stepsize_conf*r:stepsize_conf*r+stepsize_conf].reset_index(drop=True)
        data_pr_short=data_pr.iloc[stepsize_pr*r:stepsize_pr*r+stepsize_pr].reset_index(drop=True)
                
         data.iloc[8,r]=np.max(data_pr.iloc[(0+(r*stepsize_pr)):(stepsize_pr+(r*stepsize_pr)),4])
         data.iloc[10,r]=np.max(data_pr.iloc[(0+(r*stepsize_pr)):(stepsize_pr+(r*stepsize_pr)),5])
         data.iloc[12,r]=np.max(data_pr.iloc[(0+(r*stepsize_pr)):(stepsize_pr+(r*stepsize_pr)),6])
         data.iloc[14,r]=np.max(conf.iloc[(0+(r*stepsize_conf)):(stepsize_conf+(r*stepsize_conf)),13])
         data.iloc[16,r]=np.max(conf.iloc[(0+(r*stepsize_conf)):(stepsize_conf+(r*stepsize_conf)),14])
         data.iloc[9,r]=np.mean(data_pr[data_pr['F1_score']==data.iloc[8,r]]['thres_pr'])
         data.iloc[11,r]=np.mean(data_pr[data_pr['F2_score']==data.iloc[10,r]]['thres_pr'])
         data.iloc[13,r]=np.mean(data_pr[data_pr['F0.5_score']==data.iloc[12,r]]['thres_pr'])
         data.iloc[15,r]=np.mean(conf[conf['MCC']==data.iloc[16,r]]['thres'])
         data.iloc[17,r]=np.mean(conf[conf['Cost']==data.iloc[16,r]]['thres'])

        data.iloc[8,r]=np.mean(data_pr_short[((data_pr_short['thres_pr']>0.5-(1/stepsize_conf))&
                                              (data_pr_short['thres_pr']<0.5+(1/stepsize_conf)))]['F1_score'])
        data.iloc[9,r]=np.mean(data_pr_short[((data_pr_short['thres_pr']>0.5-(1/stepsize_conf))&
                                              (data_pr_short['thres_pr']<0.5+(1/stepsize_conf)))]['precision'])
        
        data.iloc[10,r]=np.mean(data_pr_short[((data_pr_short['thres_pr']>0.5-(1/stepsize_conf))&
                                              (data_pr_short['thres_pr']<0.5+(1/stepsize_conf)))]['F2_score'])
        
        data.iloc[12,r]=np.mean(data_pr_short[((data_pr_short['thres_pr']>0.5-(1/stepsize_conf))&
                                              (data_pr_short['thres_pr']<0.5+(1/stepsize_conf)))]['F0.5_score'])
        
        data.iloc[14,r]=np.mean(conf_short[conf_short['thres']==0.5]['MCC'])
        
        data.iloc[16,r]=np.min(conf_short['Expected Normalized Cost'])
        data.iloc[17,r]=1-(np.argmin(conf_short['Expected Normalized Cost'])/100)

        
XGB_index_data=data_list[0]
XGB_agg_data=data_list[1]
RFC_index_data=data_list[2]
RFC_agg_data=data_list[3]
MM_data=data_list[4]
LSTM_data=data_list[5]

data_list=[XGB_index_data,XGB_agg_data,RFC_index_data,RFC_agg_data,MM_data,LSTM_data]

XGB_idx_conf=conf_list[0]
XGB_agg_conf=conf_list[1]
RFC_idx_conf=conf_list[2]
RFC_agg_conf=conf_list[3]
MM_conf=conf_list[4]
LSTM_conf=conf_list[5]

conf_list=[XGB_idx_conf,XGB_agg_conf,RFC_idx_conf,RFC_agg_conf,MM_conf,LSTM_conf]
    
All_conf = pd.concat(conf_list,axis=0)
All_conf.reset_index(drop=True, inplace=True)


XGB_idx_chart_pr=chart_list_pr[0]
XGB_agg_chart_pr=chart_list_pr[1]
RFC_idx_chart_pr=chart_list_pr[2]
RFC_agg_chart_pr=chart_list_pr[3]
MM_chart_pr=chart_list_pr[4]
LSTM_chart_pr=chart_list_pr[5]

chart_list_pr=[XGB_idx_chart_pr,XGB_agg_chart_pr,RFC_idx_chart_pr,RFC_agg_chart_pr,MM_chart_pr,LSTM_chart_pr]
    
All_chart_pr = pd.concat(chart_list_pr,axis=0)
All_chart_pr.reset_index(drop=True, inplace=True)

# cost curves

# %% INPUTS

cost_ratio=cost_fn/(cost_fn+cost_fp)

granularity = 0.01
stepsize_cost = len(np.arange(0, 1.01, granularity))

algorithm_list=['XGB_idx','XGB_agg','RFC_idx','RFC_agg','MM','LSTM']

cost_curve_all=pd.DataFrame(data=None, index=algorithm_list, columns=XGB_agg_data_clean.columns)
cost_curve_df=pd.DataFrame(data=None, index=algorithm_list, columns=range(stepsize_cost))



for data_clean, alg in zip(data_clean_list, algorithm_list):
    
    pc=[None]*20
    lines=[None]*20
    lower_envelope=[None]*20
    area=[None]*20

    for i in range(20):

        # %% OUTPUTS

        # 1D-array of x-axis values (normalized PC)
        pc[i] = None
        # list of lines as (slope, intercept)
        lines[i] = []
        # lower envelope of the list of lines as a 1D-array of y-axis values (NEC)
        lower_envelope[i] = []

        # %% COMPUTATION

        # points from the roc curve, because a point in the ROC space <=> a line in the cost space
        roc_fpr=data_clean.loc[1,'run_%s'%(i+1)]
        roc_tpr=data_clean.loc[0,'run_%s'%(i+1)]

        # compute the normalized p(+)*C(-|+)
        pos_proportion = np.arange(0, 1.01, .01)
        pc[i] = (pos_proportion*cost_ratio) / (pos_proportion*cost_ratio + (1-pos_proportion)*cost_ratio)

        # compute a line in the cost space for each point in the roc space
        for fpr, tpr in zip(roc_fpr, roc_tpr):
            slope = (1-tpr-fpr)
            intercept = fpr
            lines[i].append((slope, intercept))

        # compute the lower envelope
        for x_value in pc[i]:
            y_value = min([slope*x_value+intercept for slope, intercept in lines[i]])
            lower_envelope[i].append(max(0, y_value))
        lower_envelope[i] = np.array(lower_envelope[i])
        
        cost_curve_all.loc[alg,'run_%s'%(i+1)]=np.array(lower_envelope[i])


    cost_curve_df.loc[alg]=pd.DataFrame(data=lower_envelope).mean().T

cost_curve_df.loc['probabilitycost']=pc[0]

#result table

score_df = pd.DataFrame(data=None, columns=['AUC-ROC (Mean)', 'AUC-ROC (Std)', 'AUC-ROC (max)','AUC-PR (Mean)', 'AUC-PR (Std)', 'AUC-PR (max)'], index=algorithm_list)
data_list=[XGB_index_data,XGB_agg_data,RFC_index_data,RFC_agg_data,MM_data,LSTM_data]


for d in range(6):
    data=data_list[d]
    algorithm=algorithm_list[d]
    
    score_df.loc[algorithm,'AUC-ROC (Mean)']=np.mean(data.loc[3].astype(float))
    score_df.loc[algorithm,'AUC-ROC (Std)']=np.std(data.loc[3].astype(float))
    score_df.loc[algorithm,'AUC-ROC (max)']=np.max(data.loc[3].astype(float))
    score_df.loc[algorithm,'AUC-PR (Mean)']=np.mean(data.loc[7].astype(float))
    score_df.loc[algorithm,'AUC-PR (Std)']=np.std(data.loc[7].astype(float))
    score_df.loc[algorithm,'AUC-PR (max)']=np.max(data.loc[7].astype(float))   
    score_df.loc[algorithm, 'F0.5-Score']=np.mean(data.loc[12].astype(float))   
    score_df.loc[algorithm, 'F1-Score']=np.mean(data.loc[8].astype(float))
    score_df.loc[algorithm, 'precision']=np.mean(data.loc[9].astype(float))
    score_df.loc[algorithm, 'F2-Score']=np.mean(data.loc[10].astype(float))  
    score_df.loc[algorithm, 'MCC']=np.mean(data.loc[14].astype(float))  
    score_df.loc[algorithm, 'Monetary Cost']=np.mean(data.loc[16].astype(float))
    score_df.loc[algorithm, '@ threshold C']=np.mean(data.loc[17].astype(float))
    
score_df


s=sns.lineplot(x='thres', y='Expected Normalized Cost', data=All_conf, hue='Algorithm', hue_order=algorithm_list, style='Algorithm', 
             dashes=False, markers=True, ci=None, markevery=10, legend=False)

plt.title('Dataset Full', fontsize=8)
plt.ylabel('Normalized Expected Cost', fontsize=8)
plt.xlabel('Threshold t', fontsize=8)

s.set(ylim=(0,1))
s.set(xlim=(0,1))
s.spines['left'].set_linewidth(0.6)
s.spines['bottom'].set_linewidth(0.6)
s.spines['top'].set_linewidth(0.1)
s.spines['right'].set_linewidth(0.1)
s.grid(which='major', linewidth=0.4)

fig.tight_layout()

plt.savefig('deploy_cost_curve_full.svg',bbox_inches='tight')


sns.set(font_scale=1.2,rc={"lines.linewidth": 1, 'figure.dpi':600, 'xtick.top' : False, 'markers.fillstyle': 'full', "lines.markersize":4,'lines.markeredgewidth':0  })
sns.set_style("whitegrid")

s=sns.lineplot(data=cost_curve_df.T.set_index('probabilitycost'), hue_order=algorithm_list,
             dashes=False, markers=True, ci=1, markevery=10, legend=False)

s.axline([0,0],[0.5,0.5], color='black', linewidth=0.5, linestyle='--')
s.axline([0,1],[0.5,0.5], color='black', linewidth=0.5, linestyle='--')

plt.xlabel("Probability Cost Function")
plt.ylabel("Normalized Expected Cost")
plt.title("Cost curve")
plt.show()

fig, axes = plt.subplots(1,3, figsize=(7.16,(7.16/3)))

fig.suptitle('Dataset Full', y=1.0)


sns.set(font_scale=0.7,rc={"lines.linewidth": 0.7, 'figure.dpi':1200, 'xtick.top' : False, 'markers.fillstyle': 'full', "lines.markersize":4,'lines.markeredgewidth':0 })
sns.set_style("white")

roc_graph=sns.lineplot(ax=axes[0], x='fpr', y='tpr', data=All_chart, hue='Algorithm', hue_order=algorithm_list, style='Algorithm', 
             dashes=False, markers=True, ci=None, markevery=int(stepsize_roc*0.2), legend=True)

roc_graph.axline([0,0],[1,1], color='black', linewidth=0.3, linestyle='--')

axes[0].set_title('Receiver Operating Characteristic Curve')
axes[0].set(xlabel=None, ylabel='True Positive Rate')
axes[0].set(ylim=(0,1))
axes[0].set(xlim=(0,1))
axes[0].spines['left'].set_linewidth(0.6)
axes[0].spines['bottom'].set_linewidth(0.6)
axes[0].grid(which='major', linewidth=0.4)

pr_graph=sns.lineplot(ax=axes[1],x='recall', y='precision', data=All_chart_pr, hue='Algorithm', hue_order=algorithm_list, style='Algorithm', 
             dashes=False, markers=True, ci=None, markevery=int(stepsize_pr*0.2), legend=False)

pr_graph.axhline((max_pos/tot_cases), color='black', linewidth=0.3, linestyle='--')


axes[1].set_title('Precision-Recall Curve')
axes[1].set(xlabel=None, ylabel='Precision')
axes[1].set(ylim=(0,1))
axes[1].set(xlim=(0,1))
axes[1].spines['left'].set_linewidth(0.6)
axes[1].spines['bottom'].set_linewidth(0.6)
axes[1].grid(which='major', linewidth=0.4)



cos_graph=sns.lineplot(ax=axes[2],data=cost_curve_df.T.set_index('probabilitycost'), hue_order=algorithm_list,
             dashes=False, markers=True, ci=None, markevery=int(stepsize_cost*0.2), legend=False)

cos_graph.axline([0,0],[0.5,0.5], color='black', linewidth=0.3, linestyle='--')
cos_graph.axline([0,1],[0.5,0.5], color='black', linewidth=0.3, linestyle='--')

axes[2].set_title('Cost Curve')
axes[2].set(xlabel=None, ylabel='Normalized Expected Cost')
axes[2].set(ylim=(0,.5))
axes[2].set(xlim=(0,1))
axes[2].spines['left'].set_linewidth(0.6)
axes[2].spines['bottom'].set_linewidth(0.6)
axes[2].grid(which='major', linewidth=0.4)


sns.despine()

handles, labels = axes[0].get_legend_handles_labels()
fig.tight_layout()

plt.savefig("Dataset_full.svg")
plt.savefig("Dataset_full.jpg")
plt.show()
