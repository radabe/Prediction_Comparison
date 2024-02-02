#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# pip install scikit-posthocs


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import copy
# import scikit_posthocs as sp
from scipy import stats


# In[ ]:


dpi=100
stepsize_roc=200
stepsize_pr=200
stepsize_conf=100

max_neg = 7037
max_pos = 2029
tot_cases = 9066


# In[ ]:


XGB_index_data = pd.read_csv('Data/Full curves/curve_xgb_index_calibrated.csv', index_col=None, header=None)
XGB_agg_data = pd.read_csv('Data/Full curves/curve_xgb_agg_calibrated.csv', index_col=None, header=None)
RFC_index_data = pd.read_csv('Data/Full curves/curve_rfc_index_calibrated.csv', index_col=None, header=None)
RFC_agg_data = pd.read_csv('Data/Full curves/curve_rfc_agg_calibrated.csv', index_col=None, header=None)
MM_data = pd.read_csv('Data/Full curves/curve_mm.csv', index_col=None, header=None)
LSTM_data = pd.read_csv('Data/Full curves/curve_lstm.csv', index_col=None, header=None)


# In[ ]:


# XGB_index_data = pd.read_csv('Data/Full curves/curve_xgb_index.csv', index_col=None, header=None)
# XGB_agg_data = pd.read_csv('Data/Full curves/curve_xgb_agg.csv', index_col=None, header=None)
# RFC_index_data = pd.read_csv('Data/Full curves/curve_rfc_index.csv', index_col=None, header=None)
# RFC_agg_data = pd.read_csv('Data/Full curves/curve_rfc_agg.csv', index_col=None, header=None)
# MM_data = pd.read_csv('Data/Full curves/curve_mm.csv', index_col=None, header=None)
# LSTM_data = pd.read_csv('Data/Full curves/curve_lstm.csv', index_col=None, header=None)


# In[ ]:


XGB_index_data.columns = ['run_%s'%(i) for i in range (1,21)]
XGB_agg_data.columns= ['run_%s'%(i) for i in range (1,21)]
RFC_index_data.columns= ['run_%s'%(i) for i in range (1,21)]
RFC_agg_data.columns= ['run_%s'%(i) for i in range (1,21)]
MM_data.columns= ['run_%s'%(i) for i in range (1,21)]
LSTM_data.columns= ['run_%s'%(i) for i in range (1,21)]


# In[ ]:


XGB_index_data_clean=pd.DataFrame(data=None, columns=XGB_index_data.columns, index=XGB_index_data.index)
XGB_agg_data_clean=pd.DataFrame(data=None, columns=XGB_agg_data.columns, index=XGB_agg_data.index)
RFC_index_data_clean=pd.DataFrame(data=None, columns=RFC_index_data.columns, index=RFC_index_data.index)
RFC_agg_data_clean=pd.DataFrame(data=None, columns=RFC_agg_data.columns, index=RFC_agg_data.index)
MM_data_clean=pd.DataFrame(data=None, columns=MM_data.columns, index=MM_data.index)
LSTM_data_clean=pd.DataFrame(data=None, columns=LSTM_data.columns, index=LSTM_data.index)


# In[ ]:


data_list=[XGB_index_data,XGB_agg_data,RFC_index_data,RFC_agg_data,MM_data,LSTM_data]
data_clean_list=[XGB_index_data_clean,XGB_agg_data_clean,RFC_index_data_clean,RFC_agg_data_clean,MM_data_clean,LSTM_data_clean]
clean_table=pd.DataFrame(data=None, columns=XGB_index_data_clean.columns, index=XGB_index_data_clean.index)


# In[ ]:


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


# In[ ]:


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


# In[ ]:


# sns.set(font_scale=1.2,rc={"lines.linewidth": 1, 'figure.figsize':(10,10), 'figure.dpi':dpi, 'xtick.top' : False, 'markers.fillstyle': 'full', "lines.markersize":10  })
# sns.set_style("white")
# s=sns.lineplot(x='fpr', y='tpr', data=All_chart, hue='Algorithm', hue_order=algorithm_list, style='Algorithm', 
#              dashes=False, markers=True, ci=1, markevery=20)

# s.set_ylim(0, 1.05)
# plt.title('Receiver-Operator Curve', fontsize=16)
# plt.ylabel('True Positive Rate', fontsize=12)
# plt.xlabel(None, fontsize=12)


# In[ ]:


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


# In[ ]:


# sns.set(font_scale=1.2,rc={"lines.linewidth": 3, 'figure.figsize':(10,10), 'figure.dpi':dpi, 'xtick.top' : False, 'markers.fillstyle': 'full', "lines.markersize":10  })
# sns.set_style("white")
# s=sns.lineplot(x='recall', y='precision', data=All_chart_pr, hue='Algorithm', hue_order=algorithm_list, style='Algorithm', 
#              dashes=False, markers=True, ci=1, markevery=stepsize_pr/10)

# s.set_ylim(0, 1)
# plt.title('Precision-Recall Curve', fontsize=16)
# plt.ylabel('Precision', fontsize=12)
# plt.xlabel(None, fontsize=16)


# In[ ]:


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


# In[ ]:


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
                
#         data.iloc[8,r]=np.max(data_pr.iloc[(0+(r*stepsize_pr)):(stepsize_pr+(r*stepsize_pr)),4])
#         data.iloc[10,r]=np.max(data_pr.iloc[(0+(r*stepsize_pr)):(stepsize_pr+(r*stepsize_pr)),5])
#         data.iloc[12,r]=np.max(data_pr.iloc[(0+(r*stepsize_pr)):(stepsize_pr+(r*stepsize_pr)),6])
#         data.iloc[14,r]=np.max(conf.iloc[(0+(r*stepsize_conf)):(stepsize_conf+(r*stepsize_conf)),13])
#         data.iloc[16,r]=np.max(conf.iloc[(0+(r*stepsize_conf)):(stepsize_conf+(r*stepsize_conf)),14])
#         data.iloc[9,r]=np.mean(data_pr[data_pr['F1_score']==data.iloc[8,r]]['thres_pr'])
#         data.iloc[11,r]=np.mean(data_pr[data_pr['F2_score']==data.iloc[10,r]]['thres_pr'])
#         data.iloc[13,r]=np.mean(data_pr[data_pr['F0.5_score']==data.iloc[12,r]]['thres_pr'])
#         data.iloc[15,r]=np.mean(conf[conf['MCC']==data.iloc[16,r]]['thres'])
#         data.iloc[17,r]=np.mean(conf[conf['Cost']==data.iloc[16,r]]['thres'])

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


# In[ ]:


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


# In[ ]:


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
#     score_df.loc[algorithm, 'F0.5@ threshold']=np.mean(data.loc[13].astype(float))
    
    score_df.loc[algorithm, 'F1-Score']=np.mean(data.loc[8].astype(float))
#     score_df.loc[algorithm, 'F1@ threshold']=np.mean(data.loc[9].astype(float))
    score_df.loc[algorithm, 'precision']=np.mean(data.loc[9].astype(float))
    score_df.loc[algorithm, 'F2-Score']=np.mean(data.loc[10].astype(float))
#     score_df.loc[algorithm, 'F2@ threshold']=np.mean(data.loc[11].astype(float))

    
    score_df.loc[algorithm, 'MCC']=np.mean(data.loc[14].astype(float))
#     score_df.loc[algorithm, 'MCC@ threshold']=np.mean(data.loc[15].astype(float))
    
    score_df.loc[algorithm, 'Monetary Cost']=np.mean(data.loc[16].astype(float))
    score_df.loc[algorithm, '@ threshold C']=np.mean(data.loc[17].astype(float))
    


# In[ ]:


score_df


# In[ ]:


# sns.set(font_scale=1.2,rc={"lines.linewidth": 3, 'figure.figsize':(10,10), 'figure.dpi':50, 'xtick.top' : False, 'markers.fillstyle': 'full', "lines.markersize":10  })
# sns.set_style("white")

# s=sns.lineplot(x='thres', y='Expected Normalized Benefit', data=All_conf, hue='Algorithm', hue_order=algorithm_list, style='Algorithm', 
#              dashes=False, markers=True, ci=1, markevery=10, legend=False)

# s.set_ylim(-0, 1)
# plt.title('Normalized Benefit Curve', fontsize=16)
# plt.ylabel('Benefit', fontsize=12)
# plt.xlabel(None, fontsize=12)


# In[ ]:


# sns.set(font_scale=0.8,rc={'figure.figsize':((7.16/3),(7.16/3)), "lines.linewidth": 0.5, 'figure.dpi':600, 'xtick.top' : False, 'markers.fillstyle': 'full', "lines.markersize":4,'lines.markeredgewidth':0 })
# sns.set_style("white")

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


# In[ ]:


cost_ratio


# In[ ]:


sns.set(font_scale=1.2,rc={"lines.linewidth": 1, 'figure.dpi':600, 'xtick.top' : False, 'markers.fillstyle': 'full', "lines.markersize":4,'lines.markeredgewidth':0  })
sns.set_style("whitegrid")

s=sns.lineplot(data=cost_curve_df.T.set_index('probabilitycost'), hue_order=algorithm_list,
             dashes=False, markers=True, ci=1, markevery=10, legend=False)

s.axline([0,0],[0.5,0.5], color='black', linewidth=0.5, linestyle='--')
s.axline([0,1],[0.5,0.5], color='black', linewidth=0.5, linestyle='--')




# plot parameters
# plt.xlim([-1.0, 1.0])
# plt.ylim([-1.0, 1.0])
plt.xlabel("Probability Cost Function")
plt.ylabel("Normalized Expected Cost")
plt.title("Cost curve")
# plt.legend(loc="lower right")

plt.show()


# In[ ]:


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
# fig.legend(handles, labels,loc='upper center', bbox_to_anchor=(0.5, 0), ncol=6, markerscale=0.8, title='Algorithm', frameon=False)
# axes[0].legend().set_visible(False)
fig.tight_layout()

plt.savefig("Dataset_full.svg")
plt.savefig("Dataset_full.jpg")
plt.show()


# In[ ]:


# for s in range(7):
    
#     num_alg=len(algorithm_list)
#     num_runs=len(XGB_index_data.columns)

#     score_list=[3,7,8,10,12,14,16]
#     score_name=['AUC-ROC','AUC-PR','F1-Score','F2-Score','F0.5-Score','MCC','Monetary Benefit']


#     data_list=[XGB_index_data,XGB_agg_data,RFC_index_data,RFC_agg_data,MM_data,LSTM_data]
#     friedmann_df=pd.DataFrame(data=None, index=algorithm_list, columns=XGB_index_data.columns)
#     i=0
#     for data in data_list:
#         friedmann_df.iloc[i]=np.around(data.loc[score_list[s]].astype(float), decimals=4)
#         i+=1


#     friedmann_rank=pd.DataFrame(data=None, index=algorithm_list)

#     for i in range (1,21):
#         friedmann_temp=friedmann_df.loc[:,'run_%s'%i].sort_values( kind='mergesort', ascending=False).reset_index()
#         friedmann_temp['rank_run_%s'%i]=range(1,7)
#         friedmann_rank=friedmann_rank.join(friedmann_temp[['index','rank_run_%s'%i]].set_index('index'))


#     friedmann_rank_average=pd.DataFrame(data=None, index=algorithm_list, columns=['average_rank'])

#     for alg in algorithm_list:
#         friedmann_rank_average.loc[alg,'average_rank']=np.mean(friedmann_rank.loc[alg])

#     nemenyi = np.array([friedmann_df.iloc[0],friedmann_df.iloc[1],friedmann_df.iloc[2],friedmann_df.iloc[3],friedmann_df.iloc[4],friedmann_df.iloc[5]])

#     friedmann_stats=stats.friedmanchisquare(friedmann_df.iloc[0],friedmann_df.iloc[1],friedmann_df.iloc[2],friedmann_df.iloc[3],friedmann_df.iloc[4],friedmann_df.iloc[5])
#     CD=2.850*(((num_alg*(num_alg+1))/(6*num_runs))**(1/2))

#     print(friedmann_stats)
#     print(CD)

#     sns.set(font_scale=1.2,rc={"lines.linewidth": 2, 'figure.figsize':(10,10), 'figure.dpi':100, 'xtick.top' : False, 'markers.fillstyle': 'full', "lines.markersize":9  })
#     sns.set_style("white")

#     sorted_friedmann=friedmann_rank_average.sort_values(by='average_rank').reset_index()

#     cd = CD


#     rank_1_value=sorted_friedmann.iloc[0,1]
#     rank_2_value=sorted_friedmann.iloc[1,1]
#     rank_3_value=sorted_friedmann.iloc[2,1]
#     rank_4_value=sorted_friedmann.iloc[3,1]
#     rank_5_value=sorted_friedmann.iloc[4,1]
#     rank_6_value=sorted_friedmann.iloc[5,1]



#     rank_1=sorted_friedmann.iloc[0,0]
#     rank_2=sorted_friedmann.iloc[1,0]
#     rank_3=sorted_friedmann.iloc[2,0]
#     rank_4=sorted_friedmann.iloc[3,0]
#     rank_5=sorted_friedmann.iloc[4,0]
#     rank_6=sorted_friedmann.iloc[5,0]

#     value_list=[rank_1_value,rank_2_value,rank_3_value,rank_4_value,rank_5_value,rank_6_value]
#     rank_list=[rank_1,rank_2,rank_3,rank_4,rank_5,rank_6]

#     limits=(1,6)

#     fig, ax = plt.subplots(figsize=(10,5))
#     plt.subplots_adjust(left=0.2, right=0.8)


#     ax.set_xlim(limits)
#     ax.set_ylim(0,1)
#     ax.spines['top'].set_position(('axes', 0.6))
#     #ax.xaxis.tick_top()
#     ax.xaxis.set_ticks_position('top')
#     ax.yaxis.set_visible(False)
#     for pos in ["bottom", "left", "right"]:
#         ax.spines[pos].set_visible(False)

#     ax.plot([limits[0],limits[0]+cd], [.9,.9], color="k")
#     ax.plot([limits[0],limits[0]], [.9-0.03,.9+0.03], color="k")
#     ax.plot([limits[0]+cd,limits[0]+cd], [.9-0.03,.9+0.03], color="k") 
#     ax.text(limits[0]+cd/2., 0.92, "CD", ha="center", va="bottom") 

#     plt.title(score_name[s])


#     bbox_props = dict(boxstyle="square,pad=0.5", fc="w", ec="k", lw=0.0)
#     arrowprops=dict(arrowstyle="-",connectionstyle="angle,angleA=0,angleB=90", color='black')
#     kw = dict(xycoords='data',textcoords="axes fraction",
#               arrowprops=arrowprops, bbox=bbox_props, va="center")
#     ax.annotate(rank_1, xy=(rank_1_value, 0.6), xytext=(0,0.35),ha="right",  **kw)
#     ax.annotate(rank_2, xy=(rank_2_value, 0.6), xytext=(0,0.25),ha="right",  **kw)
#     ax.annotate(rank_3, xy=(rank_3_value, 0.6), xytext=(0,0.15),ha="right",  **kw)
#     ax.annotate(rank_4, xy=(rank_4_value, 0.6), xytext=(1.,0.15),ha="left",  **kw)
#     ax.annotate(rank_5, xy=(rank_5_value, 0.6), xytext=(1.,0.25),ha="left",  **kw)
#     ax.annotate(rank_6, xy=(rank_6_value, 0.6), xytext=(1.,0.35),ha="left",  **kw)


#     k=0
#     row=0
#     for i in range(6):
#         value_idx=np.where((np.array(value_list[i:])<(value_list[i]+CD)))
#         if ((len(value_idx[0]) > 0) & (k==0)):
#             k=value_idx[0][-1]
#             alg_1=value_list[i]
#             alg_2=value_list[i+k]

#             ax.plot([alg_1,alg_2],[0.55-(0.05*row),0.55-(0.05*row)], color="k", lw=3)
#             row+=1
#         k-=1
#         if k<0:
#             k=0
            
#     plt.show()

        


# In[ ]:


# num_alg=len(algorithm_list)
# num_runs=60

# score_list=[3,7,8,10,12,14,16]
# score_name=['AUC-ROC','AUC-PR','Monetary Benefit']


# data_list=[XGB_index_data,XGB_agg_data,RFC_index_data,RFC_agg_data,MM_data,LSTM_data]
# friedmann_df=pd.DataFrame(data=None, index=algorithm_list, columns=range(60))
# i=0
# for data in data_list:
#     friedmann_df.iloc[i,0:20]=np.around(data.loc[score_list[0]].astype(float), decimals=4)
#     friedmann_df.iloc[i,20:40]=np.around(data.loc[score_list[1]].astype(float), decimals=4)
#     friedmann_df.iloc[i,40:60]=np.around(data.loc[score_list[6]].astype(float), decimals=4)
#     i+=1




# friedmann_rank=pd.DataFrame(data=None, index=algorithm_list)

# for i in range (60):
#     friedmann_temp=friedmann_df.iloc[:,i].sort_values( kind='mergesort', ascending=False).reset_index()
#     friedmann_temp['rank_score_%s'%i]=range(1,7)
#     friedmann_rank=friedmann_rank.join(friedmann_temp[['index','rank_score_%s'%i]].set_index('index'))


# friedmann_rank_average=pd.DataFrame(data=None, index=algorithm_list, columns=['average_rank'])

# for alg in algorithm_list:
#     friedmann_rank_average.loc[alg,'average_rank']=np.mean(friedmann_rank.loc[alg])

# nemenyi = np.array([friedmann_df.iloc[0],friedmann_df.iloc[1],friedmann_df.iloc[2],friedmann_df.iloc[3],friedmann_df.iloc[4],friedmann_df.iloc[5]])

# friedmann_stats=stats.friedmanchisquare(friedmann_df.iloc[0],friedmann_df.iloc[1],friedmann_df.iloc[2],friedmann_df.iloc[3],friedmann_df.iloc[4],friedmann_df.iloc[5])
# CD=2.850*(((num_alg*(num_alg+1))/(6*num_runs))**(1/2))

# print(friedmann_stats)
# print(CD)

# sns.set(font_scale=1.2,rc={"lines.linewidth": 2, 'figure.figsize':(10,10), 'figure.dpi':100, 'xtick.top' : False, 'markers.fillstyle': 'full', "lines.markersize":9  })
# sns.set_style("white")

# sorted_friedmann=friedmann_rank_average.sort_values(by='average_rank').reset_index()

# cd = CD


# rank_1_value=sorted_friedmann.iloc[0,1]
# rank_2_value=sorted_friedmann.iloc[1,1]
# rank_3_value=sorted_friedmann.iloc[2,1]
# rank_4_value=sorted_friedmann.iloc[3,1]
# rank_5_value=sorted_friedmann.iloc[4,1]
# rank_6_value=sorted_friedmann.iloc[5,1]



# rank_1=sorted_friedmann.iloc[0,0]
# rank_2=sorted_friedmann.iloc[1,0]
# rank_3=sorted_friedmann.iloc[2,0]
# rank_4=sorted_friedmann.iloc[3,0]
# rank_5=sorted_friedmann.iloc[4,0]
# rank_6=sorted_friedmann.iloc[5,0]

# value_list=[rank_1_value,rank_2_value,rank_3_value,rank_4_value,rank_5_value,rank_6_value]
# rank_list=[rank_1,rank_2,rank_3,rank_4,rank_5,rank_6]

# limits=(1,6)

# fig, ax = plt.subplots(figsize=(10,5))
# plt.subplots_adjust(left=0.2, right=0.8)


# ax.set_xlim(limits)
# ax.set_ylim(0,1)
# ax.spines['top'].set_position(('axes', 0.6))
# #ax.xaxis.tick_top()
# ax.xaxis.set_ticks_position('top')
# ax.yaxis.set_visible(False)
# for pos in ["bottom", "left", "right"]:
#     ax.spines[pos].set_visible(False)

# ax.plot([limits[0],limits[0]+cd], [.8,.8], color="k")
# ax.plot([limits[0],limits[0]], [.8-0.03,.8+0.03], color="k")
# ax.plot([limits[0]+cd,limits[0]+cd], [.8-0.03,.8+0.03], color="k") 
# ax.text(limits[0]+cd/2., 0.82, "CD", ha="center", va="bottom") 



# bbox_props = dict(boxstyle="square,pad=0.5", fc="w", ec="k", lw=0.0)
# arrowprops=dict(arrowstyle="-",connectionstyle="angle,angleA=0,angleB=90", color='black')
# kw = dict(xycoords='data',textcoords="axes fraction",
#           arrowprops=arrowprops, bbox=bbox_props, va="center")
# ax.annotate(rank_1, xy=(rank_1_value, 0.6), xytext=(0,0.35),ha="right",  **kw)
# ax.annotate(rank_2, xy=(rank_2_value, 0.6), xytext=(0,0.25),ha="right",  **kw)
# ax.annotate(rank_3, xy=(rank_3_value, 0.6), xytext=(0,0.15),ha="right",  **kw)
# ax.annotate(rank_4, xy=(rank_4_value, 0.6), xytext=(1.,0.15),ha="left",  **kw)
# ax.annotate(rank_5, xy=(rank_5_value, 0.6), xytext=(1.,0.25),ha="left",  **kw)
# ax.annotate(rank_6, xy=(rank_6_value, 0.6), xytext=(1.,0.35),ha="left",  **kw)


# k=0
# row=0
# for i in range(6):
#     value_idx=np.where((np.array(value_list[i:])<(value_list[i]+CD)))
#     if ((len(value_idx[0]) > 0) & (k==0)):
#         k=value_idx[0][-1]
#         alg_1=value_list[i]
#         alg_2=value_list[i+k]

#         ax.plot([alg_1,alg_2],[0.55-(0.05*row),0.55-(0.05*row)], color="k", lw=3)
#         row+=1
#     k-=1
#     if k<0:
#         k=0

# plt.show()



# In[ ]:





# In[ ]:


cost_curve_all


# In[ ]:


confidence_curve_all=pd.DataFrame(data=None, index=confidence_curve.index, columns=['XGB_idx -> XGB_agg','XGB_idx -> RFC_idx','XGB_idx -> RFC_agg','XGB_idx -> MM','XGB_idx -> LSTM','thres'])


# In[ ]:


confidence_curve_all


# In[ ]:


for alg in range(1,6):
    
    confidence_curve=pd.DataFrame(data=None, index=['value','thres'], columns=cost_curve_all.columns)
    for i in range(20):

        confidence_curve.iloc[0,i]=cost_curve_all.iloc[0,i]-cost_curve_all.iloc[alg,i]

    confidence_curve=confidence_curve.T.explode('value').reset_index()
    confidence_curve_all.iloc[:,alg-1]=confidence_curve.loc[:,'value']


for i in range(20):
    confidence_curve_all.iloc[i*101:101+101*i,5]=pc[0]


# In[ ]:


thres_list=pc[0]


# In[ ]:


sns.set(font_scale=1.2,rc={"lines.linewidth": 1, 'figure.dpi':600, 'xtick.top' : False, 'markers.fillstyle': 'full', "lines.markersize":4,'lines.markeredgewidth':0  })
sns.set_style("whitegrid")

s=sns.lineplot(x='thres', y='XGB_idx -> XGB_agg',data=confidence_curve_all, dashes=False, markers=True, ci=95, markevery=None, legend=True)
sns.lineplot(x='thres', y='XGB_idx -> RFC_idx',data=confidence_curve_all, dashes=False, markers=True, ci=95, markevery=None, legend=True)
sns.lineplot(x='thres', y='XGB_idx -> RFC_agg',data=confidence_curve_all, dashes=False, markers=True, ci=95, markevery=None, legend=True)
sns.lineplot(x='thres', y='XGB_idx -> MM',data=confidence_curve_all, dashes=False, markers=True, ci=95, markevery=None, legend=True)
sns.lineplot(x='thres', y='XGB_idx -> LSTM',data=confidence_curve_all, dashes=False, markers=True, ci=95, markevery=None, legend=True)

# s.axline([0,0],[0.5,0.5], color='black', linewidth=0.5, linestyle='--')
# s.axline([0,1],[0.5,0.5], color='black', linewidth=0.5, linestyle='--')
# s.axvline(skew, color='black', linewidth=0.4, linestyle='--')

s.legend(labels=['XGB_idx -> XGB_agg','CI:95','XGB_idx -> RFC_idx','CI:95','XGB_idx -> RFC_agg','CI:95','XGB_idx -> MM','CI:95','XGB_idx -> LSTM','CI:95'])


# plot parameters
plt.xlim([-0, 1])
plt.ylim([-0.05, 0.05])
plt.xlabel("Probability Cost Function")
plt.ylabel("Normalized Expected Cost")
plt.title("Cost curve")
# plt.legend(loc="lower right")

plt.show()


# In[ ]:


c=cost_fn/(cost_fn+cost_fp)
p=max_pos/tot_cases
(c*p)/((c*p)+(1-c)*(1-p))


# In[ ]:





# In[ ]:





# In[ ]:


c


# In[ ]:




