import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os; R=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results') + '/'; F=R+'figures/'
C={'crnn':'#2a78d6','cnn14':'#eb6834','hubert':'#1baf7a','lgbm':'#eda100'}
NAME={'lgbm':'LightGBM','crnn':'CRNN (scratch)','cnn14':'CNN14 (AudioSet)','hubert':'HuBERT (speech SSL)'}
ORDER=['lgbm','crnn','cnn14','hubert']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#888','axes.labelcolor':'#222',
 'xtick.color':'#444','ytick.color':'#444','axes.grid':True,'grid.color':'#e6e6e6','grid.linewidth':0.6,'figure.facecolor':'#fcfcfb','axes.facecolor':'#fcfcfb','savefig.facecolor':'#fcfcfb'})

# 1 internal
ci=pd.read_csv(R+'internal_pooled_ci.csv')
fig,ax=plt.subplots(figsize=(7.4,3.8))
for i,m in enumerate(ORDER):
    g=ci[(ci.model==m)&(ci.protocol=='grouped')]
    mu,lo,hi=g.event_f1.mean(),g.event_f1_lo.mean(),g.event_f1_hi.mean()
    ax.errorbar(mu,i,xerr=[[mu-lo],[hi-mu]],fmt='o',color=C[m],ms=8,capsize=4,lw=2,zorder=3)
    ax.text(hi+0.004,i,f'{mu:.3f}',va='center',fontsize=9,color='#222')
    r=ci[(ci.model==m)&(ci.protocol=='random')].event_f1.iloc[0]
    o=ci[(ci.model==m)&(ci.protocol=='official')].event_f1.iloc[0]
    ax.plot(r,i+0.22,marker='o',mfc='none',mec=C[m],ms=7,mew=1.5,ls='none')
    ax.plot(o,i-0.22,marker='D',mfc='none',mec=C[m],ms=6,mew=1.5,ls='none')
ax.set_yticks(range(4)); ax.set_yticklabels([NAME[m] for m in ORDER]); ax.set_xlabel('Event F1 (Warsaw, 10 ms frames, one-to-one matching)')
ax.set_xlim(0.62,0.76); ax.grid(axis='y',visible=False)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],marker='o',color='#555',ls='none',ms=7,label='grouped 5-fold, mean of 3 seeds, 95% CI (primary)'),
 Line2D([],[],marker='o',mfc='none',mec='#555',ls='none',ms=7,label='random 5-fold (leaky), seed 0'),
 Line2D([],[],marker='D',mfc='none',mec='#555',ls='none',ms=6,label='official split (321 clips), seed 0')],loc='upper center',bbox_to_anchor=(0.45,-0.2),ncol=3,fontsize=7.2,frameon=False)
ax.set_title('Internal performance: CNN14 ≈ CRNN > LightGBM ≈ HuBERT',loc='left',fontsize=11)
fig.tight_layout(); fig.savefig(F+'fig1_internal_event_f1.png',dpi=200); plt.close()

# 2 external
e=pd.read_csv(R+'external.csv'); dense=['record_080524002_2','record_150524003_1']
e['set']=np.where(e.recording.isin(dense),'dense','sparse')
fig,axs=plt.subplots(1,2,figsize=(9.5,3.8),gridspec_kw={'width_ratios':[1,1.15]})
ax=axs[0]; w=0.36
for j,s in enumerate(['sparse','dense']):
    for i,m in enumerate(ORDER):
        v=e[(e.model==m)&(e.set==s)].groupby('seed').event_f1.mean()
        x=i+(j-0.5)*w
        ax.bar(x,v.mean(),w*0.92,color=C[m],alpha=1 if s=='sparse' else 0.35,edgecolor='none')
        ax.errorbar(x,v.mean(),yerr=v.std(),color='#333',lw=1,capsize=2)
        ax.text(x,v.mean()+0.03,f'{v.mean():.2f}',ha='center',fontsize=7.5)
ax.set_xticks(range(4)); ax.set_xticklabels(['LightGBM','CRNN','CNN14','HuBERT']); ax.set_ylim(0,1); ax.set_ylabel('Event F1 on Oldenburg (zero-shot)')
ax.grid(axis='x',visible=False)
ax.set_title('Solid: 5 recordings dominated by single bursts\nFaded: 2 dominated by long MB/CRS/HS sounds (all models score ~0.79)',fontsize=9,loc='left')
ax=axs[1]
sp=e[e.set=='sparse'].groupby(['model','seed'])[['event_recall','fa_per_min','event_f1']].mean().reset_index()
for m in ORDER:
    s=sp[sp.model==m]; ax.scatter(s.fa_per_min,s.event_recall,color=C[m],s=45,edgecolor='#fcfcfb',lw=1.5,zorder=3)
    ax.annotate(NAME[m].split(' ')[0],(s.fa_per_min.mean(),s.event_recall.mean()),xytext=(-10,-16) if m=='cnn14' else (8,-3),textcoords='offset points',fontsize=8.5)
ax.set_xlabel('False alarms per minute (5 single-burst-dominated recordings)'); ax.set_ylabel('Event recall (one-to-one)'); ax.set_ylim(0.2,1.0); ax.set_xlim(-2,45)
ax.set_title('Pretrained models keep recall at the Warsaw threshold,\nbut pay with more false alarms (one dot per seed)',fontsize=9,loc='left')
fig.suptitle('External validation (Oldenburg wearable, 4 subjects, models trained on Warsaw only)',x=0.01,ha='left',fontsize=11)
fig.tight_layout(); fig.savefig(F+'fig2_external.png',dpi=200); plt.close()

# 3 LOSO
l=pd.read_csv(R+'loso.csv'); z=e[e.seed==0]
fig,axs=plt.subplots(1,2,figsize=(9,3.4))
VAR=[('zero-shot\n(Warsaw only)',None),('3 Oldenburg\nsubjects only','old_only'),('Warsaw +\n3 Oldenburg','warsaw+old')]
for k,met in enumerate(['frame_auroc','event_f1']):
    ax=axs[k]
    for mi,m in enumerate(['crnn','cnn14']):
        vals=[]
        for lab,v in VAR:
            d=z[z.model==m] if v is None else l[(l.model==m)&(l.variant==v)]
            vals.append(d[met].mean())
        xs=np.arange(3)+(mi-0.5)*0.12
        ax.plot(xs,vals,'-o',color=C[m],lw=2,ms=7,label=NAME[m])
        for x,v in zip(xs,vals): ax.text(x+(0.09 if mi else -0.09),v,f'{v:.3f}',fontsize=7.5,ha='left' if mi else 'right',va='center')
    ax.set_xticks(range(3)); ax.set_xticklabels([a for a,_ in VAR],fontsize=8.5); ax.grid(axis='x',visible=False); ax.set_xlim(-0.5,2.5)
    ax.set_ylabel({'frame_auroc':'Frame AUROC (threshold-free)','event_f1':'Event F1 (tuned threshold)'}[met])
axs[0].legend(frameon=False,fontsize=8,loc='lower right')
fig.suptitle('Leave-one-subject-out on Oldenburg (7 recordings, 1 seed): adding target-device data improves ranking (AUROC)\nbut not event F1 — suggesting the decision threshold, more than the representation, is what fails to transfer',x=0.01,ha='left',fontsize=9.5)
fig.tight_layout(); fig.savefig(F+'fig3_loso_adaptation.png',dpi=200); plt.close()

# 4 duration
d=pd.read_csv(R+'recall_by_duration.csv'); bins=['<15','15-25','25-40','40-80','>80']
fig,ax=plt.subplots(figsize=(6,3.4))
for m in ORDER:
    s=d[d.model==m].set_index('dur_bin').loc[bins]
    ax.plot(range(5),s['mean'],'-o',color=C[m],lw=2,ms=6,label=NAME[m])
n=d[d.model=='crnn'].set_index('dur_bin').loc[bins]['size']
ax.set_xticks(range(5)); ax.set_xticklabels([f'{b} ms\n(n={k//3})' for b,k in zip(bins,n)],fontsize=8)
ax.set_ylabel('Event recall (Warsaw, grouped CV)'); ax.set_ylim(0.4,0.9); ax.legend(frameon=False,fontsize=8); ax.grid(axis='x',visible=False)
ax.set_title('All models miss short (<15 ms) events most',loc='left',fontsize=11)
fig.tight_layout(); fig.savefig(F+'fig4_recall_by_duration.png',dpi=200); plt.close()
print('ok')
