"""Phân tích kết quả đã khóa. Module này không huấn luyện hoặc thay ngưỡng."""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve
from .experiment import ROOT,load_json,dump,load_data

def main():
    sns.set_theme(style='whitegrid',font='DejaVu Sans',font_scale=1.05)
    results=ROOT/'results';figures=results/'figures';figures.mkdir(exist_ok=True)
    def save(name):
        plt.tight_layout();plt.savefig(figures/name,dpi=160,bbox_inches='tight');plt.close()
    best=pd.read_csv(results/'best_per_model.csv').sort_values('AP_mean')
    plt.figure(figsize=(8,4.2));plt.barh(best.model,best.AP_mean,xerr=best.AP_std,color=['#9baebe','#347898','#147b78','#16475f'])
    plt.xlim(0,1);plt.xlabel('Average Precision trung bình ± SD của 5 fold');plt.title('So sánh mô hình · chỉ trên tập train');save('01_cv_ap.png')
    abl=pd.read_csv(results/'ablation_summary.csv').sort_values('AP_mean')
    plt.figure(figsize=(8,4.2));plt.barh(abl.variant,abl.AP_mean,xerr=abl.AP_std,color='#147b78');plt.xlim(0,1);plt.xlabel('Average Precision trung bình ± SD');plt.title('Ablation · Logistic Regression, cùng siêu tham số');save('02_ablation.png')
    pred=pd.read_csv(results/'test_predictions.csv');m=load_json(results/'test_metrics.json')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    p,r,_=precision_recall_curve(pred.fraudulent,pred.score);axes[0].plot(r,p,color='#147b78');axes[0].axhline(pred.fraudulent.mean(),ls='--',color='#ab7139',label='Tỷ lệ tin giả')
    axes[0].scatter(m['recall'],m['precision'],color='#bb5134',label='Ngưỡng đã khóa');axes[0].set(xlabel='Recall',ylabel='Precision',title=f"PR · AP = {m['AP']:.3f}");axes[0].legend(fontsize=8)
    fpr,tpr,_=roc_curve(pred.fraudulent,pred.score);axes[1].plot(fpr,tpr,color='#147b78');axes[1].plot([0,1],[0,1],'--',color='#a6afb5');axes[1].set(xlabel='False Positive Rate',ylabel='True Positive Rate',title=f"ROC · AUC = {m['ROC_AUC']:.3f}");save('03_test_curves.png')
    plt.figure(figsize=(6,4.5));sns.heatmap([[m['TN'],m['FP']],[m['FN'],m['TP']]],annot=True,fmt='d',cmap='Blues',cbar=False,xticklabels=['Không cảnh báo','Cảnh báo'],yticklabels=['Thật','Giả']);plt.xlabel('Dự đoán');plt.ylabel('Nhãn thực tế');plt.title('Ma trận nhầm lẫn · test độc lập');save('04_confusion.png')
    curve=pd.read_csv(results/'validation_threshold_curve.csv');v=load_json(results/'validation_metrics.json')
    plt.figure(figsize=(8,4));plt.plot(curve.threshold,curve.precision,label='Precision');plt.plot(curve.threshold,curve.recall,label='Recall');plt.axhline(.8,ls='--',color='#9b6038',label='Mục tiêu precision 0,80');plt.axvline(v['threshold'],ls=':',color='#293f4a',label='Ngưỡng chọn');plt.xlabel('Ngưỡng điểm mô hình');plt.ylabel('Giá trị');plt.title('Chọn ngưỡng chỉ trên validation');plt.legend();save('05_validation_threshold.png')
    parts,groups,folds=load_data()
    frame=parts['test'].merge(pred,on=['job_id','fraudulent'],validate='one_to_one')
    frame['error_type']=np.select([(frame.fraudulent==1)&(frame.predicted==0),(frame.fraudulent==0)&(frame.predicted==1)],['FN','FP'],default='correct')
    frame['description_words']=frame.description.fillna('').str.split().str.len()
    rows=[]
    for field in ['has_company_logo','has_questions','telecommuting']:
        for value,sub in frame.groupby(field,dropna=False):
            y=sub.fraudulent;p=sub.predicted;tp=int(((y==1)&(p==1)).sum());fn=int(((y==1)&(p==0)).sum());fp=int(((y==0)&(p==1)).sum())
            rows.append({'field':field,'value':str(value),'n':len(sub),'fake':int(y.sum()),'TP':tp,'FP':fp,'FN':fn,'recall':tp/(tp+fn) if tp+fn else None,'precision':tp/(tp+fp) if tp+fp else None})
    pd.DataFrame(rows).to_csv(results/'error_slices.csv',index=False,encoding='utf-8-sig')
    cases=[]
    for kind in ['FN','FP']:
        # Chọn lỗi tự tin nhất để giải thích giới hạn, không để sửa mô hình theo test.
        sub=frame.loc[frame.error_type.eq(kind)].sort_values('score',ascending=kind=='FN').head(3)
        for _,row in sub.iterrows():
            cases.append({'job_id':int(row.job_id),'error_type':kind,'actual':int(row.fraudulent),'predicted':int(row.predicted),'score':float(row.score),'title':str(row.title),'has_company_logo':int(row.has_company_logo),'description_excerpt':str(row.description)[:450]})
    dump(results/'error_cases.json',cases)
    bundle=joblib.load(ROOT/'models/fake_job_detector.joblib');model=bundle['pipeline'].named_steps['classifier']
    names=bundle['pipeline'].named_steps['preprocess'].named_steps['columns'].get_feature_names_out()
    importance=model.coef_[0] if hasattr(model,'coef_') else model.feature_importances_
    coef=pd.DataFrame({'feature':names,'weight':importance}).sort_values('weight')
    coef.to_csv(results/'feature_weights.csv',index=False,encoding='utf-8-sig')
    top=pd.concat([coef.head(10),coef.tail(10)]) if hasattr(model,'coef_') else coef.tail(20)
    plt.figure(figsize=(9,7));plt.barh(top.feature.str.replace('text__','',regex=False),top.weight,color=['#bb7850' if x<0 else '#147b78' for x in top.weight]);plt.xlabel('Hệ số (không phải quan hệ nhân quả)');plt.title('Các đặc trưng có hệ số lớn theo độ lớn');save('06_feature_weights.png')
    # Bốn kiểu kết quả validation: minh họa cả đúng và sai, không coi là test mới.
    examples=[]
    vp=pd.read_csv(results/'validation_predictions.csv').set_index('job_id')
    for label,predicted in [(1,1),(0,0),(0,1),(1,0)]:
        ids=vp.index[(vp.fraudulent==label)&(vp.predicted==predicted)]
        if len(ids)==0:continue
        sample=parts['validation'].set_index('job_id',drop=False).loc[ids[0]]
        fields={k:(None if pd.isna(x) else x.item() if hasattr(x,'item') else x) for k,x in sample.drop(['job_id','fraudulent']).items()}
        examples.append({'job_id':int(sample.job_id),'actual_label':int(label),'fields':fields})
    dump(ROOT/'evidence/demo_examples.json',examples)
    warnings=pd.read_csv(results/'cv_results.csv').warnings.fillna('')
    dump(results/'analysis_summary.json',{'cv_fits':125,'model_search_fits':120,'ablation_fits':20,'cv_warning_count':int(warnings.ne('').sum()),'test_errors':int(frame.error_type.ne('correct').sum()),'test_error_interpretation':'Post-hoc description only; no refit or threshold change','train_size':len(parts['train']),'validation_size':len(parts['validation']),'test_size':len(parts['test'])})
    print('Analysis figures, cases, slices and demo examples saved.')

if __name__=='__main__':main()
