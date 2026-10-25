"""Tune fusion on validation only; evaluate CNN, rules, crops and fusion on the same test IDs."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from common import ROOT,CLASSES,GestureDataset,manifest_digest,write_json
from inference import load_model,fuse
from metrics import classification_metrics
from train import choose_device

def predict(model,config,manifest,split,device,batch_size):
    dataset=GestureDataset(manifest,split,config['view'],config['size'],config['gray'])
    scores=[]
    with torch.no_grad():
        for images,_ in tqdm(DataLoader(dataset,batch_size=batch_size),desc=split+' CNN'):
            scores.extend(model(images.to(device)).float().softmax(1).cpu().tolist())
    return dataset.records,np.asarray(scores)

def fusion_scores(rows,scores,threshold):
    result=[fuse(score,row['rule'],row['detected'],threshold) for row,score in zip(rows,scores)]
    return np.array([r[0] for r in result]),np.array([r[1] for r in result])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True,help='Model trained with --view mp')
    parser.add_argument('--raw-checkpoint',type=Path,help='Same architecture/settings trained with --view raw')
    parser.add_argument('--manifest',type=Path,default=ROOT/'prepared'/'manifest.jsonl')
    parser.add_argument('--output',type=Path,default=ROOT/'reports'/'pipeline')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--device',default='auto')
    args=parser.parse_args(); torch.set_num_threads(4); device=choose_device(args.device)
    model,config=load_model(args.checkpoint,device)
    digest=manifest_digest(args.manifest)
    if config['view']!='mp' or config['manifest_sha256']!=digest:
        raise ValueError('Use an mp-trained checkpoint with its original manifest')
    if args.output.exists() and (args.output/'metrics.json').exists():
        raise ValueError('Report already exists. Choose a fresh --output.')
    val_rows,val_scores=predict(model,config,args.manifest,'val',device,args.batch_size)
    val_targets=[r['target'] for r in val_rows]
    # >= threshold uses CNN. 0 means CNN-only; >1 permits every valid rule.
    candidates=[0.0]+[round(x,2) for x in np.arange(.5,1.001,.05)]+[1.01]
    selection=[]
    for threshold in candidates:
        scores,predictions=fusion_scores(val_rows,val_scores,threshold)
        metrics=classification_metrics(val_targets,scores,CLASSES,predictions)
        selection.append({'threshold':threshold,'val_f1_macro':metrics['f1_macro'],'val_accuracy':metrics['accuracy']})
    # Tie favours lower threshold (fewer rule overrides).
    best=max(selection,key=lambda r:(r['val_f1_macro'],-r['threshold']))
    rows,scores=predict(model,config,args.manifest,'test',device,args.batch_size)
    targets=np.array([r['target'] for r in rows]); detected=np.array([r['detected'] for r in rows])
    modes={'mp_crop_cnn_full_image_fallback':(scores,scores.argmax(1))}
    strict_scores=scores.copy(); strict_scores[~detected]=0
    strict_pred=scores.argmax(1); strict_pred[~detected]=-1
    modes['mp_crop_cnn_strict_reject_misses']=(strict_scores,strict_pred)
    rule_scores=np.zeros_like(scores); rule_pred=np.full(len(rows),-1,dtype=int)
    for i,row in enumerate(rows):
        if row['detected'] and row['rule'] in CLASSES:
            rule_pred[i]=CLASSES.index(row['rule']); rule_scores[i,rule_pred[i]]=1
    modes['landmark_rules']=(rule_scores,rule_pred)
    modes['fusion_full_image_fallback']=fusion_scores(rows,scores,best['threshold'])
    if args.raw_checkpoint:
        raw_model,raw_config=load_model(args.raw_checkpoint,device)
        if raw_config['view']!='raw':
            raise ValueError('raw-checkpoint must be trained with --view raw')
        for key in ('model','size','gray','seed','epochs','batch_size','lr','weight_decay','pretrained','class_weighted','amp','manifest_sha256'):
            if raw_config[key]!=config[key]:
                raise ValueError('Uncontrolled raw/mp comparison: '+key)
        raw_rows,raw_scores=predict(raw_model,raw_config,args.manifest,'test',device,args.batch_size)
        if [r['id'] for r in raw_rows]!=[r['id'] for r in rows]:
            raise ValueError('Test sample order mismatch')
        modes['raw_image_cnn']=(raw_scores,raw_scores.argmax(1))
    result={name:classification_metrics(targets,values[0],CLASSES,values[1]) for name,values in modes.items()}
    report={'threshold_selection':best,'validation_search':selection,
        'detection_rate':float(detected.mean()),'detected':int(detected.sum()),'total_test':len(rows),
        'metrics':result,'checkpoint':str(args.checkpoint.resolve()),
        'raw_checkpoint':str(args.raw_checkpoint.resolve()) if args.raw_checkpoint else None,
        'smoke':config['smoke'],'leakage_detected':config['leakage_detected'],
        'experiment':{k:config[k] for k in ('model','seed','epochs','size','gray','pretrained','manifest_sha256','lr','batch_size','class_weighted','weight_decay','amp')},
        'classification_mAP_definition':'one-vs-rest AP averaged across classes; no detection IoU threshold',
        'fusion_score_definition':'CNN probability if selected; one-hot class score for rule override; not calibrated probability',
        'geometry':'63 wrist-centred palm-normalised xyz values; z contributes to pinch-distance rule, not concatenated into CNN',
        'limitations':'No dedicated illumination/occlusion benchmark or subject-disjoint provenance has been supplied.'}
    write_json(args.output/'metrics.json',report)
    write_json(args.output/'fusion_config.json',{'threshold':best['threshold'],
        'checkpoint':str(args.checkpoint.resolve()),'manifest_sha256':digest,
        'selection_split':'val','selection_metric':'f1_macro'})
    with (args.output/'predictions.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream); writer.writerow(['id','target','detected','rule']+list(modes))
        for i,row in enumerate(rows):
            writer.writerow([row['id'],row['label'],row['detected'],row['rule']]+[
                CLASSES[value[1][i]] if value[1][i]>=0 else 'unknown' for value in modes.values()])
    for name,metrics in result.items():
        print(name,{k:round(metrics[k],4) for k in ('accuracy','precision_macro','recall_macro','f1_macro','classification_mAP')})
    print('Saved:',args.output)

if __name__=='__main__':
    main()
