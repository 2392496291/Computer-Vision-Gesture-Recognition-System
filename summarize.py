"""Collect fresh run metrics; never carries over the old thesis scores."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from common import ROOT,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--runs',type=Path,default=ROOT/'runs')
    p.add_argument('--output',type=Path,default=ROOT/'reports'/'comparison'); p.add_argument('--pipelines',type=Path,default=ROOT/'reports'); args=p.parse_args()
    rows=[]
    for path in sorted(args.runs.rglob('test_metrics.json')):
        m=json.loads(path.read_text(encoding='utf-8')); c=json.loads((path.parent/'config.json').read_text(encoding='utf-8'))
        if m['smoke'] or m['leakage_detected']:
            print('Excluded development/leaked run:',path); continue
        row={k:c[k] for k in ('model','view','seed','parameters','epochs','size','gray','pretrained','manifest_sha256','lr','batch_size','class_weighted','weight_decay','amp')}
        row.update({k:m[k] for k in ('accuracy','precision_macro','recall_macro','f1_macro','classification_mAP','best_epoch')})
        row['run']=str(path.parent); rows.append(row)
    if not rows:
        raise RuntimeError('No completed non-smoke, non-leaked experiments found.')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'runs.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    groups={}
    keys=('model','view','epochs','size','gray','pretrained','manifest_sha256','lr','batch_size','class_weighted','weight_decay','amp')
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys),[]).append(row)
    summary=[]
    for key,group in groups.items():
        seeds=[r['seed'] for r in group]
        if len(seeds)!=len(set(seeds)):
            raise ValueError('Duplicate seed for the same experiment; remove duplicate run from comparison root')
        result=dict(zip(keys,key)); result.update(n=len(group),seeds=seeds)
        for metric in ('accuracy','precision_macro','recall_macro','f1_macro','classification_mAP'):
            vals=[r[metric] for r in group]
            result[metric]={'mean':float(np.mean(vals)),'std':float(np.std(vals,ddof=1)) if len(vals)>1 else None}
        summary.append(result)
    write_json(args.output/'summary.json',summary)
    pipeline_rows=[]
    for path in sorted(args.pipelines.rglob('metrics.json')):
        report=json.loads(path.read_text(encoding='utf-8'))
        if report.get('smoke') or report.get('leakage_detected') or 'experiment' not in report:
            continue
        for mode,metrics in report['metrics'].items():
            row=dict(report['experiment']); row['mode']=mode
            row.update({k:metrics[k] for k in ('accuracy','precision_macro','recall_macro','f1_macro','classification_mAP')})
            row.update(threshold=report['threshold_selection']['threshold'],detection_rate=report['detection_rate'],report=str(path))
            pipeline_rows.append(row)
    if pipeline_rows:
        with (args.output/'pipelines.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(pipeline_rows[0])); writer.writeheader(); writer.writerows(pipeline_rows)
        groups={}
        pipeline_keys=('mode','model','epochs','size','gray','pretrained','manifest_sha256','lr','batch_size','class_weighted','weight_decay','amp')
        for row in pipeline_rows:
            groups.setdefault(tuple(row[k] for k in pipeline_keys),[]).append(row)
        summary=[]
        for key,group in groups.items():
            seeds=[r['seed'] for r in group]
            if len(seeds)!=len(set(seeds)):
                raise ValueError('Duplicate pipeline seed for the same settings')
            result=dict(zip(pipeline_keys,key)); result.update(n=len(group),seeds=seeds)
            for metric in ('accuracy','precision_macro','recall_macro','f1_macro','classification_mAP','detection_rate'):
                vals=[r[metric] for r in group]
                result[metric]={'mean':float(np.mean(vals)),'std':float(np.std(vals,ddof=1)) if len(vals)>1 else None}
            summary.append(result)
        write_json(args.output/'pipeline_summary.json',summary)
    print('Saved:',args.output)
if __name__=='__main__':
    main()
