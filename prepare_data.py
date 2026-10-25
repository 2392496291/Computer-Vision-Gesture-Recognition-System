"""Audit existing splits and cache MediaPipe crops without relabelling samples."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from tqdm import tqdm
from common import CLASSES, EXTENSIONS, DEFAULT_DATA, ROOT, read_bgr, save_bgr, write_json

def scan(data):
    result=[]
    for split in ('train','val','test'):
        folder=data/split
        actual=sorted(p.name for p in folder.iterdir() if p.is_dir())
        if actual != sorted(CLASSES):
            raise ValueError('Class folders must be exactly '+str(CLASSES)+' in '+str(folder))
        for target,label in enumerate(CLASSES):
            paths=sorted(p for p in (folder/label).rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
            if not paths:
                raise ValueError('Empty class '+split+'/'+label)
            for p in paths:
                result.append({'id':p.relative_to(data).as_posix(),'path':str(p.resolve()),
                               'split':split,'label':label,'target':target})
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--output',type=Path,default=ROOT/'prepared')
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--clean-duplicates',action='store_true',help='Exclude label conflicts and retain one image per identical-pixel group; never deletes originals')
    parser.add_argument('--skip-mediapipe',action='store_true')
    parser.add_argument('--limit-per-class',type=int,default=0,help='Smoke test only; not a full experiment')
    args=parser.parse_args()
    args.data=args.data.resolve(); args.output=args.output.resolve()
    if (args.output/'manifest.jsonl').exists():
        raise RuntimeError('Manifest already exists; choose a fresh --output to preserve experiments.')
    if args.limit_per_class<0:
        parser.error('limit must be nonnegative')
    rows=scan(args.data)
    if args.limit_per_class:
        counts=Counter(); selected=[]
        for row in rows:
            key=(row['split'],row['label'])
            if counts[key]<args.limit_per_class:
                selected.append(row); counts[key]+=1
        rows=selected
    args.output.mkdir(parents=True,exist_ok=True)
    fingerprints=defaultdict(list); failures=[]
    for row in tqdm(rows,desc='Audit decoded pixels'):
        try:
            image=read_bgr(row['path'])
            digest=hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()
            fingerprints[digest].append(row)
        except Exception as error:
            failures.append({'id':row['id'],'error':str(error)})
    cross=[]; conflicting=[]
    for digest, group in fingerprints.items():
        if len({r['split'] for r in group})>1:
            cross.append({'sha256':digest,'ids':[r['id'] for r in group]})
        if len({r['target'] for r in group})>1:
            conflicting.append({'sha256':digest,'ids':[r['id'] for r in group]})
    report={'data_root':str(args.data),'total':len(rows),'smoke':bool(args.limit_per_class),
        'counts':{s:dict(Counter(r['label'] for r in rows if r['split']==s)) for s in ('train','val','test')},
        'unique_decoded_images':len(fingerprints),'unreadable':failures,
        'cross_split_duplicates':cross,'conflicting_labels':conflicting,
        'provenance':'Existing split reused. Original source/subject/video IDs not supplied; verify separately.'}
    excluded=[]
    if args.clean_duplicates:
        keep=set()
        for group in fingerprints.values():
            if len({r['target'] for r in group})>1:
                excluded.extend({'id':r['id'],'reason':'identical pixels with conflicting labels'} for r in group)
            else:
                keep.add(group[0]['id'])
                excluded.extend({'id':r['id'],'reason':'duplicate pixels; retained '+group[0]['id']} for r in group[1:])
        rows=[r for r in rows if r['id'] in keep]
        for split in ('train','val','test'):
            if {r['label'] for r in rows if r['split']==split} != set(CLASSES):
                raise RuntimeError('Cleaning emptied a class; inspect source data')
    report.update(cleaning_applied=args.clean_duplicates,excluded=excluded,retained_total=len(rows),
        retained_counts={s:dict(Counter(r['label'] for r in rows if r['split']==s)) for s in ('train','val','test')},
        unresolved_cross_split_groups=0 if args.clean_duplicates else len(cross),
        unresolved_label_conflicts=0 if args.clean_duplicates else len(conflicting))
    write_json(args.output/'audit.json',report)
    print('Samples:',len(rows),'cross-split duplicate groups:',len(cross),'conflicting labels:',len(conflicting))
    if args.audit_only:
        return
    if failures:
        raise RuntimeError('Unreadable images; see audit.json. No manifest published.')
    processor=None
    if not args.skip_mediapipe:
        from hands import HandProcessor
        processor=HandProcessor(static=True)
    temp=args.output/'manifest.jsonl.tmp'
    try:
        with temp.open('w',encoding='utf-8') as stream:
            for row in tqdm(rows,desc='Cache hand crops'):
                hand=None
                if processor:
                    image=read_bgr(row['path']); detected=processor.detect(image)
                    hand=detected[0] if detected else None
                row.update(detected=hand is not None,crop_path=None,rule=None,landmarks=None,geometry63=None)
                if hand:
                    row.update(hand,detected=True)
                    x1,y1,x2,y2=hand['bbox']
                    crop=args.output/'crops'/Path(row['id']+'.png')
                    save_bgr(crop,image[y1:y2,x1:x2])
                    row['crop_path']=str(crop)
                stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        write_json(args.output/'metadata.json',{'classes':CLASSES,'smoke':bool(args.limit_per_class),
            'mediapipe':not args.skip_mediapipe,'selection':'largest hand per labelled image',
            'miss_policy':'preserve original image for CNN fallback; strict metrics reject misses',
            'crop_margin':'15% of maximum hand extent, minimum 10 px','rule_pinch_palm_ratio':0.35})
        temp.replace(args.output/'manifest.jsonl')
    finally:
        if processor:
            processor.close()
    print('Manifest:',args.output/'manifest.jsonl')

if __name__=='__main__':
    main()
