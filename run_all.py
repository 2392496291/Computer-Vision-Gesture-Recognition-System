"""Run the four model experiments and pipeline evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from common import manifest_digest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.epochs<1 or args.batch_size<2:
        parser.error('epochs >= 1 and batch-size >= 2 required')
    root=Path(__file__).resolve().parent
    def run(script,*parameters):
        command=[sys.executable,'-B',str(root/script)]+[str(p) for p in parameters]
        print(subprocess.list2cmdline(command),flush=True)
        if not args.dry_run:
            subprocess.run(command,cwd=str(root),check=True)
    manifest=root/'prepared'/'manifest.jsonl'
    if not manifest.exists():
        run('prepare_data.py','--clean-duplicates')
    for model,view in [('plain_cnn','raw'),('resnext50','raw'),('resnext50_dense','raw'),('resnext50_dense','mp')]:
        out=root/'runs'/('%s_%s_seed%d'%(model,view,args.seed))
        config_file=out/'config.json'
        if config_file.exists():
            config=json.loads(config_file.read_text(encoding='utf-8'))
            expected={'model':model,'view':view,'epochs':args.epochs,'batch_size':args.batch_size,
                'seed':args.seed,'size':64,'gray':True,'pretrained':False,'class_weighted':False,
                'lr':.001,'weight_decay':0.,'amp':False,'smoke':False}
            if manifest.exists():
                expected['manifest_sha256']=manifest_digest(manifest)
            for key,value in expected.items():
                if config.get(key)!=value:
                    raise RuntimeError('Existing run has different '+key+': '+str(out))
        if (out/'test_metrics.json').exists():
            print('Already completed:',out,flush=True)
        else:
            run('train.py','--model',model,'--view',view,'--seed',args.seed,
                '--epochs',args.epochs,'--batch-size',args.batch_size,'--device',args.device,'--output',out)
    report=root/'reports'/('pipeline_seed%d'%args.seed)
    if not (report/'metrics.json').exists():
        run('evaluate_pipeline.py','--checkpoint',root/'runs'/('resnext50_dense_mp_seed%d'%args.seed)/'best.pt',
            '--raw-checkpoint',root/'runs'/('resnext50_dense_raw_seed%d'%args.seed)/'best.pt',
            '--output',report,'--batch-size',args.batch_size,'--device',args.device)
    run('summarize.py')
    if args.dry_run:
        print('Dry run only; no data preparation or training started.')

if __name__=='__main__':
    main()
