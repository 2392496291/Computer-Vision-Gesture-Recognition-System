"""Single-label classification metrics, including explicit unknown/rejection cases."""
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, confusion_matrix

def classification_metrics(targets, scores, classes, predictions=None):
    targets=np.asarray(targets,dtype=np.int64); scores=np.asarray(scores,dtype=float)
    if scores.shape != (len(targets),len(classes)) or not np.isfinite(scores).all():
        raise ValueError('Invalid score matrix')
    predictions=scores.argmax(1) if predictions is None else np.asarray(predictions,dtype=np.int64)
    p,r,f,support=precision_recall_fscore_support(targets,predictions,
        labels=list(range(len(classes))),zero_division=0)
    aps=[]
    for i in range(len(classes)):
        aps.append(float(average_precision_score(targets==i,scores[:,i])) if np.any(targets==i) else None)
    valid=[v for v in aps if v is not None]
    matrix=confusion_matrix(targets,predictions,labels=list(range(len(classes)))+[-1])
    return {'n':len(targets),'accuracy':float(np.mean(targets==predictions)),
        'precision_macro':float(p.mean()),'recall_macro':float(r.mean()),'f1_macro':float(f.mean()),
        'classification_mAP':float(np.mean(valid)) if valid else None,
        'rejected':int(np.sum(predictions==-1)),
        'per_class':{label:{'precision':float(p[i]),'recall':float(r[i]),'f1':float(f[i]),
            'AP':aps[i],'support':int(support[i])} for i,label in enumerate(classes)},
        'confusion_labels':classes+['unknown'],'confusion_matrix':matrix.tolist()}
