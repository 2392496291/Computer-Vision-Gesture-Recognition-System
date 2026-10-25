"""Qt preview with worker-thread detection and classification."""
import argparse
import json
import sys
import threading
import time
from pathlib import Path
import cv2
import torch
from PyQt5 import QtCore, QtGui, QtWidgets
from common import CLASSES, read_bgr
from inference import load_model, predict_image, fuse
from train import choose_device

class Worker(QtCore.QThread):
    frame=QtCore.pyqtSignal(object,str)
    error=QtCore.pyqtSignal(str)
    def __init__(self,checkpoint,source,threshold,device):
        super().__init__(); self.checkpoint=checkpoint; self.source=source
        self.threshold=threshold; self.device=device; self.stop_event=threading.Event()
    def run(self):
        capture=processor=None
        try:
            from hands import HandProcessor
            device=choose_device(self.device); model,config=load_model(self.checkpoint,device)
            still=isinstance(self.source,str) and Path(self.source).suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}
            processor=HandProcessor(static=still)
            if not still:
                capture=cv2.VideoCapture(self.source)
                if not capture.isOpened():
                    raise RuntimeError('Cannot open source')
            while not self.stop_event.is_set():
                if still:
                    original=read_bgr(self.source)
                else:
                    ok,original=capture.read()
                    if not ok:
                        break
                started=time.perf_counter(); hands=processor.detect(original)
                canvas=original.copy(); descriptions=[]
                if not hands:
                    scores=predict_image(model,config,original,device)
                    descriptions.append('No hand detected; full-image CNN fallback: %s (%.3f)'%(CLASSES[int(scores.argmax())],scores.max()))
                for hand in hands:
                    x1,y1,x2,y2=hand['bbox']
                    image=original[y1:y2,x1:x2] if config['view']=='mp' else original
                    scores=predict_image(model,config,image,device)
                    _,prediction,method=fuse(scores,hand['rule'],True,self.threshold)
                    label=CLASSES[prediction]
                    descriptions.append('%s | %s | CNN confidence %.3f'%(label,method,scores.max()))
                    cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,200,0),2)
                    cv2.putText(canvas,label,(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,200,0),2)
                    h,w=canvas.shape[:2]
                    for x,y,z in hand['landmarks']:
                        cv2.circle(canvas,(int(x*w),int(y*h)),2,(0,200,255),-1)
                elapsed=time.perf_counter()-started
                descriptions.append('Detection + inference + drawing: %.1f ms; hands: %d'%(elapsed*1000,len(hands)))
                self.frame.emit(canvas,'\n'.join(descriptions))
                if still:
                    break
                self.stop_event.wait(max(0,.03-elapsed))
        except Exception as error:
            self.error.emit(str(error))
        finally:
            if capture is not None:
                capture.release()
            if processor:
                processor.close()

class Window(QtWidgets.QMainWindow):
    def __init__(self,args):
        super().__init__(); self.args=args; self.worker=None
        self.setWindowTitle('Gesture Recognition - ResNeXt50 + DenseBlocks'); self.resize(1000,760)
        widget=QtWidgets.QWidget(); self.setCentralWidget(widget); layout=QtWidgets.QVBoxLayout(widget)
        self.checkpoint=QtWidgets.QLineEdit(str(args.checkpoint or '')); layout.addWidget(self.checkpoint)
        controls=QtWidgets.QHBoxLayout(); layout.addLayout(controls)
        for title,callback in [('Checkpoint',self.browse),('Image',self.image),('Video',self.video),('Camera',lambda:self.start(0)),('Stop',self.stop)]:
            button=QtWidgets.QPushButton(title); button.clicked.connect(callback); controls.addWidget(button)
        controls.addWidget(QtWidgets.QLabel('Fusion threshold'))
        self.threshold=QtWidgets.QDoubleSpinBox(); self.threshold.setRange(0,1.01); self.threshold.setSingleStep(.05)
        self.threshold.setValue(args.threshold); controls.addWidget(self.threshold)
        self.preview=QtWidgets.QLabel('Select a best.pt checkpoint and an input.'); self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setMinimumHeight(500); layout.addWidget(self.preview)
        self.status=QtWidgets.QLabel('Threshold changes apply to the next input. No synthetic confidence or IoU.')
        self.status.setWordWrap(True); layout.addWidget(self.status)
    def browse(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Checkpoint','','PyTorch (*.pt)')
        if path:
            self.checkpoint.setText(path)
    def image(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Image','','Images (*.jpg *.jpeg *.png *.bmp *.webp)')
        if path:
            self.start(path)
    def video(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Video','','Videos (*.mp4 *.avi *.mov *.mkv)')
        if path:
            self.start(path)
    def start(self,source):
        if self.worker and self.worker.isRunning():
            self.status.setText('Stop the current source first.'); return
        checkpoint=Path(self.checkpoint.text())
        if not checkpoint.is_file():
            self.status.setText('Select a valid best.pt checkpoint.'); return
        self.worker=Worker(checkpoint,source,self.threshold.value(),self.args.device)
        self.worker.frame.connect(self.display); self.worker.error.connect(self.status.setText)
        self.status.setText('Loading...'); self.worker.start()
    def stop(self):
        if self.worker:
            self.worker.stop_event.set()
    def display(self,image,status):
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB); h,w=rgb.shape[:2]
        qim=QtGui.QImage(rgb.data,w,h,rgb.strides[0],QtGui.QImage.Format_RGB888).copy()
        self.preview.setPixmap(QtGui.QPixmap.fromImage(qim).scaled(self.preview.size(),QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))
        self.status.setText(status)
    def closeEvent(self,event):
        self.stop()
        if self.worker and self.worker.isRunning() and not self.worker.wait(1000):
            self.status.setText('Stopping worker; close again once processing finishes.'); event.ignore(); return
        event.accept()

def main():
    torch.set_num_threads(4)
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--checkpoint',type=Path)
    p.add_argument('--fusion-config',type=Path); p.add_argument('--threshold',type=float,default=0.)
    p.add_argument('--device',default='auto'); args=p.parse_args()
    if args.fusion_config:
        config=json.loads(args.fusion_config.read_text(encoding='utf-8')); args.threshold=config['threshold']
        if args.checkpoint and args.checkpoint.resolve()!=Path(config['checkpoint']).resolve():
            p.error('Fusion configuration belongs to another checkpoint')
        args.checkpoint=Path(config['checkpoint'])
    app=QtWidgets.QApplication(sys.argv); window=Window(args); window.show(); sys.exit(app.exec_())
if __name__=='__main__':
    main()
