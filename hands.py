"""MediaPipe Hands localisation and scale-normalised landmark rules."""
import cv2
import numpy as np

class HandProcessor:
    def __init__(self, static=True, confidence=0.5, max_hands=2):
        import mediapipe as mp
        if not hasattr(mp, 'solutions'):
            raise RuntimeError('Install a MediaPipe version exposing mp.solutions.hands (see requirements).')
        self.hands = mp.solutions.hands.Hands(static_image_mode=static,
            max_num_hands=max_hands, model_complexity=1,
            min_detection_confidence=confidence, min_tracking_confidence=confidence)
    def close(self):
        self.hands.close()
    def detect(self, image):
        result = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        output=[]
        height,width=image.shape[:2]
        for hand in result.multi_hand_landmarks or []:
            xyz=np.array([[p.x,p.y,p.z] for p in hand.landmark],np.float32)
            # Scale x, y, z in the same width-based unit before 3D distances.
            geometric=xyz.copy(); geometric[:,1] *= height/width
            geometric -= geometric[0].copy()
            scale=float(np.linalg.norm(geometric[9]))
            geometric /= max(scale, 1e-6)
            points=xyz[:,:2]*[width,height]
            low=points.min(0); high=points.max(0)
            margin=max(10.0,float(max(high-low))*0.15)
            x1,y1=np.maximum(np.floor(low-margin),0).astype(int)
            x2,y2=np.minimum(np.ceil(high+margin),[width,height]).astype(int)
            if x2<=x1 or y2<=y1:
                continue
            output.append({'bbox':[int(x1),int(y1),int(x2),int(y2)],
                'landmarks':xyz.tolist(),'geometry63':geometric.reshape(-1).tolist(),
                'rule':rule_label(points,geometric)})
        return sorted(output,key=lambda h:(h['bbox'][2]-h['bbox'][0])*(h['bbox'][3]-h['bbox'][1]),reverse=True)

def rule_label(points, geometric):
    # Retain the original project's convex-hull finger rules. The pinch
    # test uses all three coordinates normalised by palm size (not 20 pixels).
    hull=cv2.convexHull(np.asarray(points[[0,1,2,3,6,10,14,19,18,17,10]],np.float32))
    fingers=[i for i in [4,8,12,16,20] if cv2.pointPolygonTest(hull,tuple(map(float,points[i])),False)<0]
    pinch=float(np.linalg.norm(geometric[4]-geometric[8]))<0.35
    if pinch and all(i in fingers for i in (12,16,20)):
        return 'OK'
    mapping={(): '0',(8,): '1',(8,12): '2',(8,12,16): '3',
        (8,12,16,20): '4',(4,8,12,16,20): '5',(4,20): '6',(4,): '8'}
    if fingers==[4,8]:
        return '7' if pinch else '8'
    return mapping.get(tuple(fingers))
