import cv2
import pickle
def preprocess_images(img):
    h,w,_ = img.shape
    start = int(h*4/22.)
    finish = int(h*18/22.)
    image = img[start:finish, start:finish]
    image = cv2.resize(image,(224,224))
    return image.reshape((1,224,224,3))

def preprocess_frames(img, video_no, frame_no, bboxes):
    t,b,l,r = bboxes[str(video_no)+'_'+str(frame_no)]
    image=img[t:b,l:r]
    image = cv2.resize(image,(224,224))
    return image.reshape((1,224,224,3))      
