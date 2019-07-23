import cv2
import time
import math
import sys
import argparse
import numpy as np
import copy

from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd512 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from keras.applications.imagenet_utils import preprocess_input

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

from pyimagesearch.centroidtracker import CentroidTracker
from VideoAtribut.VideoAtribut import VideoAtribut


#version -- 
def count_vec(objects1,objects2,copyImg,indicator):
    global counter_vehicleH,counter_vehicleH_menit,cent1Post,counter_vehicleR, IDCentroid, btsHijau,btsMerah
    
    idSekarang = []
    IDCentroid = []
    centroid_1 = []
    centroid_2 = []

    for i in range(len(objects1)):
        centroid_1.append(objects1[i][1])
        IDCentroid.append(objects1[i][0])

    for i in range(len(objects2)):
        centroid_2.append(objects2[i][1])

    # zero padding koordinat agar counter bisa tetap jalan 
    # ketika jumlah elemen list centroid tidak sama  
    if len(centroid_1) > len(centroid_2):
        centroid_2.extend(((0,0),)*abs(len(centroid_1)-len(centroid_2)))
    elif len(centroid_1) < len(centroid_2):
        centroid_1.extend(((0,0),)*abs(len(centroid_1)-len(centroid_2)))
    
    cent1Post = isOutside(centroid_1,IDCentroid,copyImg,indicator)
    luas_1 = areaCentroid(centroid_1,cent1Post,indicator)
    cent2Post = isOutside(centroid_2,IDCentroid,copyImg,indicator)
    luas_2 = areaCentroid(centroid_2,cent2Post,indicator)
    
    batas = va.getTreshold(indicator)

    for i in range(len(centroid_1)):

        if luas_1[i] != 0 and luas_2[i] != 0 and luas_2[i] <= batas and luas_1[i] > luas_2[i]:
            if indicator == 'merah':
                if IDCentroid[i] not in btsMerah and IDCentroid[i] in btsHijau:
                    counter_vehicleR = counter_vehicleR + 1
                    btsMerah.append(IDCentroid[i])
                    idSekarang.append(IDCentroid[i])   
                    

            else :
                if IDCentroid[i] not in btsHijau:
                    counter_vehicleH = counter_vehicleH + 1
                    counter_vehicleH_menit = counter_vehicleH_menit + 1
                    btsHijau.append(IDCentroid[i])
                    idSekarang.append(IDCentroid[i])
                    
        else:
            continue

    return idSekarang

def isOutside(centroid,ID,img,indicator):
    global btsHijau,btsMerah
    count = []

    if indicator == 'merah':
        bts = btsMerah
    else :
        bts = btsHijau

    if ID in bts:
        count.append(0)

    else:
        roi = va.getROIValue(indicator)

        for i in range(len(centroid)):
            sama = 0

            for j in range(0,3):
                if img[roi[1],roi[0],j] != img[int(centroid[i][1]),int(centroid[i][0]),j]:
                    count.append(0)
                    break
                else:
                    sama += 1
                
                if sama == 3:
                    count.append(1)
    
    return count

def areaCentroid(centroid,count,indicator):
    global pt1_r,pt2_r,dist_r,pt1_h,pt2_h,dist_h
    luasCentroid = []
    
    if indicator == 'merah':
        pt1 = pt1_r
        pt2 = pt2_r
        dist = dist_r
    else:
        pt1 = pt1_h
        pt2 = pt2_h
        dist = dist_h

    for i in range(len(count)):
        
        if count[i] == 1:
            
            dist_b1 = math.sqrt( (centroid[i][1] - pt1[1])**2 + (centroid[i][0] - pt1[0])**2 )
            dist_b2 = math.sqrt( (pt2[1] - centroid[i][1])**2 + (pt2[0] - centroid[i][0])**2 )

            s = (dist_b1+dist_b2+dist)/2
            luas = math.sqrt(s*(s-dist)*(s-dist_b1)*(s-dist_b2))
            
            luasCentroid.append(luas)
        else:
            luasCentroid.append(0)

    return luasCentroid

# OpenCV Font
font = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
font_thickness = 2

# Set the image size.
img_height = 512
img_width = 512

# 1: Build the Keras model

K.clear_session() # Clear previous models from memory.

model = ssd_512(image_size=(img_height, img_width, 3),
                n_classes=20,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05], # The scales for MS COCO are [0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06]
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
               two_boxes_for_ar1=True,
               steps=[8, 16, 32, 64, 128, 256, 512],
               offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
               clip_boxes=False,
               variances=[0.1, 0.1, 0.2, 0.2],
               normalize_coords=True,
               subtract_mean=[123, 117, 104],
               swap_channels=[2, 1, 0],
               confidence_thresh=0.55,
               iou_threshold=0.45,
               top_k=200,
               nms_max_output_size=400)

# 2: Load the trained weights into the model.

# TODO: Set the path of the trained weights.
weights_path = '/content/drive/My Drive/gcolab/VGG_VOC0712Plus_SSD_512x512_ft_iter_160000.h5'

model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
confidence_thresh=0.55
ct = CentroidTracker()

writer = None
orig_images = [] # Store the images here.
input_images = [] # Store resized versions of the images here.

VideoInput = ['gambir011','gambir014','lenteng','pulogadung','kebonsirih','kananbagus']
objects_1 = []
IDCentroid = []
btsMerah = []
btsHijau = []
hijau = 'hijau'
merah = 'merah'
counter_vehicleH = 0
counter_vehicleH_menit = 0
counter_vehicleR = 0

VIDEO_STREAM_OUT = '/content/drive/My Drive/gcolab/hasil14.avi'
jumlahkendaraan = 0


def main(args,IDVideo):
    global counter_vehicleH, counter_vehicleH_menit, counter_vehicleR,objects_1, IDCentroid, jumlahkendaraan, writer
    #home/muflih/coba_frame/kananbagus.mp4
    #rtsp://103.119.144.146/Pulo-Gadung-011-704487_3

    cap = cv2.VideoCapture('/content/drive/My Drive/gcolab/{}.mp4'.format(args.video))

    while (cap.isOpened()):
        ret, frame = cap.read()
        break

    kecepatanid = []
    framehijau = []
    framemerah = []
    idframemerah = []
    idframehijau = []
    id_kec = []
    copyImgR = copy.deepcopy(frame)
    copyImgH = copy.deepcopy(frame)

    #area merah
    
    roi_R = va.getRedROI()
    roi_H = va.getGreenROI()

    cv2.fillPoly(copyImgR, [roi_R],(0,0,255))
    cv2.fillPoly(copyImgH, [roi_H],(0,255,255))
    #result.write(str(roi_R))
    #result.write(str(roi_H))
    
    idpertama = 0
    # performance indicator
    avg_fps = []
    total_object_box = 0
    count = 0
    kepadatan = 0
    hitungan = 0
    kec_total = []
    kec_menit = []
    while (cap.isOpened()):
        # start fps measurement
        

        start = time.time()

        ret, frame = cap.read()
        try:
            img = cv2.resize(frame, (512, 512))
        except:
            break
            
        count += 1
        orig_images.append(frame)
        img = image.img_to_array(img) 
        input_images.append(img)
        input_feed = preprocess_input(np.array(input_images))
        #input_feed = np.array(input_images)
        input_feed = np.expand_dims(input_images[0], axis=0)
        
        y_pred = model.predict(input_feed)

        y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_thresh] for k in range(y_pred.shape[0])]

        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        #print("Predicted boxes:")
        #print('   class   conf xmin   ymin   xmax   ymax')
        #print(y_pred_thresh[0],'\n')


        # Display the image and draw the predicted boxes onto it.
        classes = ['background',
                'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat',
                'chair', 'cow', 'diningtable', 'dog',
                'horse', 'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor']

        # copy centroid 1 to 2
        # and clear centroid 1 to get newest value
        objects_2 = list(objects_1)
        rects = []

        for box in y_pred_thresh[0]:
            total_object_box += 1
            # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
            if classes[int(box[0])] == 'car':
                xmin = box[2] * orig_images[0].shape[1] / img_width
                ymin = box[3] * orig_images[0].shape[0] / img_height
                xmax = box[4] * orig_images[0].shape[1] / img_width
                ymax = box[5] * orig_images[0].shape[0] / img_height
                
                label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
                box = [int(xmin),int(ymin),int(xmax),int(ymax)] * np.array([1,1,1,1])
                rects.append(box)
                cv2.rectangle(frame, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0,165,255), 2)
                cv2.putText(frame,label,(int(xmin),int(ymin-10)), font, fontscale,(255,255,255),font_thickness,cv2.LINE_AA)
            

        cv2.line(frame,pt1_r,pt2_r,(0,0,255),3)
        cv2.line(frame,pt1_h,pt2_h,(0,255,0),3)
        
        
        objects = ct.update(rects)
        objects_1 = list(objects.items())

        for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            
        hitunghijau = count_vec(objects_1,objects_2,copyImgH,hijau)
        hitungmerah = count_vec(objects_1,objects_2,copyImgR,merah)

    
        #print(hitunghijau)
        counter_textH = 'Vehicle detected: {}'.format(counter_vehicleH)
        counter_textR = 'Vehicle detected: {}'.format(counter_vehicleR)

        if counter_vehicleH > 0:
        	for i in range(len(hitunghijau)):
        		framehijau.append(count)

        	idframehijau = list(zip(btsHijau,framehijau))
            #print('hijau',idframehijau)
        if counter_vehicleR > 0:
            for i in range(len(hitungmerah)):
                framemerah.append(count)

            idframemerah = list(zip(btsMerah,framemerah))
            #print('merah',idframemerah)
            
        if counter_vehicleR > 0 and counter_vehicleH > 0:
            for i in range(len(idframehijau)):
                if idframehijau[i][0] in btsMerah:

                    for j in range(len(idframemerah)):
                        if idframehijau[i][0] == idframemerah[j][0]:
                            tempframeR = idframemerah[j][1]

                    length = va.getLength()
                    selisih = tempframeR - idframehijau[i][1]
                    kecepatan = length/selisih * 30 * 3.6
                    if idframehijau[i][0] not in id_kec: #Kecepatan sama
                      kec_total.append(kecepatan)
                      kecepatan_mnt = kecepatan / 3.6 * 60
                      kec_menit.append(kecepatan_mnt)
                      id_kec.append(idframehijau[i][0])
                      kecepatanid.append([idframehijau[i][0],kecepatan])
                      if kecepatan > 90:
                          print('kec',kecepatan)
                          print('id',idframehijau[i][0])

            
        
        for (objectID, centroid) in objects.items():
            for i in range(len(kecepatanid)):
                if kecepatanid[i][0] == objectID:
                    kec = "{} km/h".format(kecepatanid[i][1])
                    cv2.putText(frame, kec, (centroid[0] - 20, centroid[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        #hitung kepadatan
        if count%1800 == 0:
            hitungan += 1
            print(hitungan)
            if len(kec_menit) == 0:
                print('Avg Speed :', 0)
                kepadatan = 0
    
            else:
                kepadatan = counter_vehicleH_menit/(sum(kec_menit)/len(kec_menit))
                print('Avg Speed : ',(sum(kec_menit)/len(kec_menit))/60*3.6)
            print(kepadatan*1000,'kendaraan/km')
            print(counter_vehicleH_menit*60,'kendaraan/jam')
            print()
            counter_vehicleH_menit = 0
            kec_menit = []

        
    
        cv2.putText(frame,counter_textH,(5,100), font, fontscale,(0,255,0),font_thickness,cv2.LINE_AA)
        cv2.putText(frame,counter_textR,(5,200), font, fontscale,(0,0,255),font_thickness,cv2.LINE_AA)
        #cv2.putText(frame,str(kepadatan),(5,300), font, fontscale,(255,255,0),font_thickness,cv2.LINE_AA)
        
        #cv2.imshow('video',frame)
        

        # end fps measurement
        end = time.time()
        
        # Time elapsed
        seconds = end - start

        # Calculate frames per second
        fps  = 1 / seconds

        # save current fps to list
        avg_fps.append(fps)

        orig_images.pop()
        input_images.pop()

        #print(centroid_1[0])
        #print(centroid_2)
        #print()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
    cv2.destroyAllWindows()
    cap.release()

    print("Average FPS: ", sum(avg_fps)/len(avg_fps))
    print("Detected Vehicles Red: ", counter_vehicleR)
    print("Detected Vehicles Green: ", counter_vehicleH)
    print("Kecepatan Rata2 ",sum(kec_total)/len(kec_total))
    print(start,end)

    #result = open('result non tracking.txt','a')
    

    # reset variables
    counter_vehicle = 0
    #centroid_1.clear()
    #centroid_2.clear()

if __name__ == '__main__':
    # jumlah uji coba
    IDVideo = None
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    args = parser.parse_args()  

    for i in range(len(VideoInput)):
        if args.video == VideoInput[i]:
            IDVideo = i
            break

    if IDVideo == None:
        print('Video {} cannot found'.format(args.video))
        sys.exit(2)    

    va = VideoAtribut(IDVideo)  #garis hijau 
    
    print('Version: 8.54')
    pt_h = va.getGreenPoint()
    pt1_h = pt_h[0] 
    pt2_h = pt_h[1] 
        
    dist_h = math.sqrt( (pt2_h[0] - pt1_h[0])**2 + (pt2_h[1] - pt1_h[1])**2 )

    #garis merah
    pt_r = va.getRedPoint()
    pt1_r = pt_r[0]
    pt2_r = pt_r[1]

    dist_r = math.sqrt( (pt2_r[0] - pt1_r[0])**2 + (pt2_r[1] - pt1_r[1])**2 )
    main(args,IDVideo)


