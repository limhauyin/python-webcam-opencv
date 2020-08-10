import numpy as np
import face_recognition as fr 
import cv2 
# take video from webcam
video_capture = cv2.VideoCapture(0)

image = fr.load_image_file("face.JPG")
wenyi_image = fr.load_image_file("wenyi.jpeg")
sz_image = fr.load_image_file("sz.jpeg")
# take video and analyse
image_face_encoding = fr.face_encodings(image)[0]
wenyi_image_face_encoding = fr.face_encodings(wenyi_image)[0]
sz_image_face_encoding = fr.face_encodings(sz_image)[0]
# put other face if have other face
known_face_encodings = [image_face_encoding,wenyi_image_face_encoding,sz_image_face_encoding]
# put other name if have other face
know_face_names = ["Hau Yin","Wen Yi","SZU NING"]

while True: 
    ret, frame = video_capture.read()

    rgb_frame = frame[:,:,::-1]
    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame,face_locations)
    # 
    for(top,right,bottom,left), face_encoding in zip (face_locations,face_encodings):
        # compare faces define and face in webcame
        matches = fr.compare_faces(known_face_encodings,face_encoding)

        name = "Unknown"

        face_distances = fr.face_distance(known_face_encodings,face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index] : 
            name = know_face_names[best_match_index]
        # make rectangle on face 
        cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
        cv2.rectangle(frame,(left,bottom-35),(right,bottom),(0,0,255),cv2.FILLED)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,name,(left+6,bottom -6),font,1.0,(255,255,255),1)

    cv2.imshow('Webcam_face_reg',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        breakpoint

video_capture.release()
cv2.destroyAllWindows()


        

