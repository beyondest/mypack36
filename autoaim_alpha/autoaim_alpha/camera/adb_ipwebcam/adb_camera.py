import cv2



    
#h264_ulaw.sdp
#h264_pcm.sdp"
    
cap = cv2.VideoCapture("rtsp://127.0.0.1:8080/h264_pcm.sdp")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
        
    cv2.imshow('press esc to quit', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
    


