# Testing Packages Working
import cv2

print(cv2.__version__)

# Read And Write Images
# Program to take in an image and read from file and write that image to file

img=cv2.imread('lena.jpg',0) #0=Grey Scale, 1=Color,-1=Unchanged
print(img)

cv2.imshow('Image',img)
k=cv2.waitKey(0)
if k==27:
    cv2.destroyAllWindows()
elif(k==ord('s')):
    cv2.imwrite('l.png',img)

# Read And Write Images From Camera

cap=cv2.VideoCapture(0) # For Video Files Enter File Name as First Argument
fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('Camera.avi',fourcc,20.0,(640,480))
while(cap.isOpened()):
    ret,frame=cap.read()
    if ret==True:
        #print(cap.get(3))
        #print(cap.get(4))
        out.write(frame)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow('Camera',gray)
        if cv2.waitKey(1)==ord('q'):
            break   
        cap.release()
        out.release()
    else:
        break
 




