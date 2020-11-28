# Testing Packages Working
import cv2
import numpy as np
import datetime
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
while(True):
    ret,frame=cap.read()

    print(cap.get(3))
    print(cap.get(4))\
    cap.set(3,1208)
    cap.set(4,720)
    font=cv2.FONT_HERSHEY_DUPLEX
    text="WIDTH: "+ str(cap.get(3))+ ' HEIGHT: '+ str(cap.get(4))
    datet=str(datetime.datetime.now())
    frame=cv2.putText(frame, text, (10,50), font, 1, (0,255,255), 1, cv2.LINE_AA)
    frame=cv2.putText(frame, datet, (10,450), font, 1, (0,255,255), 1, cv2.LINE_AA)
    out.write(frame)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('Camera',frame)
    if cv2.waitKey(1)==ord('q'):
        break   
out.release()    
cap.release()

# Creating Geometric Shapes on Images

img=cv2.imread('lena.jpg',1) #0=Grey Scale, 1=Color,-1=Unchanged
img2=np.zeros([512,512,3],dtype=np.uint8) #Blank Image Created with numpy
img=cv2.line(img,(0,0),(255,255),(255,0,0),10)
img=cv2.arrowedLine(img,(255,0),(255,255),(0,255,0),10)
img=cv2.rectangle(img,(384,0),(510,128),(255,255,0),10) #If thickness or last argument is-1 then the rectangle will be filled with the specified color
img=cv2.circle(img,(447,63),63,(255,255,0),10)
font=cv2.FONT_HERSHEY_SIMPLEX
img=cv2.putText(img,"Lena Image",(10,500),font,1,(255,255,255),5,cv2.LINE_AA)
img2=cv2.line(img2,(0,0),(255,255),(255,0,0),10)
img2=cv2.arrowedLine(img2,(255,0),(255,255),(0,255,0),10)
img2=cv2.rectangle(img2,(384,0),(510,128),(255,255,0),10) #If thickness or last argument is-1 then the rectangle will be filled with the specified color
img2=cv2.circle(img2,(447,63),63,(255,255,0),10)
img2=cv2.putText(img2,"Blank Image",(10,500),font,1,(255,255,255),5,cv2.LINE_AA)

cv2.imshow('Image2',img2)

cv2.imshow('Image',img)
cv2.waitKey(0)

cv2.destroyAllWindows()

#Mouse Events on Images

events=[i for i in dir(cv2) if 'EVENT' in i]
print(events)

def  click_event(event,x,y,flag,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print(str(x)+' '+str(y)) # Lines 76-79 Shows that clicking on the image with the left mouse shows clicked position
        font=cv2.FONT_HERSHEY_SIMPLEX
        text2='('+str(x)+','+str(y)+')'
        cv2.putText(img,text2,(x,y),font,0.5,(255,255,255),2,cv2.LINE_AA)
        cv2.circle(img,(x,y),3,(255,0,0),-1)
        points.append((x,y)) # Lines 80-83 show trailing lines. Like clicking a secondf point connects both point with a line. 
        if len(points)>=2:
            cv2.line(img,points[-1],points[-2],(255,0,0),4)

        blue=img[y,x,0] # Lines 85-90 shows the colour window a window that shows the coloyr of the clicked point in an image.
        green=img[y,x,1]
        red=img[y,x,2]
        print(str(blue)+' '+str(green)+' '+str(red))
        colorwindow=np.zeros([512,512,3],np.uint8)
        colorwindow[:]=[blue,green,red]
        cv2.imshow('color',colorwindow)

    if event==cv2.EVENT_RBUTTONDOWN: # Right click mouse button shows BGR Scale of clicked point. 
        blue=img[y,x,0]
        green=img[y,x,1]
        red=img[y,x,2]
        print(str(blue)+' '+str(green)+' '+str(red))
        font=cv2.FONT_HERSHEY_SIMPLEX
        text3='('+str(blue)+','+str(green)+','+str(red)+')'
        cv2.putText(img,text3,(x,y),font,0.5,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow('Image',img)

img=cv2.imread('lena.jpg',1) #0=Grey Scale, 1=Color,-1=Unchanged

points=[]
cv2.imshow('Image',img)

cv2.setMouseCallback('Image',click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
