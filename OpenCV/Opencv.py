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

# Performing Arithmatic Operations in Images

img3=cv2.imread('messi5.jpg',1)
img4=cv2.imread('opencv-logo.png',1)

print(img3.shape)# Basic measures of image, shape gives numbers of rows, columns and channels(b,g,r)
print(img3.size) # Gives number of pixels in image (rows*columns*channels)
print(img3.dtype) # Data type of pixel values
print(img4.shape)
print(img4.size)
print(img4.dtype)

b,g,r=cv2.split(img3) #Splits image into Blue green and red channels
img3=cv2.merge((b,g,r)) # Merges individual channels into image

cv2.imshow('Messib',b)
cv2.imshow('Messig',g)
cv2.imshow('Messir',r)

ball=img3[280:340,330:390] # Takes coordinates of the ball in the image
img3[273:333,100:160]=ball # Places these coordinates in a different part of image
img3=cv2.resize(img3,(512,512)) # resizing 2 images so that we can add 2 images together
img4=cv2.resize(img4,(512,512))
dst=cv2.add(img3,img4)
wdst=cv2.addWeighted(img3,.8,img4,.2,100) # Unlike Add function, this weigherd Add function gives dominance factor to 2 added images (src*alpha+src2*beta +gamma)
cv2.imshow('Messi',img3)
cv2.imshow('Logo',img4)
cv2.imshow('Mlogo',dst)
cv2.imshow('MWlogo',wdst)
cv2.waitKey(0)

cv2.destroyAllWindows()

# Bitwise operations on Images

img5=np.zeros((250,500,3),np.uint8)
img5=cv2.rectangle(img5,(200,0),(300,100),(255,255,255),-1)
img6=np.zeros((250,500,3),np.uint8)
img6=cv2.rectangle(img6,(250,0),(500,1000),(255,255,255),-1)
bitand=cv2.bitwise_and(img5,img6) # (And Operation)
bitor=cv2.bitwise_or(img5,img6) # (Or Operation)
bitnot=cv2.bitwise_not(img5) # (Not Operation)
bitxor=cv2.bitwise_xor(img5,img6) # (Xor Operation)
cv2.imshow(' image',img5)
cv2.imshow('vert image',img6)
cv2.imshow('And image',bitand)
cv2.imshow('Or image',bitor)
cv2.imshow('Not image',bitnot)
cv2.imshow('Xor image',bitxor)

cv2.waitKey(0)
cv2.destroyAllWindows()

#Trackbars in OpenCV
def nothing(x):
    pass

tbimg=np.zeros((300,512,3),np.uint8)
cv2.namedWindow('Image')

cv2.createTrackbar('B','Image',0,255,nothing)  # Creating a trackbar
cv2.createTrackbar('G','Image',0,255,nothing)
cv2.createTrackbar('R','Image',0,255,nothing)
cv2.createTrackbar('Switch','Image',0,1,nothing)
while(True):
    
    cv2.imshow('Image',tbimg)

    k=cv2.waitKey(1)
    if k==ord('q'):
        break
    b=cv2.getTrackbarPos('B','Image') # Getting Trackbar position for analysis
    g=cv2.getTrackbarPos('G','Image')
    r=cv2.getTrackbarPos('R','Image')
    
    s=cv2.getTrackbarPos('Switch','Image')
    
    if s==0:
        tbimg[:]=[0,0,0]
    else:
        tbimg[:]=[b,g,r]
cv2.destroyAllWindows()



# Hue Saturation Value With Images

cap2=cv2.VideoCapture(0)
cv2.namedWindow('Tracking')
cv2.createTrackbar('LH','Tracking',0,255,nothing)
cv2.createTrackbar('LS','Tracking',0,255,nothing)
cv2.createTrackbar('LV','Tracking',0,255,nothing)
cv2.createTrackbar('UH','Tracking',255,255,nothing)
cv2.createTrackbar('US','Tracking',255,255,nothing)
cv2.createTrackbar('UV','Tracking',255,255,nothing)

while(True):
    #frame=cv2.imread('smarties.png')
    #cv2.imshow('Frame',frame)
    ret,frame=cap2.read()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lh=cv2.getTrackbarPos('LH','Tracking')
    ls=cv2.getTrackbarPos('LS','Tracking')
    lv=cv2.getTrackbarPos('LV','Tracking')

    uh=cv2.getTrackbarPos('UH','Tracking')
    us=cv2.getTrackbarPos('US','Tracking')
    uv=cv2.getTrackbarPos('UV','Tracking')

    
    l_b=np.array([lh,ls,lv])
    u_b=np.array([uh,us,uv])
    mask=cv2.inRange(hsv,l_b,u_b)

    res=cv2.bitwise_and(frame,frame,mask=mask)
    cv2.imshow('Mask',mask)
    cv2.imshow('Result',res)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
cv2.destroyAllWindows()
cap2.release()
