# Testing Packages Working
import cv2
import numpy as np
import datetime
import matplotlib.pyplot as plt
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

# Image Tresholding


grad=cv2.imread('gradient.png',1)
ret1, th1=cv2.threshold(grad,127,255,cv2.THRESH_BINARY)
_, th2=cv2.threshold(grad,127,255,cv2.THRESH_BINARY_INV)
_, th3=cv2.threshold(grad,127,255,cv2.THRESH_TRUNC)
_, th4=cv2.threshold(grad,127,255,cv2.THRESH_TOZERO)
_, th5=cv2.threshold(grad,127,255,cv2.THRESH_TOZERO_INV)  
cv2.imshow('Gradient',grad) 
cv2.imshow('TBGradient',th1) # All pixels below threshold will be 0 and above threshold 1 
cv2.imshow('TBIGradient',th2) # All pixels below threshold will be 1 and above threshold 0
cv2.imshow('TTGradient',th3) # All pixels below threshold will remain unchanged and above threshold will be same pixel value as threshold
cv2.imshow('TZGradient',th4) # All pixels below threshold will be 0 and above threshold will remain unchanged
cv2.imshow('TZIGradient',th5) # All pixels below threshold will trmain unchanged and above threshold 0

cv2.waitKey(0)
cv2.destroyAllWindows()

# Adaptive thresholding

sudoku=cv2.imread('sudoku.png',0)
ret1, th1=cv2.threshold(sudoku,127,255,cv2.THRESH_BINARY)
th2=cv2.adaptiveThreshold(sudoku,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th3=cv2.adaptiveThreshold(sudoku,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
cv2.imshow('Sudoku', sudoku)
cv2.imshow('Normal Thresholding', th1) # Binary Thresholding
cv2.imshow('MAT', th2)  # Adaptive thresholding on many regions where thresholding is decided upon mean of pixel values in the region along a value c
cv2.imshow('GAT', th3) # Adaptive thresholding on many regions where thresholding is decided upon Weighted Gaussian mean of pixel values in the region along a value c
cv2.waitKey(0)
cv2.destroyAllWindows()



# Matplotlib

lena=cv2.imread('lena.jpg',1)
cv2.imshow('Lena', lena)
lena=cv2.cvtColor(lena, cv2.COLOR_BGR2RGB)

plt.imshow(lena)
#plt.xticks([]),plt.yticks([])
plt.show() 
cv2.waitKey(0)
cv2.destroyAllWindows()


grad=cv2.imread('gradient.png',0)
ret1, th1=cv2.threshold(grad,127,255,cv2.THRESH_BINARY)
_, th2=cv2.threshold(grad,127,255,cv2.THRESH_BINARY_INV)
_, th3=cv2.threshold(grad,127,255,cv2.THRESH_TRUNC)
_, th4=cv2.threshold(grad,127,255,cv2.THRESH_TOZERO)
_, th5=cv2.threshold(grad,127,255,cv2.THRESH_TOZERO_INV)  

titles=['Gradient','TBGradient','TBIGradient','TTGradient','TZGradient','TZIGradient']
imgs=[grad,th1,th2,th3,th4,th5]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(imgs[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
#cv2.imshow('Gradient',grad) 
#cv2.imshow('TBGradient',th1) # All pixels below threshold will be 0 and above threshold 1 
#cv2.imshow('TBIGradient',th2) # All pixels below threshold will be 1 and above threshold 0
#cv2.imshow('TTGradient',th3) # All pixels below threshold will remain unchanged and above threshold will be same pixel value as threshold
#cv2.imshow('TZGradient',th4) # All pixels below threshold will be 0 and above threshold will remain unchanged
#cv2.imshow('TZIGradient',th5) # All pixels below threshold will trmain unchanged and above threshold 0
plt.savefig('Thresholding.jpg')
plt.show()
#cv2.waitKey(0)
#cv2.destroyAllWindows()


# Morphological Transformations in OpenCV

smart=cv2.imread('smarties.png',0) 
_,mask=cv2.threshold(smart,220,255,cv2.THRESH_BINARY_INV) # Binary Threshold value
kernel=np.ones((5,5),np.uint8) # A sliding window that compares its values with small portions of the image values
dilation=cv2.dilate(mask,kernel,iterations=2) # if all pixels under kernel are atleast 1 then pixels will light up 
erosion=cv2.erode(mask,kernel,iterations=2) # If all pixels under kernel are 1 only then pixel will be made 1
opening=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel) #first erosion is carried out then dilation over erosion
closing=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel) # first dilation is carried out followed by erosion
mg=cv2.morphologyEx(mask,cv2.MORPH_GRADIENT,kernel) # dilatipon-erosion
th=cv2.morphologyEx(mask,cv2.MORPH_TOPHAT,kernel) # Original image-opening


titles=['Smart','Mask','Dilation','Erosion','Opening','Closing','Gradient','Top Hat']
imgs=[smart,mask, dilation,erosion,opening,closing,mg,th]

for i in range(8):
    plt.subplot(2,4,i+1),plt.imshow(imgs[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.savefig('Morphology.jpg')
plt.show()



# Smoothening and Blurring Of Images

logo=cv2.imread('opencv-logo.png')
water=cv2.imread('water.png')
lena=cv2.imread('lena.jpg')


lena=cv2.cvtColor(lena,cv2.COLOR_BGR2RGB) 

kernel=np.ones((5,5),np.float32)/25
homog=cv2.filter2D(lena,-1,kernel)
blur=cv2.blur(lena,(5,5))
gblur=cv2.GaussianBlur(lena,(5,5),0)
mblur=cv2.medianBlur(lena,5)
bfilter=cv2.bilateralFilter(lena,9,75,75)
titles=['Lena','2D Filtering','Blurring','Gaussian Blur','Median Blur','Bilateral Filter']
imgs=[lena,homog,blur,gblur,mblur,bfilter]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(imgs[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.savefig('Filters.jpg')
plt.show()

# Image Gradients

messi=cv2.imread('messi5.jpg',0)
sudoku=cv2.imread('sudoku.png',0)
lap=cv2.Laplacian(sudoku,cv2.CV_64F,ksize=3)
lap=np.uint8(np.absolute(lap))
sobelx=cv2.Sobel(sudoku,cv2.CV_64F,1,0)
sobely=cv2.Sobel(sudoku,cv2.CV_64F,0,1)

sobelx=np.uint8(np.absolute(sobelx))
sobely=np.uint8(np.absolute(sobely))

sobelxy=cv2.bitwise_or(sobelx,sobely)

titles=['Sudoku','Laplacian Gradient','SobelX Gradient','SobelY Gradient','SobelXY Gradient']
imgs=[sudoku,lap,sobelx,sobely,sobelxy]

for i in range(5):
    plt.subplot(2,3,i+1),plt.imshow(imgs[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.savefig('Image Gradients.jpg')
plt.show()


# Canny Edge Detectors Using Open CV

messi=cv2.imread('messi5.jpg',0)
canny=cv2.Canny(messi,100,200)
lap=cv2.Laplacian(messi,cv2.CV_64F,ksize=3)
lap=np.uint8(np.absolute(lap))
sobelx=cv2.Sobel(messi,cv2.CV_64F,1,0)
sobely=cv2.Sobel(messi,cv2.CV_64F,0,1)

sobelx=np.uint8(np.absolute(sobelx))
sobely=np.uint8(np.absolute(sobely))

sobelxy=cv2.bitwise_or(sobelx,sobely)

titles=['Messi','Laplacian Gradient','SobelX Gradient','SobelY Gradient','SobelXY Gradient','Canny']
imgs=[messi,lap,sobelx,sobely,sobelxy,canny]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(imgs[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.savefig('Canny.jpg')
plt.show()


# Image Pyramids

lena=cv2.imread('lena.jpg')
lc=lena.copy()

gp=[lc]

for i in range(6):
    lc=cv2.pyrDown(lc)
    gp.append(lc)
    #cv2.imshow(str(i),lc)

layer=gp[5]
cv2.imshow('Top Layer Gaussian Pyramid',layer)
lp=[layer]

for i in range(5,0,-1):
    gaussian_extended=cv2.pyrUp(gp[i])
    laplacian=cv2.subtract(gp[i-1],gaussian_extended)
    cv2.imshow(str(i),laplacian)

#lr1=cv2.pyrDown(lena)
#lr2=cv2.pyrDown(lr1)
#hr2=cv2.pyrUp(lr2)
#cv2.imshow('Original',lena)
#cv2.imshow('Scaled Down1',lr1)
#cv2.imshow('Scaled Down2',lr2)
#cv2.imshow('Scaled Up2',hr2)

cv2.waitKey(0)

cv2.destroyAllWindows()


