import numpy as np
import cv2

#cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture('video2.mov')


lower = np.array([0, 0, 130], dtype = "uint8")
upper = np.array([100, 100, 255], dtype = "uint8")
global n
n = 0
 
# find the colors within the specified boundaries and apply
# the mask


# function for X,Y and pressure measurement
def threedimmap(image):
    global n 
    n = n+1
    roi = image[0:390,120:550]
    grayroi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) # grayscale
    blur = cv2.GaussianBlur(roi, (11, 11), 0) # blur
    mask = cv2.inRange(blur, lower, upper)
    kernel = np.ones((2,2),np.uint8) # kernel for the morphological operators
    output = cv2.bitwise_and(blur, blur, mask = mask)
    
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY) # grayscale
    inverse = 255 - gray
    thresh= cv2.adaptiveThreshold(inverse,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
           cv2.THRESH_BINARY,51,1)
    kernel = np.ones((3,3),np.uint8) # kernel for the morphological operators
    terosion = cv2.erode(thresh,kernel,iterations = 2)
    closing = cv2.morphologyEx(terosion, cv2.MORPH_CLOSE, kernel)

    inverseinverse = 255 - thresh
    temperature = cv2.bitwise_and(grayroi, grayroi, mask = closing)
    ret3,th3 = cv2.threshold(temperature,130,255,cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8) # kernel for the morphological operators
    opening = cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel)

    dilation = cv2.dilate(inverseinverse,kernel,iterations = 0) # dilation
    colour = cv2.cvtColor(th3, cv2.COLOR_GRAY2BGR)

    colour[opening == 0 ] = [220, 100, 0]
    colour[opening == 255 ] = [0, 20, 200]

    final = output.copy()


    # Setup SimpleBlobDetector parameters.


    im2, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[4]
    cv2.drawContours(final, contours, -1, (0,255,0), 3)

    im2, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centres = []
    for i in range(len(contours)):
        moments = cv2.moments(contours[i])
        if (moments['m00'] == 0 or moments['m00'] == 0):
            mx = 0
        else:
            centres.append((int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])))
            cv2.circle(final, centres[-1], 1, (255, 255, 255), -1)

    
    # if n = 100 then save images
    if (n == 370):
        cv2.imwrite('Troi.png',roi)
        cv2.imwrite('Ttemperature.png',temperature)
        cv2.imwrite('Tthreshold.png',th3)
        cv2.imwrite('Topening.png',opening)
        cv2.imwrite('Tcolour.png',colour)
    

    # X and Y
    mx = 0
    my = 0
    black = 0 

    # lower resolution
    height, width = final.shape[:2]
    h = height/9
    h = round(h)
    w = width/13
    w = round(w)
    
    # mome



    # show results
    cv2.imshow("Output", colour) #closing

    # output variables for X and Y
    cxout = 0
    cyout = 0
    nob   = 0
    print(n)
    return cyout, cxout, nob

while(cap.isOpened()):
    ret, frame = cap.read() # camera readout
    cx, cy, nob = threedimmap(frame) # X, Y and pressure
    print(cx, cy, nob) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()