import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Color an image')
  parser.add_argument('--path', dest='path',
                      help='path of image to colorize',
                      default=None, type=str) # this is an optional argument 
  parser.add_argument('--task', dest='task',
                      help='what task to perform, skin detect or colorize?',
                      default="skin detect", type=str) #this is also an optional argument 

  args = parser.parse_args()

  return args


def colorize(image):
    """
    """
    #Converting rgb image to a grayscale image
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Trying to convert it back to a rgb image 
    rgb_image = cv2.cvtColor(gray_image,cv2.COLOR_GRAY2RGB)

    return rgb_image


def skin_detect(image):
    """
    """
    # define the upper and lower boundaries of the HSV pixel
    # intensities to be considered 'skin'
    minVal = np.array([0, 48, 80], dtype = "uint8")
    maxVal = np.array([20, 255, 255], dtype = "uint8")

    #read in a test image 
    img = cv2.imread(image)

    # convert test image to hsv
    converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #  determine the HSV pixel intensities that fall into the speicifed upper and lower boundaries
    #This is the main skin detection function 
    skinMask = cv2.inRange(converted, minVal, maxVal)

    # apply a series of erosions and dilations to the mask using an elliptical kernel
    #This is to remove the small false-positive skin regions in the image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

    # blur the mask to help remove noise    
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    # then apply the mask to the frame
    skin = cv2.bitwise_and(img, img, mask = skinMask)


    return img , skin


def main():
    """
    """
    arg = parse_args()
    task = arg.task
    path = arg.path
    output = None
    
    if task=="skin detect":
        img, skin = skin_detect(path)

        # show the skin in the image along with the mask
        cv2.imshow("images", np.hstack([img, skin]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif task=="colorize":
        rgb_image = colorize(path)
        plt.imshow(rgb_image)
        

if __name__=="__main__":
    main()