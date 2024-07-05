import argparse
import queue
import sys
import threading
import time
import logging

import matplotlib.pyplot as plt
import numpy as np
import cv2

import roypy
from roypy_sample_utils import CameraOpener, add_camera_opener_options
from roypy_platform_utils import PlatformHelper

# Set up logging
logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s - %(levelname)s - %(message)s')

class MyListener(roypy.IDepthDataListener):

    def __init__(self, q):
        super(MyListener, self).__init__()
        self.frame = 0
        self.done = False
        self.undistortImage = True
        self.lock = threading.Lock()
        self.once = False
        self.queue = q
        self.last_time = time.time()

    def onNewData(self, data):
        self.frame += 1
        p = data.npoints()
        self.queue.put(p)
        print(f"Frame Number:  {self.frame}")

    def analyze_confidence(self, data):

        """
        Analyze and filter the confidence data.
        Args:
        data (numpy array): The complete data array including confidence as the last channel.
        """

        # Ensure data contains confidence channel
        if data.shape[2] <= 4:
            raise ValueError("Data does not contain a confidence channel at index 4")

        # Analyze confidence data for debugging
        confidence = data[:, :, 4].astype(float)
        
        return confidence

    def enhance_image(self, image):
        """
            Enhance the image data.
        """
        
        # Get the number of channels in the image
        channels = image.shape[2] if len(image.shape) > 2 else 1 
        # Apply CLAHE to enhance image
        clahe = cv2.createCLAHE(clipLimit = 4.0, tileGridSize = (8, 8))
        # Create an empty array to store the enhanced image
        enhanced = np.zeros_like(image)

        if channels == 1:
            enhanced = clahe.apply(image)
        else:
            # Apply CLAHE to each channel
            for i in range(3):
                enhanced[:, :, i] = clahe.apply(image[:, :, i])
        
        # Return the enhanced image
        return enhanced

    def filter_normalize_depth(self, depth, confidence, confidence_threshold = 0.95):
        """
        Filters and normalizes the depth image based on confidence values

        Parameters:
            - depth: The depth data array
            - confidence: The confidence data array
            - confidence_threshold: The threshold for confidence values to filter depth data.
        
        Returns:
            - depth_img: The filtered and normalized depth image
            - valid_depth: The filtered depth data
        """

        # Convert depth to a grayscale image, filtering out low confidence values
        confidence_threshold = 0.95 # Define a suitable confidence threshold
        confidence_scaled = confidence * 255 if confidence.max() <= 1.0 else confidence
        confidence_threshold_scaled = confidence_threshold * 255 if np.max(confidence) <= 1.0 else confidence_threshold

        # Convert depth to a grayscale image, filtering out low confidence values
        valid_depth = np.where(confidence_scaled > confidence_threshold_scaled , depth, 0)

        if np.any(valid_depth) > 0:
            depth_normalized = cv2.normalize(valid_depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_img = np.uint8(depth_normalized) 
        else:
            print("Depth data contains only zero values.")
            depth_img = np.zeros_like(valid_depth, dtype = np.uint8)
        
        return depth_img, valid_depth

    def process_extract_contours(self, depth_img, min_contour_area = 100):
        """
        Applies bilateral filtering, adaptive thresholding to filter, morphological operations, and find contours.

        Parameters:
            - depth_img: The depth image to be processed
            - min_contour_area: The minimum area threshold to filter out small contours.

        Return:
            - cleaned: The cleaned binary image after processing.
            - contours_filtered: List of filtered contours based on the minimum contour area.
        """

        # Apply bilateral filtering
        depth_filtered = cv2.bilateralFilter(depth_img, 9, 75, 75)

        # Apply adaptive thresholding
        thresholded = cv2.adaptiveThreshold(depth_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Optionally apply morphological operations to clean the image
        kernel = np.ones((5, 5), np.uint8) 
        cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out small contours
        min_contour_area = 1000 # Minimum contour area in pixels
        contours_filtered = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

        # Print some debugging information
        print("Number of contours found: ", len(contours))

        return cleaned, contours_filtered

    def paint (self, data):
        """
            Called in the main thread, with data containing one of the items that was added to the
            queue in onNewData.
        """

        # mutex to lock out changes to the distortion while drawing
        self.lock.acquire()

        try:

            # Ensure data contains RGB, depth, and confidence channels
            if data.shape[2] < 3:
                raise ValueError("Data does not contain enough channels for RGB, depth, and confidence")

            # Extract RGB, depth, and confidence data from the data array
            rgb = data[:, :, 0:3].astype(np.uint8)
            depth = data[:, :, 2] if data.shape[2] > 2 else None
            confidence = data[:, :, 4] if data.shape[2] > 4 else None 

            # Before processing, check if depth and confidence data are available and correct
            if depth is None or confidence is None:
                print("Depth or confidence data is missing")
                return # Exit the function if critical data is missing
            
            # Verify depth and confidence data dimensions match expected sizes
            if depth.shape != confidence.shape:
                print("Depth and confidence data dimensions do not match")
                return
        
            # Enhance the image
            enhanced_rgb = self.enhance_image(rgb)
            # Analyze confidence data for debugging
            self.analyze_confidence(data)

            # Filter and normalize depth image
            depth_img, valid_depth = self.filter_normalize_depth(depth, confidence)
            
            # Process and extract contours
            cleaned, contours_filtered = self.process_extract_contours(depth_img)
                
            # Convert the depth image to a color image
            depth_colored = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)

            # Use the depth image in grayscale as the background for the output image
            output_image = cv2.cvtColor(depth_img, cv2.COLOR_GRAY2BGR)

            # Cleaned the depth image on the color image
            cleaned_image = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        
            # Overlay the depth image on the color image
            overlay_image = cv2.addWeighted(enhanced_rgb, 0.6, depth_colored, 0.4, 0)            
            
            for i ,contour in enumerate(contours_filtered):
                
                # Draw contours on the output image
                self.process_contour(contour, i, valid_depth, output_image, overlay_image, cleaned_image)       

            # Display the results
            cv2.imshow('Depth Contours', output_image)
            cv2.imshow('Depth Cleaned', cleaned_image)
            cv2.imshow('Depth Overlay RGB', overlay_image)
            cv2.waitKey(1)  # Required for imshow to work properly
        
        except Exception as e:
            print("Failed to process data:", e)
        finally:
            self.lock.release()
            self.done = True
    
    def process_contour(self, contour, i, valid_depth, output_image, overlay_image, cleaned_image):
        """
        Processes each contour to draw bounding boxes and calculate depth values.

        Parameters:
            - contour: The contour to be processed.
            - valid_depth: The filtered depth data
            - output_image: The image on which the bounding boxes and contours will be drawn.
            - overlay_image: The image overlaying the depth image on the color image.
            - cleaned_image: The image with cleaned depth data.
        """

        # Draw the bounding box around the largest contour
        x, y, w, h = cv2.boundingRect(contour)

        contour_ratio = w / h if h != 0 else 0
        if not (0.5 < contour_ratio < 2):  # Filter out long, narrow, or very wide contours
            return

        cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(overlay_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 3)
        cv2.drawContours(cleaned_image, [contour], -1, (0, 255, 0), 3)

        area = cv2.contourArea(contour)

        # Get the depth values within the contour
        mask = np.zeros(valid_depth.shape, dtype = np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness = cv2.FILLED)
        contour_depth = valid_depth[mask == 255]

        # Filter out zero and extremely low depth values
        contour_depth = contour_depth[contour_depth > 0]
        if contour_depth.size == 0:
            print("No valid depth data within this contour")
            return

        # Calculate the average depth of the object
        object_distance_m = np.mean(contour_depth)

        # Print contour analysis results
        print(f"Contour {i + 1}:")
        print(f"  Area: {area} pixels")
        print(f"  Distance: {object_distance_m:.2f} m")

        # Print warnings based on object distance
        if object_distance_m <= 0.5:
            print("Imminent danger!!: Object is very close!!")
        elif 0.9 <= object_distance_m <= 1.0:
            print("Warning!!: Object is within 1 meter of the camera!")



    def setLensParameters(self, lensParameters):
        # Construct the camera matrix
        # (fx   0    cx)
        # (0    fy   cy)
        # (0    0    1 )
        self.cameraMatrix = np.zeros((3,3),np.float32)
        self.cameraMatrix[0,0] = lensParameters['fx']
        self.cameraMatrix[0,2] = lensParameters['cx']
        self.cameraMatrix[1,1] = lensParameters['fy']
        self.cameraMatrix[1,2] = lensParameters['cy']
        self.cameraMatrix[2,2] = 1

        # Construct the distortion coefficients
        # k1 k2 p1 p2 k3
        self.distortionCoefficients = np.zeros((1,5),np.float32)
        self.distortionCoefficients[0,0] = lensParameters['k1']
        self.distortionCoefficients[0,1] = lensParameters['k2']
        self.distortionCoefficients[0,2] = lensParameters['p1']
        self.distortionCoefficients[0,3] = lensParameters['p2']
        self.distortionCoefficients[0,4] = lensParameters['k3']

    def toggleUndistort(self):
        self.lock.acquire()
        self.undistortImage = not self.undistortImage
        self.lock.release()

    # Map the gray values from the camera to 0..255
    def adjustGrayValue(self,grayValue):
        clampedVal = min(400,grayValue) # try different values, to find the one that fits your environment best
        newGrayValue = clampedVal / 400 * 255
        return newGrayValue

def main ():
    # Set the available arguments
    platformhelper = PlatformHelper()
    parser = argparse.ArgumentParser (usage = __doc__)
    add_camera_opener_options (parser)
    options = parser.parse_args()
   
    opener = CameraOpener (options)

    try:
        cam = opener.open_camera ()
    except:
        print("could not open Camera Interface")
        sys.exit(1)

    try:
        # retrieve the interface that is available for recordings
        replay = cam.asReplay()
        print ("Using a recording")
        print ("Framecount : ", replay.frameCount())
        print ("File version : ", replay.getFileVersion())
    except SystemError:
        print ("Using a live camera")

    q = queue.Queue()
    l = MyListener(q)
    cam.registerDataListener(l)
    cam.startCapture()

    lensP = cam.getLensParameters()
    l.setLensParameters(lensP)

    process_event_queue (q, l)

    cam.stopCapture()
    print("Done")

def process_event_queue (q, painter):

    while True:
        try:
            # try to retrieve an item from the queue.
            # this will block until an item can be retrieved
            # or the timeout of 1 second is hit
           item = q.get(timeout = 1)
        except queue.Empty:
            # this will be thrown when the timeout is hit
            continue
        else:
            painter.paint(item) 
            # waitKey is required to use imshow, we wait for 1 millisecond
            currentKey = cv2.waitKey(1)
            """print(f"Current  key pressed: {currentKey}")""" #for debuging purposes
            if currentKey == ord('d'):
                painter.toggleUndistort()
            # close if escape key pressed
            if currentKey == 27: 
                break

if (__name__ == "__main__"):
    main()