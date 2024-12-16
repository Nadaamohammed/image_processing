#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

class ImageProcessorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Processor")
        self.geometry("900x900") 

        self.image_label = tk.Label(self)
        self.image_label.pack()
        self.image_label2 = tk.Label(self)
        self.image_label2.pack()
        
        self.upload_button = tk.Button(self, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()
        
        self.upload_button = tk.Button(self, text="Upload Image2", command=self.upload_image2)
        self.upload_button.pack()
        
        self.halftone_button = tk.Button(self, text="Image Color", command=self.image_gray)
        self.halftone_button.pack()
        
        self.upload_button = tk.Button(self, text="Threshold", command=self.threshold)
        self.upload_button.pack()

        self.halftone_button = tk.Button(self, text="Apply Halftone", command=self.apply_halftone)
        self.halftone_button.pack()
        
        self.halftone_button = tk.Button(self, text="Apply Advanced Halftone", command=self.Advanced_halftone)
        self.halftone_button.pack()
        
        self.halftone_button = tk.Button(self, text="Histogram", command=self.histogram)
        self.halftone_button.pack()
        
        self.halftone_button = tk.Button(self, text="Histogram Equalization", command=self.histogram_equalization)
        self.halftone_button.pack()
        
        self.low_pass_filter_button = tk.Button(
            self, text="low pass filter Image", command=self.low_pass_filter
        )
        self.low_pass_filter_button.pack()

        self.high_pass_filter_button = tk.Button(
            self, text="high pass filter Image", command=self.high_pass_filter
        )
        self.high_pass_filter_button.pack()

        self.median_filter_button = tk.Button(
            self, text="median filter Image", command=self.median_filter
        )
        self.median_filter_button.pack()

        self.adding_button = tk.Button(
            self, text="Adding Image", command=self.add
        )
        self.adding_button.pack()

        self.subtracting_button = tk.Button(
            self, text="Subtracting Image", command=self.subtract
        )
        self.subtracting_button.pack()

        self.inverting_button = tk.Button(
            self, text="Inverting Image", command=self.invert
        )
        self.inverting_button.pack()

        self.cut_paste_button = tk.Button(
            self, text="cut_paste Image", command=self.cut_paste
        )
        self.cut_paste_button.pack()

        
        self.manual_segmentation_button = tk.Button(self, text="Manual Segmentation", command=self.manual_segmentation)
        self.manual_segmentation_button.pack()

        self.histogram_peak_segmentation_button = tk.Button(
            self, text="Histogram Peak Segmentation", command=self.histogram_peak_segmentation
        )
        self.histogram_peak_segmentation_button.pack()

        self.histogram_valley_segmentation_button = tk.Button(
            self, text="Histogram Valley Segmentation", command=self.histogram_valley_segmentation
        )
        self.histogram_valley_segmentation_button.pack()

        self.histogram_adaptive_segmentation_button = tk.Button(
            self, text="Histogram Adaptive Segmentation", command=self.adaptive_segmentation
        )
        self.histogram_adaptive_segmentation_button.pack()
        
        

        self.image = None
        self.image2=None

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.image = cv2.resize(self.image, (300, 300))
            self.display_image(self.image)

    def upload_image2(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image2 = cv2.imread(file_path)
            self.image2 = cv2.resize(self.image2, (300, 300))
            self.display_image2(self.image2)
            
    def display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk
        self.image_label.pack(side=tk.LEFT, padx=10) 
        # الصورة الأولى على العمود الأول
    def display_image2(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk
        self.image_label2.pack(side=tk.RIGHT, padx=20)         

    
     
   ####################################################################  
    def image_gray(self):
        if self.image is None:
            messagebox.showerror("Error", "No image uploaded!")
            return
        gray_image = self.do_gray(self.image)
        self.display_image(gray_image)
    
    def do_gray(self, image):
        if len(image.shape) == 3:  # Image has 3 channels (color)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:  # Image is already grayscale
            gray_image = image
        return gray_image
      
    ##################################################################
    def threshold(self):
        if self.image is None:
            messagebox.showerror("Error", "No image uploaded!")
            return
        value=self.do_threshold(self.image)
        messagebox.showerror("The Threshold",int(value))
        
    
    def do_threshold(self,image):
        grayimg=self.do_gray(image)
        total_sum = 0
        pixel_count = 0
        img = np.array(grayimg)
        for row in img:
            for pixel in row:
                total_sum += pixel
                pixel_count += 1 
        threshold = total_sum / pixel_count
        return threshold
    #######################################################################
    def apply_halftone(self):
        if self.image is None:
            messagebox.showerror("Error", "No image uploaded!")
            return
        halftone_image = self.do_halftone(self.image)
        self.display_image(halftone_image)

    def do_halftone(self, image):       
        threshold = self.do_threshold(image)
        grayimage = self.do_gray(image)
        halftone = np.array(grayimage)
        for row in range(halftone.shape[0]):
            for col in range(halftone.shape[1]):
                if halftone[row,col] < threshold:
                    halftone[row,col] = 0
                else:
                    halftone[row,col] = 255   
        return halftone

     ###################################
    def Advanced_halftone (self):
        if self.image is None:
            messagebox.showerror("Error", "No image uploaded!")
            return
        threshold = self.do_threshold(self.image)
        grayimage=self.do_gray(self.image)
        img_array = np.array(grayimage)
        halftone_array = np.zeros_like(img_array)
        for row in range(img_array.shape[0]):
            for col in range(img_array.shape[1]):
                old_pixel = img_array[row, col]
                new_pixel = 255 if old_pixel > threshold else 0  
                halftone_array[row, col] = new_pixel
                error = old_pixel - new_pixel
                if col + 1 < img_array.shape[1]:
                    img_array[row, col + 1] += error * 7 / 16
                if col - 1 >= 0 and row + 1 < img_array.shape[0]:
                    img_array[row + 1, col - 1] += error * 3 / 16
                if row + 1 < img_array.shape[0]:
                    img_array[row + 1, col] += error * 5 / 16
                if col + 1 < img_array.shape[1] and row + 1 < img_array.shape[0]:
                    img_array[row + 1, col + 1] += error * 1 / 16
        
       
        self.display_image(halftone_array)
            
    #######################################################
    def histogram (self):
        if self.image is None:
            messagebox.showerror("Error", "No image uploaded!")
            return
        histogram=self.do_histogram(self.image)
        hist_img = np.zeros((300, 256), dtype=np.uint8) 
        max_value = max(histogram)
        for x in range(256):
            bar_height = int((histogram[x] / max_value) * 300) if max_value > 0 else 0
            for y in range(bar_height):
                hist_img[300 - 1 - y, x] = 255  
        self.display_image(hist_img)
        
        
    def do_histogram(self,image):
        grayimage=self.do_gray(image)
        arrayimage= np.array(grayimage)
        histogram = np.zeros(256, dtype=int)
        for x in range(arrayimage.shape[0]):
            for y in range(arrayimage.shape[1]):
                value = arrayimage[x, y]
                histogram[value] += 1 
        return histogram
    ###########################################################(error)
    def cumlative(self,hist):
        sum_of_hist = np.zeros(256, dtype=int)  # Array to store cumulative histogram
        running_sum = 0  # Running sum for cumulative distribution
        for i in range(256):
            running_sum += hist[i]
            sum_of_hist[i] = running_sum  # Ensure scalar assignment
        return sum_of_hist
    def histogram_equalization (self):
        gray_image=self.do_gray(self.image)
        hist = self.do_histogram(gray_image)  # Calculate histogram
        sum_of_hist = self.cumlative(hist)  # Calculate cumulative histogram
        total_pixels = gray_image.size  # Total number of pixels

    # Normalize the cumulative histogram to [0, 255]
        norm_cdf = (sum_of_hist * 255 / total_pixels).astype(np.uint8)

    # Map original pixels to equalized values
        equalized_image = np.zeros_like(image)
        for i in range(gray_image.shape[0]):
           for j in range(gray_image.shape[1]):
              equalized_image[i, j] = norm_cdf[image[i, j]]
        self.display_image(equalized_image)
            ###########################################################
    def low_pass_filter(self):#reduce noise and details, blur img
        #Checking if an image is loaded:
        if self.image is None:
            messagebox.showerror("Error", "No image uploaded!")
            return
            #Converting the image to grayscale:
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) 
        #creates a smoothing effect by averaging the pixel intensities.
        mask_3x3_low_pass=np.array([
            [0,1/6,0],
            [1/6,2/6,1/6],
            [0,1/6,0]
        ], dtype=np.float32)
        #cv2.filter2d applies convolution
        result= cv2.filter2D(gray_img, -1 , mask_3x3_low_pass)# -1 --> output image will have the same depth
        self.display_image(result)
        ###########################################################
    def high_pass_filter(self):
        if self.image is None:
            messagebox.showerror("Error", "No image uploaded!")
            return
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)    
        mask_3x3_high_pass=np.array([
            [0,-1,0],
            [-1,5,-1],
            [0,-1,0]
        ], dtype=np.float32)
        result= cv2.filter2D(gray_img, -1 , mask_3x3_high_pass)
        self.display_image(result)
        ###########################################################
    def median_filter(self):# replace the center point to median of the surrounding pixels , smooter img
        if self.image is None:
            messagebox.showerror("Error", "No image uploaded!")
            return
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) 
        #get dimentions of img
        height, width = gray_img.shape
        #output array
        filtered_image = np.zeros_like(gray_img)
        # loop to points accept border pixels (to avoid out-of-bound)
        for i in range(1,height -1):
            for j in range(1, width -1):
                #extract 3x3 block of pixel
                neightborhood = gray_img[i-1:i+2 , j-1:j+2]
                median_value=np.median(neightborhood)
                filtered_image[i,j] = median_value
                #noise reduction , edge prevention 
        self.display_image(filtered_image)
        ###########################################################
    def add(self):
    # Ensure the images have the same shape
        if self.image is None:
            messagebox.showerror("Error", "No image uploaded!")
            return
        if self.image2 is None:
            messagebox.showerror("Error", "No image uploaded!")
            return    
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray_img2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)
        #get the dimentions
        height, width = gray_img.shape
        #unit8--> to ensure the pixel value stay in valid range[0,255]
        added_image = np.zeros((height, width), dtype=np.uint8)
        for i in range(height):
          for j in range(width):
             value = gray_img[i, j] + gray_img2[i, j]
             added_image[i, j] = max(0, min(int(value), 255))  # Clipping to [0, 255]
        self.display_image(added_image)
     ###########################################################
    def subtract(self):
    # Ensure the images have the same shape
        if self.image is None:
            messagebox.showerror("Error", "No image uploaded!")
            return
        if self.image2 is None:
            messagebox.showerror("Error", "No image uploaded!")
            return    
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray_img2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)
        height, width = gray_img.shape
        subtracted_image = np.zeros((height, width), dtype=np.uint8)
        for i in range(height):
          for j in range(width):
             value = gray_img[i, j] - gray_img2[i, j]
             subtracted_image[i, j] = max(0, min(int(value), 255))  # Clipping to [0, 255]
        self.display_image(subtracted_image) 
         ###########################################################
    def invert(self):
    # Ensure the images have the same shape
        if self.image is None:
            messagebox.showerror("Error", "No image uploaded!")
            return
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        inverted_image = 255 - gray_img
        self.display_image(inverted_image)
           ###########################################################
    def cut_paste(self):#useful for composite image creation and experimenting with image processing results
        if self.image is None:
            messagebox.showerror("Error", "No image uploaded!")
            return
        if self.image2 is None:
            messagebox.showerror("Error", "No image uploaded!")
            return    
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray_img2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)
        x, y = 50,50 #position
        w, h = 100,100 #size
        h1, w1 = gray_img.shape
        h2, w2 = gray_img2.shape
        #validating the region
        if x + w > gray_img.shape[1] or y + h > gray_img.shape[0]:
            messagebox.showerror("Error", "Cut region exceeds image 1 dimensions!")
            return
        if x + w > gray_img2.shape[1] or y + h > gray_img2.shape[0]:
            messagebox.showerror("Error", "Paste region exceeds image 2 dimensions!")
            return

    # Cut the region from image1
        cut_image = gray_img[y:y+h, x:x+w]

    # Copy image2 and paste the cut region onto it
        output_image = np.copy(gray_img2)
        output_image[y:y+h, x:x+w] = cut_image

        self.display_image(output_image)
    


     ###########################################################
    def manual_segmentation(self):#Suitable for fine-tuning but time-consuming.
        if self.image is None:
            messagebox.showerror("Error", "No image uploaded!")
            return
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        #define the thresholdind
        low_threshold = 50
        high_threshold = 150
        #applay thresolding
        segmented_img = np.where((gray_img >= low_threshold) & (gray_img <= high_threshold), 255, 0).astype(np.uint8)
        self.display_image(segmented_img)

    ###########################################################(error)  
    #1-compute histogram, 2-find peaks,3-sort peaks,4-select background , object , 5-cal threshold,6-apply threshold
    #choose the midpoint
    from scipy.signal import find_peaks
    def find_histogram_peaks(self, hist):
        peaks = find_peaks(hist)  # find_peaks function
        # Sort peaks by the height of the histogram
        stored_peaks = sorted(peaks, key=lambda x: hist[x], reverse=True)
        return stored_peaks[:2]  # Keep the highest 2 peaks

    def calculate_threshold(self, peaks_indices, hist):
        peak1 = peaks_indices[0]
        peak2 = peaks_indices[1]
        low_threshold = (peak1 + peak2) / 2  # Midpoint
        high_threshold = peak2  # Set second peak as high threshold (object)
        return low_threshold, high_threshold

    def histogram_peak_segmentation(self):
        if self.image is None:
            messagebox.showerror("Error", "No image uploaded!")
            return

        # Convert image to grayscale
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Calculate the histogram
        hist = self.do_histogram(gray_img)

        # Find peaks in the histogram
        peaks_indices = self.find_histogram_peaks(hist)

        # Calculate the low and high thresholds based on the peaks
        low_threshold, high_threshold = self.calculate_threshold(peaks_indices, hist)

        # Create a segmented image
        segmented_image = np.zeros_like(gray_img)
        segmented_image[(gray_img >= low_threshold) & (gray_img <= high_threshold)] = 255  # Set to white (255)

        # Display the segmented image (implement this method to display the result)
        self.display_image(segmented_image)
    ###########################################################(error)
    #1-calculate histo,2-detect peaks,3-sort peaks,4-find the point,5-segment the img
    #choose the lowestpoint
    def histogram_valley_segmentation(self):
         # Step 1: Ensure an image is loaded
        if self.image is None:
           messagebox.showerror("Error", "No image uploaded!")
           return

    # Step 2: Convert the image to grayscale
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    # Step 3: Calculate the histogram
        hist = self.do_histogram(gray_img)
        if hist is None or len(hist) == 0:
           messagebox.showerror("Error", "Failed to compute histogram!")
           return

    # Step 4: Detect peaks in the histogram
        peaks = find_peaks(hist)
        if len(peaks) < 2:
           messagebox.showerror("Error", "Not enough peaks found!")
           return

    # Step 5: Sort peaks and find the valley point
        sorted_peaks = sorted(peaks, key=lambda x: hist[x], reverse=True)
        peak1, peak2 = sorted(sorted_peaks[:2])  # Take the two most prominent peaks
        valley = np.argmin(hist[peak1:peak2]) + peak1  # Find the lowest point (valley) between peaks

    # Segment the image based on the valley
        segmented_img = np.where(gray_img <= valley, 0, 255).astype(np.uint8)

    # Display the segmented image
        try:
           self.display_image(segmented_img)
        except Exception as e:
           messagebox.showerror("Error", f"Failed to display image: {str(e)}")
        self.display_image(segmented_img)
    ###########################################################(error)
        #1-calculate histo,2-detect peaks,3-sort peaks,4-findthe vally point,5-first-pass segmentation,6-calculate mean, 
    #7-adjust the threshold, 8-second-pass segmentation
    from scipy.signal import find_peaks
    def adaptive_segmentation(self):
    # Step 1: Ensure an image is loaded
        if self.image is None:
           messagebox.showerror("Error", "No image uploaded!")
           return

    # Step 2: Convert the image to grayscale
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    # Step 3: Calculate the histogram
        hist =self.do_histogram(gray_img)
        hist = hist.flatten()
        if hist is None or len(hist) == 0:
           messagebox.showerror("Error", "Failed to compute histogram!")
           return

    # Step 4: Detect peaks in the histogram
        peaks, _ = self.find_peaks(hist)
        if len(peaks) < 2:
           messagebox.showerror("Error", "Not enough peaks found!")
           return

    # Step 5: Sort peaks by intensity value and find the valley point
        sorted_peaks = sorted(peaks, key=lambda x: hist[x], reverse=True)
        peak1, peak2 = sorted(sorted_peaks[:2])  # Take the two most prominent peaks
        valley = np.argmin(hist[peak1:peak2]) + peak1  # Index of the minimum value between the peaks

    # First-pass segmentation
        segmented_img = np.where(
        (gray_img >= peak1) & (gray_img <= valley), 255, 0
         ).astype(np.uint8)

    # Step 6: Calculate mean intensity for object and background pixels
        object_pixels = gray_img[segmented_img == 255]
        background_pixels = gray_img[segmented_img == 0]

        object_mean = object_pixels.mean() if object_pixels.size > 0 else 0
        background_mean = background_pixels.mean() if background_pixels.size > 0 else 0

    # Step 7: Adjust the threshold using calculated means
        new_low_threshold, new_high_threshold = sorted([object_mean, background_mean])

    # Step 8: Second-pass segmentation
        final_segmented_img = np.where(
        (gray_img >= new_low_threshold) & (gray_img <= new_high_threshold), 255, 0
    ).astype(np.uint8)

    # Display the final segmented image
        try:
           self.display_image(final_segmented_img)
        except Exception as e:
           messagebox.showerror("Error", f"Failed to display image: {str(e)}")   
        self.display_image(final_segmented_img)    

    ###########################################################
   
        




if __name__ == "__main__":
    app = ImageProcessorApp()
    app.mainloop()


# In[ ]:





# In[ ]:




