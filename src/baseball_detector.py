import cv2 as cv
import numpy as np
import time 


class BaseballDetector:
    def __init__(self,
                 image_height=480,
                 image_width=640,
                 grayscale=True,
                 display=True) -> None:
        self.prev_image = np.zeros((image_height, image_width), dtype=np.uint8)
        self.incoming_in_gray = grayscale
        self.display = display
        self.plot_img = None
        self.x_bound = 0
        self.y_bound = 0
        self.w_bound = 0
        self.h_bound = 0
        self.first_detect = False
        self.background = np.zeros((image_height, image_width), dtype=np.uint8)

        params = cv.SimpleBlobDetector_Params()
        params.filterByCircularity = True
        params.minCircularity = 0.8
        self.save_counter = 0

        self.blob_detector = cv.SimpleBlobDetector_create(params)

    def _get_ROI_fast(self, img):
        #  gaussian blur diff image
        blur = cv.medianBlur(img, 9)



    def _get_ROI(self, img, profile=False):
        if profile:
            start_total = time.time()
        # print(img.shape)
        # print(self.prev_image.shape)

        # img comes in with background removed

        # if profile:
        #     start_blur = time.time()
        # # Gaussian blur diff image
        # blur = cv.medianBlur(img, 9)
        # if profile:
        #     end_blur = time.time()
        #     print("Gaussian Blur Time:", (end_blur - start_blur) * 1000, "milliseconds")

        blur = img
        # use time history to only look in region of interest so that thresholds can be low.
        time_diff = cv.absdiff(blur, self.prev_image)

        # threshold blur on about grayscale of 5ish
        ret, thresh = cv.threshold(time_diff, 15, 255, cv.THRESH_BINARY)

        # erosion to remove noise and fill in gaps
        kernel = np.ones((15, 15), np.uint8)
        dilated = cv.dilate(thresh, kernel, iterations=3)
        eroded = cv.erode(dilated, kernel, iterations=3)

        # find contour with largest bounding box
        if profile:
            start_contour = time.time()
        contours, hierarchy = cv.findContours(eroded, cv.RETR_TREE,
                                            cv.CHAIN_APPROX_SIMPLE)
        if profile:
            end_contour = time.time()
            print("Contour Detection Time:", (end_contour - start_contour) * 1000, "milliseconds")

        # get max area bounding box
        max_area_idx = 0
        if len(contours) > 0:
            for i, contour in enumerate(contours):
                area = cv.contourArea(contour)
                if area > cv.contourArea(contours[max_area_idx]):
                    max_area_idx = i
            # draw bounding box
            self.x_bound, self.y_bound, self.w_bound, self.h_bound = cv.boundingRect(
                contours[max_area_idx])
            self.w_bound = int(self.w_bound * 1.5)
            self.h_bound = int(self.h_bound * 1.5)

            if self.display:
                # Define the text and position
                text = "ROI"
                position = (self.x_bound, self.y_bound - 5
                            )  # Position the text above the rectangle

                # Add the text
                cv.putText(self.plot_img, text, position,
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv.rectangle(
                    self.plot_img, (self.x_bound, self.y_bound),
                    (self.x_bound + self.w_bound, self.y_bound + self.h_bound),
                    (255, 0, 0), 2)

            # crop everything inside of bounding box
            crop_img = blur[self.y_bound:self.y_bound + self.h_bound,
                            self.x_bound:self.x_bound + self.w_bound]
            # threshold on cropped image
            ret, crop_img = cv.threshold(crop_img, 7, 255, cv.THRESH_BINARY)

        else:
            crop_img = None

        self.prev_image = blur

        if profile:
            end_total = time.time()
            print("Total Processing Time:", (end_total - start_total) * 1000, "milliseconds")

        return blur, thresh, eroded, crop_img, time_diff



    def _get_hough_circle(self, img):

        if img is not None:
            #find circles
            circles = cv.HoughCircles(img,
                                      cv.HOUGH_GRADIENT,
                                      1,
                                      1000,
                                      param1=250,
                                      param2=1,
                                      minRadius=5,
                                      maxRadius=30)
            #
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (circ_x, circ_y, circ_r) in circles:
                    x = self.x_bound + circ_x
                    y = self.y_bound + circ_y
                    #onyl plot circles that are within the bounding box
                    if self.display:
                        cv.circle(self.plot_img, (x, y), circ_r, (0, 255, 0), 1)
                        cv.rectangle(self.plot_img, (x - 2, y - 2),
                                     (x + 2, y + 2), (0, 0, 255), -1)
                return (x, y)
            
        return None, None

    def _get_blob_circle(self, img):
        raise NotImplementedError

    def _remove_background(self, img):
        return cv.absdiff(img, self.background)

    def detect(self, img, display=False, profile=False):
        self.plot_img = img.copy()
        if not self.incoming_in_gray:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        if not self.first_detect:
            self.background = img.copy()
            self.first_detect = True

        start = time.time()
        img = self._remove_background(img)
        
        if profile:
            print(f"Remove background: {(time.time()-start)*1000} ms")
            start = time.time()

        blur, thresh, eroded, crop_img, time_diff = self._get_ROI(img, profile=profile)

        if profile:
            print(f"get ROI: {(time.time()-start)*1000} ms")
            start = time.time()

        x_loc, y_loc = self._get_hough_circle(crop_img)

        if profile:
            print(f"get circle: {(time.time()-start)*1000} ms")
            start = time.time()

        if self.display:
            cv.imshow('raw', self.plot_img)
            cv.imshow('blur', blur)
            cv.imshow('thresh', thresh)
            cv.imshow('dilated and eroded', eroded)
            if crop_img is not None:
                cv.imshow('cropped', crop_img)
            cv.imshow('time diff', time_diff)
            # cv.imshow('background', self.background)
            # cv.imshow('circle', circle)
            key = cv.waitKey(1)

            if key == ord('q'):
                cv.destroyAllWindows()
                raise SystemExit
            # elif key == ord('s'):
            #     cv.imwrite(f'{self.save_counter}.png', self.plot_img)
            #     self.save_counter += 1

        return x_loc, y_loc
