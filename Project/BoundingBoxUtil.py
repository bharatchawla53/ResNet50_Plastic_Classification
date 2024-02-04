import cv2
import os
import sys

def main(argv):

    plastic_categories = ['heavy_plastic', 'no_image', 'no_plastic', 'some_plastic']
    class_index = 0
    directoryPath = 'data/train/'
    filteredDataDirectoryPath = 'FilteredData/train/'

    for pc in plastic_categories:
       for filename in os.listdir(directoryPath + pc):
            img = cv2.imread(os.path.join(directoryPath + pc, filename))
            if img is not None:

                # check if the image corners are white
                rows, cols, _ = img.shape

                if (all(img[0][0] == (255, 255, 255)) 
                    & all(img[0][cols-1] == (255, 255, 255)) 
                    & all(img[rows-1][0] == (255, 255, 255)) 
                    & all(img[rows-1][cols-1] == (255, 255, 255))):

                    # calculate bounding box
                    bin = cv2.inRange(img, (255, 255, 255), (255, 255,255))
                    cv2.bitwise_not(bin, bin)

                    # find the contours
                    contours, _ = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    # sort contours in decreasing order of area
                    cnt = sorted(contours, key = cv2.contourArea, reverse = True)

                    # take the first contour
                    firstCnt = cnt[0]

                    # compute the bounding rectangle of the contour
                    x, y, w, h = cv2.boundingRect(firstCnt)

                    # compute the center of the bounding rectangle
                    x_center = (x + w) / 2
                    y_center = (y + h) / 2

                    # label parameters
                    p1 = round(x_center/cols, 6)
                    p2 = round(y_center/rows, 6)
                    p3 = round(w/cols, 6)
                    p4 = round(h/rows, 6)

                    # write to a file with bounding box details and img name
                    with open(filteredDataDirectoryPath + pc + '/labels/' + filename.split(".")[0] + '.txt', 'a') as file:
                        #str_to_write = "%d %d %d %d %d" % (class_index, {round(x_center/cols, 6):.6f}, round(y_center/rows, 6), round(w/cols, 6), round(h/rows, 6))
                        #file.write(str_to_write)

                        file.write(f"{class_index} {p1:.6f} {p2:.6f} {p3:.6f} {p4:.6f}")

                    # save this image to a new filtered data directory for yolo processing 
                    cv2.imwrite(filteredDataDirectoryPath + pc + '/images/' + filename, img)

       class_index = class_index + 1

    return

if __name__ == "__main__":
    main(sys.argv)

