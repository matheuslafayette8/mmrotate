import xml.etree.ElementTree as ET
import math
import glob
import numpy as np
import cv2

def rotate_point(x, y, cx, cy, angle):
    # Rotate point (x, y) around center (cx, cy) by angle (in radians)
    x_rot = math.cos(angle) * (x - cx) - math.sin(angle) * (y - cy) + cx
    y_rot = math.sin(angle) * (x - cx) + math.cos(angle) * (y - cy) + cy
    return x_rot, y_rot

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # open image
    image_path = xml_file.replace("labels", "imgs").replace(".xml", ".png")
    image = cv2.imread(image_path)

    # resize image to 800x800
    im_original_shape = image.shape
    im_target_shape = (300, 300)
    image = cv2.resize(image, im_target_shape)

    image_cpy = image.copy()

    bounding_boxes = []

    for obj in root.findall('object'):
        name = obj.find('name').text
        robndbox = obj.find('robndbox')
        points = obj.find('points').text

        points = points.replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace(',', '').split(' ')

        points = np.array(points, dtype=np.float32).reshape(-1, 2)
        # rescale points to target image shape
        points[:, 0] = points[:, 0] / im_original_shape[1] * im_target_shape[1]
        points[:, 1] = points[:, 1] / im_original_shape[0] * im_target_shape[0]

        # import pdb; pdb.set_trace()

        x1, y1 = points[0]
        x2, y2 = points[1]
        x3, y3 = points[2]
        x4, y4 = points[3]

        # draw poligon on image
        cv2.polylines(image, [points.astype(np.int32)], True, (0, 0, 255), 2)

        class_name = "parking_spot"
        difficult = 0
        bounding_boxes.append(([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], class_name, difficult))

    # show image
    #cv2.imshow("image", image)
    #cv2.waitKey(0)

    # write image 
    output_image_path = xml_file.replace(input_xml_folder, output_txt_folder).replace(".xml", ".png")
    cv2.imwrite(output_image_path, image_cpy)

    return bounding_boxes

def write_to_txt(bounding_boxes, output_file):
    with open(output_file, 'w') as f:
        #f.write("imagesource:Test\n")
        #f.write("gsd:0.00\n")

        for corners, class_name, difficult in bounding_boxes:

            # save line using format x1, y1, x2, y2, x3, y3, x4, y4, category, difficult
            line = ""
            for corner in corners:
                line += str(round(corner[0],1) )+ " " + str(round(corner[1],1)) + " "
            line += class_name + " " + str(difficult) + "\n"
            f.write(line)


if __name__ == "__main__":
    dataset_name = "15_01"
    input_xml_folder = "dataset/parking_spaces/complete_datasets/" + dataset_name + "/labels/"
    input_image_folder = "dataset/parking_spaces/complete_datasets/" + dataset_name + "/imgs/"
    output_txt_folder = "dataset/parking_spaces/dota/" + dataset_name + "/"

    #cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    # iterate on the input_xml_folder
    for input_xml_file in glob.glob(input_xml_folder + "*.xml"):
        try:
            bounding_boxes = parse_xml(input_xml_file)
        except:
            print(input_xml_file)

        output_txt_file = input_xml_file.replace(input_xml_folder, output_txt_folder).replace(".xml", ".txt")
        write_to_txt(bounding_boxes, output_txt_file)