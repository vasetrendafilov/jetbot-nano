#!/usr/bin/env python3
import os
import cv2
import rospy
import numpy as np
from my_package import Duckiebot
from my_package import grayscale,gaussian_blur,canny,hough_lines,weighted_img,draw_car_and_target_direction,calc_angle,get_x

def process_image(img):
    dec_img = duckie.decode_image(img)
    if duckie.img_count > 20:
        undistorted_img = cv2.remap(dec_img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[120:,:]
        gray_img = grayscale(undistorted_img)
        canny_img = canny(gray_img, 60, 60*2)
        lines_canny_img,pot_lines = hough_lines(canny_img, 3, np.pi/180, 50, 10, 5)
        rospy.loginfo("%s", str(pot_lines))
        if pot_lines[0] and pot_lines[1]:
            final_img,current_line,target_line = draw_car_and_target_direction(weighted_img(lines_canny_img, undistorted_img),pot_lines,'left')
            angle = calc_angle(current_line,target_line)
            #rospy.loginfo("Angle is %s", str(angle))
            turn = 0.05 if angle > 0 else -0.05
            turn = 0 if abs(angle) < 0.5 else turn
            duckie.publish_wheel_cmd(0.3-turn, 0.3+turn)
        else:
            final_img = weighted_img(lines_canny_img, undistorted_img)
            duckie.publish_wheel_cmd(0.0, 0.0)
        #duckie.save_image('turn1',dec_img)
if __name__ == '__main__':
    # create the node
    duckie = Duckiebot(node_name='my_node')
    # connect camera
    DIM=(408, 308)
    K=np.array([[200.45528254856666, 0.0, 193.58838229266124], [0.0, 204.98105125775646, 136.8997130353786], [0.0, 0.0, 1.0]])
    D=np.array([[-0.051386452545095], [0.18476371756060023], [-0.5729487022183332], [0.49089690391610674]])
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    duckie.connect_camera(process_image)
    # test wheels
    duckie.publish_car_cmd(0,0)
    # keep spinning
    rospy.spin()