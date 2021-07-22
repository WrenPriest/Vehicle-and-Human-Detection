#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 27th January 2018
# ----------------------------------------------
from numpy import isin
from utils.image_utils import image_saver

is_vehicle_detected = [0]
current_frame_number_list = [0]
current_frame_number_list_2 = [0]
position_of_detected_vehicle = [0]


def predict_speed(
    top,
    bottom,
    right,
    left,
    current_frame_number,
    crop_img,
    roi_info
    ):
    update_csv = False
    if roi_info['direction'] == 'top_down':
        position = bottom
    elif roi_info['direction'] == 'bottom_up':
        position = top
    elif roi_info['direction'] == 'left_right':
        position = right
    else:
        position = left

    if len(position_of_detected_vehicle) != 0 \
        and roi_info['position'] - 5 < position_of_detected_vehicle[0] \
        and (current_frame_number - current_frame_number_list_2[0])>24:
        
        if (roi_info['direction'] in ['bottom_up', 'right_left'] and position - position_of_detected_vehicle[0] <= 0)\
           or (roi_info['direction'] in ['top_down', 'left_right'] and position - position_of_detected_vehicle[0] > 0):
            is_vehicle_detected.insert(0, 1)
            update_csv = True
            image_saver.save_image(crop_img)  # save detected vehicle image
            current_frame_number_list_2.insert(0, current_frame_number)
    # for debugging
    # print("bottom_position_of_detected_vehicle[0]: " + str(bottom_position_of_detected_vehicle[0]))
    # print("bottom: " + str(bottom))

    position_of_detected_vehicle.insert(0, position)
    return (is_vehicle_detected, update_csv)
