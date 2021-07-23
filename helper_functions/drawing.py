def draw_roi(current_frame):
    if ROI_Line == 'vertical':
        roi_start = (int(width/2), 0)
        rot_end = (int(width/2), height)
    else:
        roi_start = (0, int(height/2))
        rot_end = (width, int(height/2))

    if is_vehicle_detected is True:
        cv2.line(current_frame, roi_start, roi_end, (0, 0, 0xFF), 5)
    else:
        cv2.line(current_frame, roi_start, roi_end, (0, 0xFF, 0), 5)