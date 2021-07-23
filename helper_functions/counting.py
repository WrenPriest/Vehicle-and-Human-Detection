def counting(box_centers, width):
    counter = 0
    for box_center in box_centers:
        if box_center in range(int(width / 2 - .02 * width),
                               int(width / 2 + .02 * width)):
            counter += 1
            global is_vehicle_detected = True
        else:
            global is_vehicle_detected = False
    return counter
