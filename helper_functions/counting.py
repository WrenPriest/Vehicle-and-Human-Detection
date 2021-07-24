def counting(box_centers, classes, width, counter):
    is_vehicle_detected = False
    for i in range(len(box_centers)):
        if classes[i] == 3 or 6 or 8:
            if box_centers[i] in range(int(width / 2 - .02 * width),
                                       int(width / 2 + .02 * width)):
                counter += 1
                is_vehicle_detected = True
            else:
                is_vehicle_detected = False
    return counter, is_vehicle_detected
