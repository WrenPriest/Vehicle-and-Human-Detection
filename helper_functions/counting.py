def counting(box_centers, classes, width, counter):
    is_vehicle_detected = False
    print(classes)
    for i in range(len(box_centers)):
        #print(classes[i])
        if classes[i] in [3,6,8]:
            if box_centers[i] in range(int(width / 2 - .02 * width),
                                       int(width / 2 + .02 * width)):
                counter += 1
                is_vehicle_detected = True
            else:
                is_vehicle_detected = False
        elif classes[i] == 1:
            if box_centers[i] in range(int(width / 2 - .003 * width),
                                       int(width / 2 + .003 * width)):
                counter += 1
                is_vehicle_detected = True
            else:
                is_vehicle_detected = False

    return counter, is_vehicle_detected
