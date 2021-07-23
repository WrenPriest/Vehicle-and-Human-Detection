# <center> Vehicle-and-Human-Detection

## General Approach

Use a cv2 while ret is True loop to process entire video
- call object detection function (creates bounding boxes)
- call counter function (check if bounding box intersects ROI)
- call drawing functions (ROI line, bounding boxes, counter text)
- return output video

### Object Detection Function

- Import SSD_MobileNet_V1
- Create label dictionary for converting detection classes to label strings
- Output vertices for every detected vehicle or person *(for Drawing Functions)*
- Compute and store average/center of two vertices *(for Counter Function)*
    - use top-left and top-right vertices for vertical ROI
    - use top-right and bottom-right vertices for horizontal ROI

### Counter Function

Setting a ROI range that is dynamic based on width of the video should be able to account for faster
vehicles and flickering bounding boxes. Center of bounding boxes will be used since the center of
bounding boxes should vary less than vertices. Videos should all be 30 FPS but range may need to be
adjusted depending on speed of vehicles or pedestrians. Since pedestrians are slower than vehicles,
pedestrians may require a smaller ROI to avoid double counting.

if box_center is within a certain range of the width +/- 2% of width then increase counter by 1

Will also return is_vehicle_detected as boolean *(for Drawing Functions)*

```
if box_center in range(int(width/2 - .02 * width),
                       int(width/2 + .02 * width)):
    counter += 1
    is_vehicle_detected = True
else:
    is_vehicle_detected = False
```
### Drawing Functions

#### ROI Line

Will have a conditional for a vertical line or horizontal line.
If is_vehicle_detected is True then ROI line will be green, otherwise it will be red.


```
if ROI_Line == 'vertical':
    roi_start = (int(width/2), 0)
    rot_end = (int(width/2), height)
else:
    roi_start = (0, int(height/2))
    rot_end = (width, int(height/2))

if is_vehicle_detected is True:
    cv2.line(input_frame, roi_start, roi_end, (0, 0, 0xFF), 5)
else:
    cv2.line(input_frame, roi_start, roi_end, (0, 0xFF, 0), 5)
```

#### Bounding Boxes

Investigate implementation in reference API and decide whether to use
Pillow (reference API) or cv2.rectangle (new implementation from scratch)

```
# Actual detection
(boxes, scores, classes, num) = \
    sess.run([detection_boxes, detection_scores,
             detection_classes, num_detections],
             feed_dict={image_tensor: image_np_expanded})

# Visualization of the results of a detection.
(counter, csv_line) = \
    vis_util.visualize_boxes_and_labels_on_image_array(
    cap.get(1),
    input_frame,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    roi_info,
    use_normalized_coordinates=True,
    line_thickness=4,
    )
```




#### Counter Text

Could be fine-tuned to be more dynamic and not be set to precise pixels

```
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(
    input_frame,
    'Detected Vehicles: ' + str(counter),
    (10, 35),
    font,
    0.8,
    (0, 0xFF, 0xFF),
    2,
    cv2.FONT_HERSHEY_SIMPLEX,
    )
 ```
