import os
import numpy as np

path = 'QuickDraw414k/coordinate_files//test'
samples = os.listdir(path)
total = 0

print(samples)
exit(0)

from PIL import Image
import cv2

def draw_three(tsketch, random_color=False, img_size=224, stroke_flag=1):
    color_idx = 0
    thickness = int(img_size * 0.025)
    sketch = tsketch.copy()

    sketch[:, 0:2] = sketch[:, 0:2] * img_size / 256 + thickness

    canvas = np.ones((img_size + 3 * (thickness + 1), img_size + 3 * (thickness + 1), 3), dtype='uint8') * 255


    color = (0, 0, 0)
    pen_now = np.array([sketch[0, 0], sketch[0, 1]])
    first_zero = False

    for stroke in sketch:
        delta_x_y = stroke[0:0 + 2] - pen_now
        state = stroke[-1:]

        if int(state) == -1:
            break
        if first_zero:  # 首个零是偏移量, 不画
            pen_now += delta_x_y
            first_zero = False
            continue


        cv2.line(canvas, tuple([int(pen_now[0]), int(pen_now[1])]), tuple([int((pen_now + delta_x_y)[0]),int((pen_now + delta_x_y)[1])]), color, thickness=thickness)
        #cv2.line(canvas1, tuple(pen_now), tuple(pen_now + delta_x_y), color, thickness=1)
        #cv2.line(canvas2, tuple(pen_now), tuple(pen_now + delta_x_y), color, thickness=10)


        if int(state) == stroke_flag:  # next stroke
            first_zero = True

        pen_now += delta_x_y
    #canvas[:,:,0] = canvas1[:,:,0]
    #canvas[:,:,2] = canvas2[:,:,2]
    return Image.fromarray(cv2.resize(canvas, (img_size, img_size)))

data = np.load('QuickDraw414k/coordinate_files/test/aircraft_carrier/aircraft_carrier_2.npy')
image = draw_three(data, img_size=224)
image.save('aircraft_carrier_2.png')