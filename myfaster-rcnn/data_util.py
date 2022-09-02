import numpy as np


# def image_preprocessing(images):
#     return cv2.resize(images, (800, 800))

def generate_anchor():
    feature_size = 800 // 16
    
    x = np.linspace(16, feature_size * 16, 50)
    y = np.linspace(16, feature_size * 16, 50)
    X, Y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()

    ratios = [0.5, 1, 2]
    scales = [8, 16, 32]
    sub_sample = 16

    anchor_boxes = np.zeros(((feature_size * feature_size * 9), 4))
    index = 0

    for c_y, c_x in zip(Y, X):               
        for i in range(len(ratios)):     
            for j in range(len(scales)): 
                
                h = sub_sample * scales[j] * np.sqrt(ratios[i])
                w = sub_sample * scales[j] * np.sqrt(1./ ratios[i])
                
                anchor_boxes[index, 1] = c_y - h / 2.
                anchor_boxes[index, 0] = c_x - w / 2.
                anchor_boxes[index, 3] = c_y + h / 2.
                anchor_boxes[index, 2] = c_x + w / 2.
                index += 1
    index_inside = np.where(
        (anchor_boxes[:, 0] >= 0) &
        (anchor_boxes[:, 1] >= 0) &
        (anchor_boxes[:, 2] <= 800) &
        (anchor_boxes[:, 3] <= 800))[0]
    print(index_inside[100])
    return anchor_boxes, index_inside