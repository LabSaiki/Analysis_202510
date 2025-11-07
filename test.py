import cv2
import numpy as np
from matplotlib import pyplot as plt

def array_append(target, new_array, axis=0):
    if target is None:
        return new_array
    else:
        return  np.append(target, new_array, axis=axis)

def piv(
        input_name,
        output_dir,
        window_size=32,
        overlap=16,
        vector_threshold=0):
    # Load images
    input_path = input_name
    caputure = cv2.VideoCapture(input_path)

    ret, pre_frame = caputure.read()
    vector_lengths = []
    vector_infos = None
    correlation_coeffs = []

    add_vector_frames = []


    forcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_dir,
        forcc,
        fps=10,
        frameSize=(pre_frame.shape[1], pre_frame.shape[0])
    )
    pre_frame = None
    i = 0
    while True:
        i += 1
        print("start ", i)
        ret, frame = caputure.read()
        if not ret or i > 500:
            print("failed to read frame", i)
            caputure.release()
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        threshold_value, frame = cv2.threshold(
            frame, 70, 255, cv2.THRESH_BINARY)
        if pre_frame is None:
            pre_frame = frame
            continue

        copy_pre_frame = pre_frame.copy()
        copy_pre_frame = cv2.cvtColor(copy_pre_frame, cv2.COLOR_GRAY2BGR)
        # Calculate PIV
        h, w = frame.shape

        h_steps = int(h / (window_size - overlap))
        w_steps = int(w / (window_size - overlap))

        frame_vector_infos = None
        for h_step in range(h_steps - 1):
            for w_step in range(w_steps - 1):
                # define window to set template
                win_h_start = h_step * (window_size - overlap)
                win_h_end = win_h_start + window_size
                win_w_start = w_step * (window_size - overlap)
                win_w_end = win_w_start + window_size

                # set template
                template = pre_frame[win_h_start:win_h_end,win_w_start:win_w_end]

                method = cv2.TM_CCOEFF_NORMED
                res = cv2.matchTemplate(frame, template, method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                # view translated vectors
                win_h_center = win_h_start + int(window_size / 2)
                win_w_center = win_w_start + int(window_size / 2)
                after_h_center = max_loc[1] + int(window_size / 2)
                after_w_center = max_loc[0] + int(window_size / 2)

                dx = after_w_center - win_w_center
                dy = after_h_center - win_h_center
                vector_length = np.sqrt(dx**2 + dy**2)

                # count white px in template and skip if too few
                temp_h = template.shape[0]
                count = 0
                for z in range(temp_h):
                    count += int(np.sum(template[z, :])/255)
                if count < np.square(window_size)*0.05:
                    dx, dy = np.nan, np.nan

                # apply vector threshold
                vector_threshold = window_size * 0.7
                if vector_length >= vector_threshold:
                    dx, dy = np.nan, np.nan
                coordinate = np.array([win_w_center, win_h_center, dx, dy]).reshape(1, 4)

                frame_vector_infos = array_append(frame_vector_infos, coordinate)

                if np.isnan(dx):
                    continue


                after_h_center = win_h_center + dy * 10
                after_w_center = win_w_center + dx * 10

                cv2.arrowedLine(
                    copy_pre_frame,
                    (win_w_center, win_h_center),
                    (after_w_center, after_h_center),
                    (255, 0, 0),
                    thickness=3
                )
        # cv2.arrowedLine(
        #     copy_pre_frame,
        #     (win_w_center, win_h_center),
        #     (after_w_center, after_h_center),
        #     (0, 255, 0)
        # )
        # print((win_w_center, win_h_center),
        #     (after_w_center, after_h_center))

        # cv2.imshow("PIV Result", copy_pre_frame)
        # cv2.waitKey(0)

        r, c = frame_vector_infos.shape
        frame_vector_infos = frame_vector_infos.reshape(1, r, c)
        vector_infos = array_append(vector_infos, frame_vector_infos)
        out.write(copy_pre_frame)

        pre_frame = frame
    out.release()
    caputure.release()

def main():
    path = "C:\\Users\\tsaik\\PythonCode\\Analysis_202510_Data\\0.40_1.mp4"
    output = "C:\\Users\\tsaik\\PythonCode\\Analysis_202510_Data\\output2.mp4"
    piv(
        input_name=path,
        output_dir=output,
        window_size=100,
        overlap=50,
        vector_threshold=0
    )

import random
    
def test():
    save_array = np.array(
        [[a,2*a,np.nan] for a in range(5)]
    )
    print(save_array)
    print(type(save_array))
    np.save("test", save_array)

    load = np.load("test.npy")
    print(load)

if __name__ == "__main__":
    main()
    # test()