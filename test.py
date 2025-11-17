import cv2
import numpy as np
from matplotlib import pyplot as plt

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



    # 出力動画の準備
    forcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_dir,
        forcc,
        fps=10,
        frameSize=(pre_frame.shape[1], pre_frame.shape[0])
    )

    # いろいろな初期化
    vector_infos = []
    pre_frame = None
    i = 0

    # PIV解析ループ
    while True:
        # 読み込み　読み込みの失敗または規定回数を超えたら終了
        i += 1
        print("start ", i)
        ret, frame = caputure.read()
        if not ret or i > 50:
            print("failed to read frame", i)
            caputure.release()
            break

        # 前処理　画像を二値化
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, frame = cv2.threshold(
            frame, 70, 255, cv2.THRESH_BINARY)
        
        # 初回フレームの処理
        if pre_frame is None:
            pre_frame = frame
            continue

        # のちの描画用の画像を用意
        copy_pre_frame = pre_frame.copy()
        copy_pre_frame = cv2.cvtColor(copy_pre_frame, cv2.COLOR_GRAY2BGR)

        #   ^^^^^^^^^^^^^^
        # < Calculate PIV >
        #   vvvvvvvvvvvvvv

        h, w = frame.shape
        h_steps = int(h / (window_size - overlap))
        w_steps = int(w / (window_size - overlap))

        frame_vector_infos = []

        for h_step in range(h_steps - 1):
            for w_step in range(w_steps - 1):
                # define window to set template
                win_h_start = h_step * (window_size - overlap)
                win_h_end = win_h_start + window_size
                win_w_start = w_step * (window_size - overlap)
                win_w_end = win_w_start + window_size

                # template matching
                template = pre_frame[win_h_start:win_h_end,win_w_start:win_w_end]
                method = cv2.TM_CCOEFF_NORMED
                res = cv2.matchTemplate(frame, template, method)
                _,_,_, max_loc = cv2.minMaxLoc(res)

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
                if count < np.square(window_size)*0.05:     # threshold: 5% white px
                    dx, dy = np.nan, np.nan

                # apply vector threshold
                vector_threshold = window_size * 0.7    # threshold: √2/2 of window size
                if vector_length >= vector_threshold:
                    dx, dy = np.nan, np.nan
                coordinate = [win_w_center, win_h_center, dx, dy]

                frame_vector_infos.append(coordinate)

                if np.isnan(dx):
                    continue


                after_h_center = win_h_center + dy * 10
                after_w_center = win_w_center + dx * 10


                cv2.arrowedLine(
                    copy_pre_frame,
                    (win_w_center, win_h_center),
                    (int(after_w_center), int(after_h_center)),
                    (255, 0, 0),
                    thickness=3
                )

        vector_infos.append(frame_vector_infos)
        out.write(copy_pre_frame)

        pre_frame = frame

    vector_infos = np.array(vector_infos)
    np.save("C:\\Users\\tsaik\\PythonCode\\Analysis_202510_Data\\vector_infos.npy", vector_infos)
    out.release()
    caputure.release()

def get_average_velocity():
    vector_infos = np.load("C:\\Users\\tsaik\\PythonCode\\Analysis_202510_Data\\vector_infos.npy", allow_pickle=True)
    info_path = ""
    with open(info_path, "r") as f:
        info = {}
        for line in f:
            key, value = line.strip().split(":")
            info[key] = float(value)

    setted_line_position = list(map(int, info["Line positions(px,10s)"].split(",")))
    skep_frames = 100

    continuous_line_positions = []
    for pre, after in zip(setted_line_position, setted_line_position[1:]):
        lines = list(map(int, np.linspace(pre, after, skep_frames+1)))
        continuous_line_positions.extend(lines[:-1])
    
    for frame_vector, line_position in zip(vector_infos, continuous_line_positions):
        frame_vectors = np.array(frame_vector)
        average_velocity_x = 0




    


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

    
def test():

    import random
    A:list[list[list[float]]] = [
        [
            list(range(a*b, a*b+4)) for b in range(1,3)
        ] for a in range(0,10,2)
    ]
    A.append([[10,20,np.nan,40],list(np.linspace(50,80,4))])
    print(A)
    print(np.array(A))
    
    B = np.linspace(0,1,10)
    print(list(filter(
        lambda x: x%2 == 0 in B, 
    )
    ))


    # save_array = np.array(
    #     [[a,2*a,np.nan] for a in range(5)]
    # )
    # print(save_array)
    # print(type(save_array))
    # np.save("test", save_array)

    # load = np.load("test.npy")
    # print(load)

if __name__ == "__main__":
    # main()
    test()