# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: task
@time: 2020/11/18 19:32
@desc: 

"""


from video_dialogue_model import VideoDialogueTask


def video_task():
    class Args:
        data_dir = "/data/yuxian/datasets/video/preprocessed_data"
        # data_dir = "../sample_data/preprocessed_data"
        max_src_sent = 5
        img_type = "features"
    split = "valid"
    task = VideoDialogueTask.setup_task(args=Args)
    task.load_dataset(split)
    dataset = task.datasets[split]
    print(dataset[len(dataset)-1])


if __name__ == '__main__':
    video_task()
