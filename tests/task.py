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
        data_dir = "/home/mengyuxian/video-dialogue-model/sample_data/preprocessed_data"
        max_src_sent = 5

    task = VideoDialogueTask.setup_task(args=Args)
    task.load_dataset("train")
    dataset = task.datasets["train"]
    print(dataset[0])


if __name__ == '__main__':
    video_task()
