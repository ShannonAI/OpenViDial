import os

dict_dir = "/data/wangshuhe/test_mmi/mmi.dict"
data_dir = "/data/wangshuhe/test_mmi"

print("read dict ...")
word_dict = {}
word_cnt = 0
with open(dict_dir, "r") as f:
    for line in f:
        line = line.strip()
        word_dict[line] = word_cnt
        word_cnt += 1
    f.close()

print("preprocess data ...")
for sub_name in ['train', 'valid', 'test']:
    print(f"{sub_name} ...")
    with open(os.path.join(data_dir, sub_name+'.src.txt'), "r") as read_file, open(os.path.join(data_dir, sub_name+'.mmi'), "w") as write_file:
        for line in read_file:
            line = line.strip().split()
            new_line = ""
            for word in line:
                if (word not in word_dict):
                    new_line += str(word_dict['<unk>']) + " "
                else:
                    new_line += str(word_dict[word]) + " "
            write_file.write(new_line+'\n')
        read_file.close()
        write_file.close()