word = {}
word['<pad>'] = 0
word['<unk>'] = 1
word_cnt = 2

input_path = "/data/wangshuhe/test_mmi/train.src.txt"
output_path = "/data/wangshuhe/test_mmi/mmi.dict"

print("read ...")
with open(input_path, "r") as f:
    for line in f:
        line = line.strip().split()
        for sub_word in line:
            if (sub_word not in word):
                word[sub_word] = word_cnt
                word_cnt += 1
    f.close()

print("write ...")
with open(output_path, "w") as f:
    for key, value in word.items():
        f.write(key+'\n')
    f.close()