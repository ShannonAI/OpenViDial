import os

# the output file path of the forward models
path = ["/userhome/shuhe/movie_plus/pre_feature/OpenViDial/text_ori_result_data_2.0_fp32/",
        "/userhome/shuhe/movie_plus/pre_feature/OpenViDial/feature_result_data_2.0_fp32/",
        "/userhome/shuhe/movie_plus/pre_feature/latest/OpenViDial/object_data_2.0/"]

# the output file path of the MMI models
mmi_path = ["/userhome/shuhe/movie_plus/pre_feature/OpenViDial/text_ori_result_data_2.0_fp32/test_best5/bidirection0.037.out",
        "/userhome/shuhe/movie_plus/pre_feature/OpenViDial/feature_result_data_2.0_fp32/test_best5/bidirection0.0575.out",
        "/userhome/shuhe/movie_plus/pre_feature/latest/OpenViDial/object_data_2.0/test_15_best5/bidirection0.000957.out"]

# path to store tmp files of mmi
model_path = ["/userhome/shuhe/movie_plus/pre_feature/OpenViDial/text_ori_result_data_2.0_fp32/test_best5/tmp_model",
        "/userhome/shuhe/movie_plus/pre_feature/OpenViDial/feature_result_data_2.0_fp32/test_best5/tmp_model",
        "/userhome/shuhe/movie_plus/pre_feature/latest/OpenViDial/object_data_2.0/test_15_best5/tmp_model"]

for sub_path in path:
    print(sub_path)
    shuhe_list = []
    # You can get it by `grep ^D gen.out | cut -f3- > sys.txt`, where `gen.out` is output file of fairseq-generate
    with open(sub_path+'gen.out.sys', "r") as f:
        for line in f:
            shuhe_list.append(line)
        f.close()
    for i in range(len(shuhe_list)):
        with open(sub_path+'tmp_model/'+str(i)+'_model.txt', "w") as f:
            f.write(shuhe_list[i])
            f.close()

print("====")
shuhe_list = []
with open(path[0]+'test_gen.ref', "r") as f:
    for line in f:
        shuhe_list.append(line)
    f.close()
for i in range(len(shuhe_list)):
    with open(path[0]+'tmp_ref/'+str(i)+'_ref.txt', "w") as f:
        f.write(shuhe_list[i])
        f.close()

for idx_, sub_path in enumerate(mmi_path):
    print(sub_path)
    shuhe_list = []
    with open(sub_path, "r") as f:
        for line in f:
            shuhe_list.append(line)
        f.close()
    for i in range(len(shuhe_list)):
        with open(model_path[idx_]+"/"+str(i)+'_model.txt', "w") as f:
            f.write(shuhe_list[i])
            f.close()

from pyrouge import Rouge155

path.append("/userhome/shuhe/movie_plus/pre_feature/OpenViDial/text_ori_result_data_2.0_fp32/test_best5/")
path.append("/userhome/shuhe/movie_plus/pre_feature/OpenViDial/feature_result_data_2.0_fp32/test_best5/test_best5/")
path.append("/userhome/shuhe/movie_plus/pre_feature/latest/OpenViDial/object_data_2.0/test_15_best5/test_best5/")

for sub_path in path:
    r = Rouge155()
    # set directories
    r.system_dir = sub_path+'tmp_model/'
    r.model_dir = "/userhome/shuhe/movie_plus/pre_feature/OpenViDial/text_ori_result_data_2.0_fp32/tmp_ref/"
 
    # define the patterns
    r.system_filename_pattern = '(\d+)_model.txt'
    r.model_filename_pattern = '#ID#_ref.txt'
 
    # use default parameters to run the evaluation
    output = r.convert_and_evaluate()
    print(output)