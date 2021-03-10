#coding:utf-8
# calculate nus dataset result among every camera folders
import os

cam_names = ["canon_eos_1D_mark3", "canon_eos_600D", "fuji", "nikonD5200", "panasonic", "olympus", "sony", "samsung"]
ex_id = "ex8"
head = "new_ex_output"
exam_name = "table3_finetune"
result_path = os.path.join(head,ex_id,"all_result.txt")
F = open(result_path,"w+")
mean = med = tri = e25 = e75 = e95 = 0

for i in cam_names:
    # path = os.path.join(head,ex_id,i,"checkpoint/Nus",i,exam_name,"stdout.txt")
    path = os.path.join(head,ex_id,i, "stdout.txt")
    with open(path, 'r', encoding='utf-8') as f: 
        lines = f.readlines()
        F.write(i+"\n")
        F.write(lines[-2]+"\n")
        line = lines[-2].split()
        mean += float(line[2])
        med += float(line[4])
        tri += float(line[6])
        e25 += float(line[8])
        e75 += float(line[10])
        e95 += float(line[12])

mean /= 8.0
med /= 8.0
tri /= 8.0
e25 /= 8.0
e75 /= 8.0
e95 /= 8.0
F.writelines(["total mean: ",str(mean)," ","med: ", str(med), " ","tri: ", str(tri), " ","25: ", str(e25), " ","75: ", str(e75), " ","95: ", str(e95), " "])
F.close()
