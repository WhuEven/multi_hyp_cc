#coding:utf-8
# calculate nus dataset result among every camera folders
import os

cam_names = ["canon_eos_1D_mark3" "canon_eos_600D" "fuji" "nikonD5200" "panasonic" "olympus" "sony" "samsung"]
ex_id = "ex3"
exam_name = ""
F = open("./result.txt","w+")
mean = med = tri = e25 = e75 = e95 = 0

for i in cam_names:
    path = os.path.join(ex_id,i,"checkpoint/Nus",i,exam_name,"stdout.txt")
    with open(path, 'r', encoding='utf-8') as f: 
        F.write(i+"\n")
        F.write(f[-1]+"\n")
        line = f[-1].split()
        mean += float(line[0][-6:])
        med += float(line[1][-6:])
        tri += float(line[2][-6:])
        e25 += float(line[3][-6:])
        e75 += float(line[4][-6:])
        e95 += float(line[5][-6:])

mean /= 8.0
med /= 8.0
tri /= 8.0
e25 /= 8.0
e75 /= 8.0
e95 /= 8.0
F.writelines("total mean: ",str(mean)," ","med: ", str(med), " ","tri: ", str(tri), " ","25: ", str(e25), " ","75: ", str(e75), " ","95: ", str(e95), " ",)
F.close()
