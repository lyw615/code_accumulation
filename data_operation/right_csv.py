import os
"用于纠正多折训练数据划分输出的csv格式"

csv_dir=r"/home/data1/yw/data/compt_data/qzb_ship/k-fold-fine/fold_v5"
csv_files=os.listdir(csv_dir)
for csv_f in csv_files:
    csv_path=os.path.join(csv_dir,csv_f)
    with open(csv_path, "r") as f:
        lines_record = f.readlines()
        lines_record.pop(0)

    lines = []

    for line in lines_record:
        lines.append(line.strip("\n").split(',')[1] )

    new_csv_path=os.path.join(csv_dir,"new_%s"%csv_f)
    with open(new_csv_path, "w") as f:
        for i in lines:
            f.write("%s\n"%i)