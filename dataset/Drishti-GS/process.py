
mask_path= './all_cropped.txt'
classification_path= './classification_label.txt'
mask_with_classification_path = './seg_classification_label.txt'
with open(mask_path, 'r') as f:
    ids = f.read().splitlines()
with open(classification_path, 'r') as f:
    label_ids = f.read().splitlines()
with open(mask_with_classification_path,'a') as f:
    for i in ids:
        now_id = i.split()[0].split('/')[-1].replace('.png','')
        for j in label_ids:
            if now_id == j.split()[0]:
                print(now_id)
                print(j.split()[1])
                f.write(i + ' ' + j.split()[1] + '\n')
