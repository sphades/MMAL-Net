import pandas as pd
import os
from sklearn.model_selection import train_test_split
import re

def generate_dog_files(root):
  dog = pd.read_csv(os.path.join(root,'labels.csv'))

  labels_file_list = []
  labels_file_dict = {}
  count = 0
  for i in dog['breed'].unique():
    labels_file_list.append(str(count) + "\t" + i)
    labels_file_dict[i]=count
    count = count + 1
  labels_file_str = "\n".join(labels_file_list)
  labels_file = open(os.path.join(root,"classes.txt"),"w")
  labels_file.write(labels_file_str)
  labels_file.close()

  trainval_list = []
  for image in os.listdir(os.path.join(root,"train")):
    img_base = os.path.basename(image)
    id = os.path.splitext(img_base)[0]
    if id in dog['id'].tolist():
      breed = dog[dog['id']==id]['breed'].to_string(index=False).strip()
      breed_num = labels_file_dict[breed]
      trainval_list.append(img_base+"\t"+str(breed_num))
  trainval_file = open(os.path.join(root,"trainval.txt"),"w")
  trainval_file.write("\n".join(trainval_list))
  trainval_file.close()

  train, test = train_test_split(trainval_list, test_size=0.1, random_state=42)

  train_file = open(os.path.join(root,"train.txt"),"w")
  train_file.write("\n".join(sorted(train, key=lambda x: int(re.search(r'\d+$',x).group()))))
  train_file.close()
  val_file = open(os.path.join(root,"validation.txt"),"w")
  val_file.write("\n".join(sorted(test, key=lambda x: int(re.search(r'\d+$',x).group()))))
  val_file.close()