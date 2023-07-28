import os
from zipfile import ZipFile

path = ''
zip_file = ZipFile(path, 'r')
# for folder_name in folders_of_interest:
#     zip_file.extract(folder_name, output_folder)

# 关闭zip文件
zip_file.namelist()
zip_file.close()