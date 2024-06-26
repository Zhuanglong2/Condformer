import os
import fnmatch
import shutil
from PIL import Image

original_dir = r'E:\360MoveData\Users\409\Desktop\data'
read_dir = r'E:\360MoveData\Users\409\Desktop\test_t'
save_dir = r'E:\360MoveData\Users\409\Desktop\data2'


dir_or = os.listdir(original_dir)
dir_read = os.listdir(read_dir)
if not os.path.exists(save_dir):
        os.makedirs(save_dir)

i = 0
for filename in dir_or:
    if fnmatch.fnmatch(filename, '*.gt.jpg'):
        continue
    else:
        src = original_dir + '\\' + filename
        new_name = save_dir + '\\' + dir_read[i]
        im = Image.open(src)
        out = im.resize((640, 480), Image.ANTIALIAS)
        out.save(new_name)
        i += 1



# original_dir = r'E:\lane-detecion\CULane\driver_100_30frame'
# save_dir = r'E:\lane-detecion\data\driver_100_30frame'
# read_dir = r'E:\360MoveData\Users\409\Desktop\test_t'
#
# # dir_or = os.listdir(original_dir)
# dir_or = ['05251517_0433.MP4', '05251520_0434.MP4']
# i = 0
# for filename in dir_or:
#     now_dir = original_dir + '\\' + filename
#     dir_or_ = os.listdir(now_dir)
#     data = os.listdir(read_dir)
#
#     create_dir =save_dir + '\\' + filename
#     # read_dir_ = read_dir + '\\' + filename
#     if not os.path.exists(create_dir):
#         os.makedirs(create_dir)
#     for filename2 in dir_or_:
#         if fnmatch.fnmatch(filename2, '*.jpg'):
#             src = os.path.join(read_dir, data[i])
#             dst = os.path.join(create_dir)
#             new_name = create_dir + '\\' + filename2
#             im = Image.open(src)
#             out = im.resize((1640, 590), Image.ANTIALIAS)
#             out.save(new_name)
#             # # 复制图像
#             # shutil.copy(src, dst)
#             # # 重命名
#             # os.rename(src, new_name)
#             i += 1
