import os
import glob

def smooth(datas, win_size=10000):
    ret = []
    tmp_sum = 0
    for i in range(len(datas)):
        tmp_sum += datas[i]
        if i >= win_size - 1:
            ret.append(tmp_sum / win_size)
            tmp_sum -= datas[i - win_size + 1]
    return ret
        

def cleanup_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        files = glob.glob(os.path.join(path, '*'))
        for f in files:
            try:
                os.remove(f)
            except:
                pass      