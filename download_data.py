import wget
import os.path
import zipfile

def get_file(fname, origin, untar=False, cache_dir='data'):
    data_dir = os.path.join(cache_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    fpath = os.path.join(data_dir, fname)

    print(fpath)
    if not os.path.exists(fpath):
        print('Downloading data from', origin)

        try:
            wget.download(fpath, origin)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            print("Download Aborted!")
            print(Exception)
    return fpath


def unzip_dataset(fpath, target_path):
    with zipfile.ZipFile(fpath) as myzip:
        myzip.extractall(target_path)

if __name__ == "__main__":
    path = get_file("sheep", 'http://dl.yf.io/lsun/objects/sheep.zip')