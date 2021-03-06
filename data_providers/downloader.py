import sys ,os
from urllib import urlretrieve
import tarfile

import zipfile



def report_download_progress(count , block_size , total_size):
    pct_complete = float(count * block_size) / total_size
    msg = "\r {0:1%} already downloader".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()

def download_data_url(url, download_dir):
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir , filename)
    if not os.path.exists(file_path):
        try:
            os.makedirs(download_dir)
        except Exception :
            pass

        print "Download %s  to %s" %(url , file_path)
        file_path , _ = urlretrieve(url=url,filename=file_path,reporthook=report_download_progress)
        print file_path
        print('\nExtracting files')
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path , mode="r").extracall(download_dir)
        elif file_path.endswith(".tar.gz" , ".tgz"):
            tarfile.open(name=file_path , mode='r:gz').extractall(download_dir)


