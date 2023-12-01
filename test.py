from audiocaps_download import Downloader

d = Downloader(root_path='audiocaps/', n_jobs=96)
d.download(format = 'wav', quality=0)