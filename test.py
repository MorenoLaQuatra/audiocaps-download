from audiocaps_download import Downloader

d = Downloader(root_path='../AudioCaps/', n_jobs=64)
d.download(format = 'wav', quality=0)
