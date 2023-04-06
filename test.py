from audiocaps_download import Downloader

d = Downloader(root_path='AudioCaps_dataset/', n_jobs=64)
#d.download(format = 'wav', quality=0)
d.format = 'wav'
d.quality = 0
d.cross_check()