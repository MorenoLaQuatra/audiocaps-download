# AudioCaps Download

**DISCLAIMER: This repository is a modified version of the [AudioSet Download](https://github.com/MorenoLaQuatra/audioset-download) repository.**

This repository contains code for downloading the [AudioCaps](https://github.com/cdjkim/audiocaps) dataset.
The repository is **not officially affiliated** with the AudioCaps dataset.

## Requirements

* Python 3.9 (it may work with other versions, but it has not been tested)

## Installation

```bash
# Install ffmpeg
sudo apt install ffmpeg
# Install audiocaps-download
pip install audiocaps-download
```

## Usage

The following code snippet downloads the complete dataset in WAV format, and stores it in the `test` directory.

```python
from audiocaps_download import Downloader
d = Downloader(root_path='audiocaps/', n_jobs=16)
d.download(format = 'wav') # it will cross-check the files with the csv files in the original repository
```

## Download updated Dec 2023

As of December 2023, the repository has been used to download the dataset in WAV format. The following table shows the number of files downloaded.

Difference between the original training set and the updated one: 4651
Difference between the original validation set and the updated one: 255
Difference between the original test set and the updated one: 460
Training set: 45187
Validation set: 2220
Test set: 4415

| Split | Number of files (using this repo) | Number of files (original) |
| --- | --- | --- |
| Training | 45187 | 49838 |
| Validation | 2220 | 2475 |
| Test | 4415 | 4875 |

The missing files may be due to the missing youtube videos, please open an issue [here](https://github.com/MorenoLaQuatra/audiocaps-download/issues/new) if you find any bug.

## Implementation

The main class is `audiocaps_download.Downloader`. It is initialized using the following parameters:
* `root_path`: the path to the directory where the dataset will be downloaded.
* `n_jobs`: the number of parallel downloads. Default is 1.

The methods of the class are:
* `download(format='vorbis', quality=5)`: downloads the dataset. 
* The format can be one of the following (supported by [yt-dlp](https://github.com/yt-dlp/yt-dlp#post-processing-options) `--audio-format` parameter):
    * `vorbis`: downloads the dataset in Ogg Vorbis format. This is the default.
    * `wav`: downloads the dataset in WAV format.
    * `mp3`: downloads the dataset in MP3 format.
    * `m4a`: downloads the dataset in M4A format.
    * `flac`: downloads the dataset in FLAC format.
    * `opus`: downloads the dataset in Opus format.
    * `webm`: downloads the dataset in WebM format.
    * ... and many more.
  * The quality can be an integer between 0 and 10. Default is 5.
* `load_dataset()`: reads the csv files from the original repository. It is not used externally.
* `download_file(...)`: downloads a single file. It is not used externally.