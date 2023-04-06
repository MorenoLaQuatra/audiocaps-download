import os
import time
import joblib
import pandas as pd
import torchaudio
from tqdm import tqdm

class Downloader:
    """
    This class implements the download of the AudioSet dataset.
    It only downloads the audio files according to the provided list of labels and associated timestamps.
    """

    def __init__(self, 
                    root_path: str,
                    n_jobs: int = 1,
                    ):
        """
        This method initializes the class.
        :param root_path: root path of the dataset
        :param n_jobs: number of parallel jobs
        :param download_type: type of download (unbalanced_train, balanced_train, eval)
        :param copy_and_replicate: if True, the audio file is copied and replicated for each label. 
                                    If False, the audio file is stored only once in the folder corresponding to the first label.
        """
        # Set the parameters
        self.root_path = root_path
        self.n_jobs = n_jobs

        self.format_dict = {
            'vorbis': 'ogg',
            'mp3': 'mp3',
            'm4a': 'm4a',
            'wav': 'wav',
        }

        # Create the path
        os.makedirs(self.root_path, exist_ok=True)
        os.makedirs(self.root_path + '/train/', exist_ok=True)
        os.makedirs(self.root_path + '/val/', exist_ok=True)
        os.makedirs(self.root_path + '/test/', exist_ok=True)

        self.load_dataset()

    def load_dataset(self):
        """
        This method reads the metadata of the dataset.
        """

        self.train_df = pd.read_csv(
            f"https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/train.csv", 
            sep=',',
        )

        self.val_df = pd.read_csv(
            f"https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/val.csv",
            sep=',',
        )

        self.test_df = pd.read_csv(
            f"https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/test.csv",
            sep=',',
        )

        return

    def download(
        self,
        format: str = 'vorbis',
        quality: int = 5,    
    ):
        """
        This method downloads the dataset using the provided parameters.
        :param format: format of the audio file (vorbis, mp3, m4a, wav), default is vorbis
        :param quality: quality of the audio file (0: best, 10: worst), default is 5
        """

        t1 = time.time()

        self.format = format
        self.quality = quality

        # make sure to cast audiocap_id to string
        self.train_df['audiocap_id'] = self.train_df['audiocap_id'].astype(str)
        self.val_df['audiocap_id'] = self.val_df['audiocap_id'].astype(str)
        self.test_df['audiocap_id'] = self.test_df['audiocap_id'].astype(str)

        # Training set 
        print("Downloading the training set...")
        # parallel + tqdm
        joblib.Parallel(n_jobs=self.n_jobs, verbose=10)(
            joblib.delayed(self.download_file)(
                root_path=self.root_path + '/train/',
                ytid=row['youtube_id'],
                start_seconds=row['start_time'],
                end_seconds=row['start_time'] + 10.0,
                audiocaps_id=row['audiocap_id'],
            ) for _, row in tqdm(self.train_df.iterrows())
        )

        # Validation set
        print("Downloading the validation set...")
        # parallel + tqdm
        joblib.Parallel(n_jobs=self.n_jobs, verbose=10)(
            joblib.delayed(self.download_file)(
                root_path=self.root_path + '/val/',
                ytid=row['youtube_id'],
                start_seconds=row['start_time'],
                end_seconds=row['start_time'] + 10.0,
                audiocaps_id=row['audiocap_id'],
            ) for _, row in tqdm(self.val_df.iterrows())
        )

        # Test set
        print("Downloading the test set...")
        # parallel + tqdm
        joblib.Parallel(n_jobs=self.n_jobs, verbose=10)(
            joblib.delayed(self.download_file)(
                root_path=self.root_path + '/test/',
                ytid=row['youtube_id'],
                start_seconds=row['start_time'],
                end_seconds=row['start_time'] + 10.0,
                audiocaps_id=row['audiocap_id'],
            ) for _, row in tqdm(self.test_df.iterrows())
        )

        t2 = time.time()

        # print stats on the number of files per split (all self.format files)
        print(f"[Before cross-ckeck] Number of files in the training set: {len(os.listdir(self.root_path + '/train/'))}")
        print(f"[Before cross-ckeck] Number of files in the validation set: {len(os.listdir(self.root_path + '/val/'))}")
        print(f"[Before cross-ckeck] Number of files in the test set: {len(os.listdir(self.root_path + '/test/'))}")
        print(f"[Before cross-ckeck] Time to download the dataset: {t2 - t1:.2f} seconds")

        # cross-check the files
        self.cross_check()

        print("AudioCaps dataset downloaded.")

    def is_valid_file(self, path):
        '''
        It checks if the file is valid.
        Returns True if the file is valid, False otherwise.
        '''

        # check if the file exists
        if not os.path.isfile(path):
            return False

        try:
            # load the file
            waveform, sample_rate = torchaudio.load(path)
            # check if length is 0
            if waveform.shape[1] == 0:
                print('Error loading audio file: ', path)
                return False
        except Exception as e:
            print('Error loading audio file: ', path)
            print(e)
            return False

        return True

    def cross_check_file(self, root_path, row):
        '''
        It checks if the file is valid.
        Returns True if the file is valid, False otherwise.
        '''

        audiocaps_id = str(row['audiocap_id'])

        if self.is_valid_file(root_path + audiocaps_id + '.' + self.format):
            return int(audiocaps_id)
        else:
            # delete file if it exists
            if os.path.isfile(root_path + audiocaps_id + '.' + self.format):
                # delete file
                os.remove(root_path + audiocaps_id + '.' + self.format)

        return None


    def cross_check(self):
        '''
        This function aims at cross-checking the downloaded dataset.
        It should remove files that are empty or corrupted.
        '''
        '''
        for _, row in tqdm(self.train_df.iterrows()):
            path = self.root_path + '/train/' + row['audiocap_id'] + '.' + self.format
            if self.is_valid_file(path):
                updated_train_df = updated_train_df.append(row, ignore_index=True)
        '''

        # Training set
        print("Cross-checking the training set...")
        list_of_audiocaps_id = []
        self.updated_train_df = pd.DataFrame(columns=self.train_df.columns)
        list_of_audiocaps_id = joblib.Parallel(n_jobs=self.n_jobs, verbose=10)(
            joblib.delayed(self.cross_check_file)(
                root_path=self.root_path + '/train/',
                row=row,
            ) for _, row in tqdm(self.train_df.iterrows())
        )
        # cast self.train_df['audiocap_id'] to str
        self.train_df['audiocap_id'] = self.train_df['audiocap_id'].astype(str)
        list_of_audiocaps_id = [str(x) for x in list_of_audiocaps_id if x is not None]
        self.updated_train_df = self.train_df[self.train_df['audiocap_id'].isin(list_of_audiocaps_id)]

        print(list_of_audiocaps_id)
        print("len: ", len(list_of_audiocaps_id))

        # Validation set
        print("Cross-checking the validation set...")
        self.updated_val_df = pd.DataFrame(columns=self.val_df.columns)
        list_of_audiocaps_id = []
        list_of_audiocaps_id = joblib.Parallel(n_jobs=self.n_jobs, verbose=10)(
            joblib.delayed(self.cross_check_file)(
                root_path=self.root_path + '/val/',
                row=row,
            ) for _, row in tqdm(self.val_df.iterrows())
        )
        # cast self.val_df['audiocap_id'] to str
        self.val_df['audiocap_id'] = self.val_df['audiocap_id'].astype(str)
        list_of_audiocaps_id = [str(x) for x in list_of_audiocaps_id if x is not None]
        self.updated_val_df = self.val_df[self.val_df['audiocap_id'].isin(list_of_audiocaps_id)]
        
        print(list_of_audiocaps_id)
        print("len: ", len(list_of_audiocaps_id))

        # Test set
        print("Cross-checking the test set...")
        self.updated_test_df = pd.DataFrame(columns=self.test_df.columns)
        list_of_audiocaps_id = []
        list_of_audiocaps_id = joblib.Parallel(n_jobs=self.n_jobs, verbose=10)(
            joblib.delayed(self.cross_check_file)(
                root_path=self.root_path + '/test/',
                row=row,
            ) for _, row in tqdm(self.test_df.iterrows())
        )

        print(list_of_audiocaps_id)
        print("len: ", len(list_of_audiocaps_id))

        # cast self.test_df['audiocap_id'] to str
        self.test_df['audiocap_id'] = self.test_df['audiocap_id'].astype(str)
        list_of_audiocaps_id = [str(x) for x in list_of_audiocaps_id if x is not None]
        self.updated_test_df = self.test_df[self.test_df['audiocap_id'].isin(list_of_audiocaps_id)]

        # update the dataframes
        print(f"Difference between the original training set and the updated one: {len(self.train_df) - len(self.updated_train_df)}")
        print(f"Difference between the original validation set and the updated one: {len(self.val_df) - len(self.updated_val_df)}")
        print(f"Difference between the original test set and the updated one: {len(self.test_df) - len(self.updated_test_df)}")

        print(f"Training set: {len(self.updated_train_df)}")
        print(f"Validation set: {len(self.updated_val_df)}")
        print(f"Test set: {len(self.updated_test_df)}")

        # update the dataframes
        self.train_df = self.updated_train_df
        self.val_df = self.updated_val_df
        self.test_df = self.updated_test_df

        
        # store the CSV files
        self.updated_train_df.to_csv(self.root_path + '/train.csv', index=False)
        self.updated_val_df.to_csv(self.root_path + '/val.csv', index=False)
        self.updated_test_df.to_csv(self.root_path + '/test.csv', index=False)


    def download_file(
            self, 
            root_path: str,
            ytid: str, 
            start_seconds: float,
            end_seconds: float,
            audiocaps_id: str = None,
        ):
        """
        This method downloads a single file. It only download the audio file at 16kHz.
        If a file is associated to multiple labels, it will be stored multiple times.
        :param ytid: YouTube ID.
        :param start_seconds: start time of the audio clip.
        :param end_seconds: end time of the audio clip.
        """
        

        target_file_path = os.path.join(root_path, audiocaps_id + '.' + self.format_dict[self.format])

        # skip if the file already exists
        if os.path.isfile(target_file_path):
            return

        # Download the file using yt-dlp
        os.system(f'yt-dlp -x --audio-format {self.format} --audio-quality {self.quality} --output "{target_file_path}" --postprocessor-args "-ss {start_seconds} -to {end_seconds}" https://www.youtube.com/watch?v={ytid}')
        
        return