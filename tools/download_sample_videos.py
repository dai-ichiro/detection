from torchvision.datasets.utils import download_url

video1_url = 'https://github.com/dai-ichiro/robo-one/raw/main/video_1.mp4'
download_url(video1_url, root = '.', filename = 'target.mp4')

video2_url = 'https://github.com/dai-ichiro/robo-one/raw/main/video_2.mp4'
video2_fname = 'non_target.mp4'
download_url(video2_url, root = '.', filename = 'non_target.mp4')