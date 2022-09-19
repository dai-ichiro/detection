import zipfile
from gdown import download

gdown_fname = download('https://drive.google.com/uc?id=1Ko9yIfiRdD3TLPwG8nlcjNMaADABgp5s', quiet = False)

with zipfile.ZipFile(gdown_fname) as f:
    f.extractall(path='.')

