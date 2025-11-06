import torch
import numpy as np
import librosa
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchaudio
from MusiCNN import Musicnn
from VGG import VGG_Res
from proyecciones import proyectar_embeddings

MSD_W_MUSICNN = './pesos/msd/musicnn.pth'
MTAT_W_MUSICNN = './pesos/mtat/musicnn.pth'

MSD_W_VGG = './pesos/msd/vgg.pth'
#MTAT_W_vgg = './pesos/mtat/vgg.pth'

N_TAGS = 50      # Número de etiquetas (tags)
EMBEDDING_DIM = 200  # Dimensión de los Embeddings (la clase Musicnn lo define internamente) 
SR_MUSICNN = 16000     # Tasa de muestreo que MusiCNN espera

TAGS = ['genre---downtempo', 'genre---ambient', 'genre---rock', 'instrument---synthesizer', 'genre---atmospheric', 'genre---indie', 'instrument---electricpiano', 'genre---newage', 'instrument---strings', 'instrument---drums', 'instrument---drummachine', 'genre---techno', 'instrument---guitar', 'genre---alternative', 'genre---easylistening', 'genre---instrumentalpop', 'genre---chillout', 'genre---metal', 'mood/theme---happy', 'genre---lounge', 'genre---reggae', 'genre---popfolk', 'genre---orchestral', 'instrument---acousticguitar', 'genre---poprock', 'instrument---piano', 'genre---trance', 'genre---dance', 'instrument---electricguitar', 'genre---soundtrack', 'genre---house', 'genre---hiphop', 'genre---classical', 'mood/theme---energetic', 'genre---electronic', 'genre---world', 'genre---experimental', 'instrument---violin', 'genre---folk', 'mood/theme---emotional', 'instrument---voice', 'instrument---keyboard', 'genre---pop', 'instrument---bass', 'instrument---computer', 'mood/theme---film', 'genre---triphop', 'genre---jazz', 'genre---funk', 'mood/theme---relaxing']

DC = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def load_and_preprocess(audio_path, sr=SR_MUSICNN):
    try:
        y, _ = librosa.load(audio_path, sr=sr)
    except FileNotFoundError:
        print("\n[ERROR] Archivo no encontrado. Asegúrate de que la ruta sea correcto.")
        return None
    except Exception as e:
        print(f"\n[ERROR] Error al cargar el archivo: {e}")
        return None
        
    return torch.from_numpy(y).float()

def embeddings_y_taggrams_MusiCNN(pesos, audio, dataset_name='msd' ,segment_size=3.0, sr=SR_MUSICNN):

    print(pesos, audio, dataset_name)
    model = Musicnn(n_class=N_TAGS, dataset=dataset_name) 
    model.load_state_dict(torch.load(pesos, map_location=DC))
    model.eval()

    y, _ = librosa.load(audio, sr=sr)

    seg_len = int(segment_size * sr)
    segments = [y[i:i+seg_len] for i in range(0, len(y), seg_len) if len(y[i:i+seg_len]) == seg_len]

    all_embeddings = []
    all_taggrams = []

    with torch.no_grad():
        for seg in segments:
            x = torch.from_numpy(seg).float().unsqueeze(0)
            taggrams_tensor, embeddings_tensor = model(x)
            all_embeddings.append(embeddings_tensor.squeeze(0).numpy())
            all_taggrams.append(taggrams_tensor.squeeze(0).numpy())


    embeddings = np.vstack(all_embeddings)  # (n_segmentos, dim)
    taggrams = np.vstack(all_taggrams)      # (n_segmentos, n_tags)
    
    return embeddings, taggrams

def embeddings_y_taggrams_VGG(pesos, audio, segment_size=3.0, sr=SR_MUSICNN):
    model = VGG_Res(n_class=N_TAGS)
    model.load_state_dict(torch.load(pesos, map_location=DC))
    model.to(DC)
    model.eval()

    y, _ = librosa.load(audio, sr=sr)

    seg_len = int(segment_size * sr)
    segments = [y[i:i+seg_len] for i in range(0, len(y), seg_len) if len(y[i:i+seg_len]) == seg_len]

    all_embeddings = []
    all_taggrams = []

    with torch.no_grad():
        for seg in segments:
            x = torch.from_numpy(seg).float().unsqueeze(0).to(DC)  # (1, n_samples)
            taggrams_tensor, embeddings_tensor = model(x)

            all_embeddings.append(embeddings_tensor.squeeze(0).cpu().numpy())
            all_taggrams.append(taggrams_tensor.squeeze(0).cpu().numpy())

    embeddings = np.vstack(all_embeddings)
    taggrams = np.vstack(all_taggrams)
    
    return embeddings, taggrams


if __name__ == "__main__":
    embeddings, taggrams = embeddings_y_taggrams_VGG(MSD_W_VGG, './audio/1.mp3')
    print("Embeddings:", embeddings.shape)
    print("Taggrams:", taggrams.shape)

    coords = proyectar_embeddings(embeddings, metodo='umap')
    print(coords[:5])
