from main import embeddings_y_taggrams_MusiCNN, embeddings_y_taggrams_VGG, MSD_W_MUSICNN, MTAT_W_MUSICNN,MSD_W_VGG
AUDIO_ROUTE = './audio/'

def embeddings_y_taggrams(red,pista,dataset):
    if red == '':
        red = 'MusiCNN'
    if pista == "":
        pista = '1.mp3'
    if dataset == "":
        dataset = 'MSD'

    funcion = embeddings_y_taggrams_MusiCNN
    ds = MSD_W_MUSICNN

    if red == 'VGG':
        funcion = embeddings_y_taggrams_VGG
        if dataset == 'MSD':
            ds = MSD_W_VGG
    else:
        if dataset == 'MTAT':
            ds=  MTAT_W_MUSICNN

    print('Obteniendo embeddings y taggrams para ',red,pista,dataset,"...")
    embeddings, taggrams = funcion(ds,AUDIO_ROUTE+pista)
    return embeddings, taggrams