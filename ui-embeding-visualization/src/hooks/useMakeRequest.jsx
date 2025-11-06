
export const useMakeRequest = () => {
    const URL = "http://localhost:5000";

    const getEmbeddingsTaggrams = async (red, dataset, pista) => {
        let reqRed = '';
        if (red) {
            reqRed = 'red=' + red
        }
        let reqDataset = '';
        if (dataset) {
            reqDataset = 'dataset=' + dataset;
        }
        let reqPista = ''
        if (pista) {
            reqPista = 'pista=' + pista;
        }
        let params = ''
        if (reqRed !== '' && reqDataset !== '' && reqPista !== '') {
            params = '?' + reqRed + '&' + reqDataset + '&' + reqPista
        }

        const data = await fetch(URL + '/embedding' + params);
        const eyt = await data.json()
        console.log(eyt)
        return eyt;
    }

    const obtenerAudios = async () => {
        const data = await fetch(URL + '/audios');
        const audios = await data.json()
        return audios
    }
    const obtenerTags = async () => {
        const data = await fetch(URL + '/tags');
        const tags = await data.json()
        return tags
    }

    const obtenerRepresentacion = async (red, dataset, pista, metodo) => {
        let reqRed = '';
        if (red) {
            reqRed = 'red=' + red
        }
        let reqDataset = '';
        if (dataset) {
            reqDataset = 'dataset=' + dataset;
        }
        let reqPista = ''
        if (pista) {
            reqPista = 'pista=' + pista;
        }
        let reqMetodo = ''
        if (metodo) {
            reqMetodo = 'metodo=' + metodo
        }
        let params = ''
        if (reqRed !== '' && reqDataset !== '' && reqPista !== '') {
            params = '?' + reqRed + '&' + reqDataset + '&' + reqPista + '&' + reqMetodo
        }
        const data = await fetch(URL + '/representacion' + params);
        const rep = await data.json()
        console.log(rep)
        return rep;
    }

    return {
        getEmbeddingsTaggrams,
        obtenerAudios,
        obtenerRepresentacion,
        obtenerTags
    }
}