import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css'
import { Grafica } from './components/Grafica';
import { SidePane } from './components/SidePane';
import { useMakeRequest } from './hooks/useMakeRequest';

import { useEffect, useState } from 'react';

function App() {

  // Selector SidePane
  const [tags, setTags] = useState([]);
  const [listaTags, setListaTags] = useState([]);
  const [canciones, setCanciones] = useState([]);
  const [listaCanciones, setListaCanciones] = useState([]);

  // Selector Grafica 1
  const [arquitectura1, setArquitectura1] = useState('MusiCNN');
  const [dataset1, setDataset1] = useState('MSD');
  const [tipoGrafica1, setTipoGrafica1] = useState('PCA')
  const [graf1, setGraf1] = useState([]);

  const layout = { width: 500, height: 500, title: { text: '' } }
  // Selector Grafica 2
  const [arquitectura2, setArquitectura2] = useState('MusiCNN');
  const [dataset2, setDataset2] = useState('MSD');
  const [tipoGrafica2, setTipoGrafica2] = useState('PCA')
  const [graf2, setGraf2] = useState([]);

  const { obtenerAudios, obtenerRepresentacion, obtenerTags } = useMakeRequest();


  const cargarDatos = async () => {
    console.log('Ejecutando request...',arquitectura1, dataset1, tipoGrafica1)
    // Primera grafica
    let primeraGrafica = []
    for (let can of canciones) {
      console.log('Obteniendo para ',can)
      primeraGrafica.push(await obtenerRepresentacion(arquitectura1, dataset1, can, tipoGrafica1))
    }
    console.log(primeraGrafica)
    const plotData1 = primeraGrafica.map((resp, idx) => {
      const key = Object.keys(resp)[0];
      const coords = resp[key];

      const x = coords.map(point => point[0]);
      const y = coords.map(point => point[1]);

      return {
        x,
        y,
        type: 'scatter',
        mode: 'markers',
        name: canciones[idx]
      };
    });
    setGraf1(plotData1)

    console.log('Ejecutando request...',arquitectura2, dataset2, tipoGrafica2)
    // Segunda grafica
    let segundaGrafica = []
    for (let can of canciones) {
      console.log('Obteniendo para ',can)
      segundaGrafica.push(await obtenerRepresentacion(arquitectura2, dataset2, can, tipoGrafica2))
    }
    console.log(segundaGrafica)
    const plotData2 = segundaGrafica.map((resp, idx) => {
      const key = Object.keys(resp)[0];
      const coords = resp[key];

      const x = coords.map(point => point[0]);
      const y = coords.map(point => point[1]);

      return {
        x,
        y,
        type: 'scatter',
        mode: 'markers',
        name: canciones[idx]
      };
    });
    setGraf2(plotData2)
  }

  useEffect(() => {
    obtenerAudios().then(data => {
      setListaCanciones(data)
    })
    obtenerTags().then(data=>{
      setListaTags(data);
    })
  }, [])

  return (
    <div className='main'>
      <div className='mainsection'>
        <SidePane listaTags={listaTags} tags={listaTags} setTags={setTags} canciones={canciones} setCanciones={setCanciones} listaCanciones={listaCanciones} cargarDatos={cargarDatos}></SidePane>
        <div className='graficas'>
          <Grafica arquitectura={arquitectura1} setArquitectura={setArquitectura1} dataset1={dataset1} setDataset={setDataset1} layout={layout} data={graf1} tipoGrafica={tipoGrafica1} setTipoGrafica={setTipoGrafica1}></Grafica>
          <Grafica arquitectura={arquitectura2} setArquitectura={setArquitectura2} dataset1={dataset2} setDataset={setDataset2} layout={layout} data={graf2} tipoGrafica={tipoGrafica2} setTipoGrafica={setTipoGrafica2}></Grafica>
        </div>
      </div>
    </div>
  );
}

export default App;
