import { SelectorGrafico } from "./SelectorGrafico"
import Plot from 'react-plotly.js';

export const Grafica = ({arquitectura, setArquitectura, dataset, setDataset,data,layout, tipoGrafica,setTipoGrafica, izq}) => {

    return (
        <div className="divGrafico" style={izq?{borderRight: '1px solid black'}:{}}>
            <SelectorGrafico arquitectura={arquitectura} setArquitectura={setArquitectura} dataset1={dataset} setDataset={setDataset} tipoGrafica={tipoGrafica} setTipoGrafica={setTipoGrafica}></SelectorGrafico>
            <div className="grafica">
                <Plot
                    data={data}
                    layout={layout }
                />
            </div>
        </div>
    );
}