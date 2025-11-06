import { SelectorGrafico } from "./SelectorGrafico"
import Plot from 'react-plotly.js';

export const Grafica = ({arquitectura, setArquitectura, dataset, setDataset,data,layout, tipoGrafica,setTipoGrafica}) => {

    return (
        <div className="divGrafico">
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