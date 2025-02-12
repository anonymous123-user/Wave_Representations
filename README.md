# Traveling Waves Integrate Spatial Information Into Spectral Representations

## Figure 1: Single Polygon
<img src="gifs/wave_hexagon.gif" alt="Figure 1 hexagon"/>

## coRNN states - polygons
<table>
  <tr>
    <td><img src="gifs/polygon_0.gif" alt="Polygon shape 0" width="200"/></td>
    <td><img src="gifs/polygon_1.gif" alt="Polygon shape 1" width="200"/></td>
    <td><img src="gifs/polygon_2.gif" alt="Polygon shape 2" width="200"/></td>
  </tr>
  <tr>
    <td><img src="gifs/polygon_3.gif" alt="Polygon shape 3" width="200"/></td>
    <td><img src="gifs/polygon_4.gif" alt="Polygon shape 4" width="200"/></td>
    <td><img src="gifs/polygon_5.gif" alt="Polygon shape 5" width="200"/></td>
  </tr>
  <tr>
    <td><img src="gifs/polygon_6.gif" alt="Polygon shape 6" width="200"/></td>
    <td><img src="gifs/polygon_7.gif" alt="Polygon shape 7" width="200"/></td>
    <td><img src="gifs/polygon_8.gif" alt="Polygon shape 8" width="200"/></td>
  </tr>
</table>


## coRNN states
<table>
  <tr>
    <td><img src="gifs/cornn_sample-2.gif" alt="Description 1" width="200"/></td>
    <td><img src="gifs/cornn_sample-4.gif" alt="Description 2" width="200"/></td>
    <td><img src="gifs/cornn_sample-5.gif" alt="Description 3" width="200"/></td>
  </tr>
  <tr>
    <td><img src="gifs/cornn_sample-7.gif" alt="Description 4" width="200"/></td>
    <td><img src="gifs/cornn_sample-8.gif" alt="Description 5" width="200"/></td>
    <td><img src="gifs/cornn_sample-9.gif" alt="Description 6" width="200"/></td>
  </tr>
  <tr>
    <td><img src="gifs/cornn_sample-10.gif" alt="Description 7" width="200"/></td>
    <td><img src="gifs/cornn_sample-11.gif" alt="Description 8" width="200"/></td>
    <td><img src="gifs/cornn_sample-12.gif" alt="Description 9" width="200"/></td>
  </tr>
</table>

## LSTM states
<table>
  <tr>
    <td><img src="gifs/lstm_sample-2.gif" alt="Description 1" width="200"/></td>
    <td><img src="gifs/lstm_sample-4.gif" alt="Description 2" width="200"/></td>
    <td><img src="gifs/lstm_sample-5.gif" alt="Description 3" width="200"/></td>
  </tr>
  <tr>
    <td><img src="gifs/lstm_sample-7.gif" alt="Description 4" width="200"/></td>
    <td><img src="gifs/lstm_sample-8.gif" alt="Description 5" width="200"/></td>
    <td><img src="gifs/lstm_sample-9.gif" alt="Description 6" width="200"/></td>
  </tr>
  <tr>
    <td><img src="gifs/lstm_sample-10.gif" alt="Description 7" width="200"/></td>
    <td><img src="gifs/lstm_sample-11.gif" alt="Description 8" width="200"/></td>
    <td><img src="gifs/lstm_sample-12.gif" alt="Description 9" width="200"/></td>
  </tr>
</table>

## Reproducing results
1. Set up the conda environment using environment.yml.
2. Create data by running data_scripts/create_tetrominoes.py and data_scripts/download_mnist.py
3. Edit dataset_config.py with absolute paths to the generated data.
4. Train all the MNIST and Tetrominoes models using ./run_all_mnist.sh and ./run_all_tetro.sh
5. Produce Figure 3 using produce_figs3.ipynb.
6. Produce the scores for the table using produce_scores_4_most.ipynb and then produce_scores_4_last.ipynb
7. Produce polygons results using multi_polygon_classification_iclr.ipynb