# coding: utf-8

import plotly
from plotly.graph_objs import Layout, Figure, Marker
import numpy as np
import pandas as pd


def plot_precision_recall_fscore(precision, recall, fscore, filename='new_result.html'):

    avg_data = np.array([
        np.mean(precision),
        np.mean(recall),
        np.mean(fscore)
    ])

    panda_avg = pd.DataFrame(avg_data).round(3)

    data = np.array(panda_avg.values).T[0]

    print(avg_data)
    print(panda_avg)
    print(data)

    trace = [plotly.graph_objs.Bar(
        x=['precision', 'recall', 'fscore'],
        y=data,
        marker=Marker(color='#04b486'),
        text=data,
        textposition='auto',
        textfont=dict(color='brack', size='20')
    )]

    layout = Layout(bargap=0.6, plot_bgcolor='#E6E6E6')
    plot_data = Figure(data=trace, layout=layout)
    plotly.offline.plot(plot_data, filename=filename)