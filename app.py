# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import plotly.graph_objects as go

import dash
from dash.dependencies import Input, Output, State
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
import random

import base64

from dash import dash_table
import json
import math
from skimage import measure

import utils
import pathlib as pl
import os

external_stylesheets = [dbc.themes.BOOTSTRAP, "assets/segmentation-style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.title = "Interactive image segmentation based on machine learning"

APP_PATH = str(pl.Path(__file__).parent.resolve())

DATA_CFG = {1: {'desc': 'Neoplastic',
                'color': 'rgba(255,0,0,1)',
                'color_tr': 'rgba(0,0,0,0)'},
            2: {'desc': 'No-neoplastic',
                'color': 'rgba(0,255,0,1)',
                'color_tr': 'rgba(0,0,0,0)'}}


def get_data_descript(ann_type, vis_type):
    if ann_type == 'GT':
        if vis_type == 'SEGM':
            data = {'classe': {'name': 'Class',
                               'type': 'numeric'},
                    'descr_class': {'name': 'Descr',
                                    'type': 'text'},
                    'area': {'name': 'Area',
                             'type': 'numeric'},
                    'perimeter': {'name': 'Perimeter',
                                  'type': 'numeric'},
                    'circularity': {'name': 'Circularity',
                                    'type': 'numeric'},
                    'solidity': {'name': 'Solidity',
                                 'type': 'numeric'}
                    }
            return data
        elif vis_type == 'BBOX':
            data = {'classe': {'name': 'Class',
                               'type': 'numeric'},
                    'descr_class': {'name': 'Descr',
                                    'type': 'text'}
                    }
            return data
    elif ann_type == 'PRED':
        if vis_type == 'SEGM':
            data = {'id': {'name': 'Id',
                           'type': 'numeric'},
                    'score': {'name': 'Score',
                              'type': 'numeric'},
                    'classe': {'name': 'Class',
                               'type': 'numeric'},
                    'descr_class': {'name': 'Descr',
                                    'type': 'text'},
                    'area': {'name': 'Area',
                             'type': 'numeric'},
                    'perimeter': {'name': 'Perimeter',
                                  'type': 'numeric'},
                    'circularity': {'name': 'Circularity',
                                    'type': 'numeric'},
                    'solidity': {'name': 'Solidity',
                                 'type': 'numeric'}
                    }
            return data
        elif vis_type == 'BBOX':
            data = {'id': {'name': 'Id',
                           'type': 'numeric'},
                    'score': {'name': 'Score',
                              'type': 'numeric'},
                    'classe': {'name': 'Class',
                               'type': 'numeric'},
                    'descr_class': {'name': 'Descr',
                                    'type': 'text'}
                    }
            return data


def read_local_file(filename, folder):
    file_path = os.path.join(APP_PATH, os.path.join(folder, filename))
    file_extension = os.path.splitext(filename)[1]

    if file_extension in ['.png', '.jpg']:
        encoded_image = base64.b64encode(open(file_path, 'rb').read())
        return 'data:image/png;base64,{}'.format(encoded_image.decode('utf-8'))
    elif file_extension == '.json':
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
    elif file_extension == '.md':
        with open(file_path, 'r') as f:
            return f.read()


button_howto = dbc.Button(
    "Learn more",
    id="howto-open",
    outline=True,
    color="info"
)

button_github = dbc.Button(
    "View Code on github",
    outline=True,
    color="primary",
    href="https://github.com/zuccaandrea/breast-neopl-cells-segm",
    id="gh-link"
)

modal_overlay = dbc.Modal(
    [
        dbc.ModalBody(html.Div([
            html.H5('Autore'),
            html.Img(src=read_local_file('author.jpg', 'docs'), style={'height': '35%', 'width': '35%'}),
            html.P('Andrea Zucca', id="author"),
            html.A("LinkedIn", href='https://www.linkedin.com/in/andrea-zucca-62b6a6174/'),
            dcc.Markdown(read_local_file('app.md', 'docs')),
            dcc.Markdown(read_local_file('working.md', 'docs')),
            dcc.Markdown(read_local_file('dataset_1.md', 'docs')),
            html.Img(src=read_local_file('distrib.png', 'docs'),
                     style={'height': '75%', 'width': '75%', 'display': 'block', 'margin': 'auto'}),
            dcc.Markdown('''**Figura 1.** Distribuzione delle cellule nelle 5 classi''', style={'font-size': '0.9em'}),
            dcc.Markdown(read_local_file('dataset_2.md', 'docs')),
            html.Img(src=read_local_file('merging.png', 'docs'),
                     style={'height': '85%', 'width': '85%', 'display': 'block', 'margin': 'auto'}),
            dcc.Markdown(
                '''**Figura 2.** Procedura di merging delle matrici contenenti le maschere di segmentazione delle cellule non-neoplastiche''',
                style={'font-size': '0.9em'}),
            dcc.Markdown(read_local_file('dataset_3.md', 'docs')),
            dcc.Markdown(read_local_file('training.md', 'docs')),
            html.Img(src=read_local_file('loss.png', 'docs'),
                     style={'height': '60%', 'width': '60%', 'display': 'block', 'margin': 'auto'}),
            dcc.Markdown(
                '''**Figura 3.** Perdita totale sul training e validation set. La perdita sul validation set è stata calcolata ogni 100 iterazioni di addestramento. Le linee verticali indicano un cambio di learning rate. Le prime 1000 iterazioni sono di warm up.''',
                style={'font-size': '0.9em'}),
            dcc.Markdown(read_local_file('evaluation_1.md', 'docs')),
            html.Img(src=read_local_file('4_fold_cross_validation.png', 'docs'),
                     style={'height': '70%', 'width': '70%', 'display': 'block', 'margin': 'auto'}),
            dcc.Markdown('''**Figura 4.** Partizionamento del dataset per la 4-fold cross-validation.''',
                         style={'font-size': '0.9em'}),
            dcc.Markdown(read_local_file('evaluation_2.md', 'docs')),
            html.Img(src=read_local_file('ap.png', 'docs'), style={'height': '100%', 'width': '100%'}),
            dcc.Markdown(
                '''**Figura 5.** Average Precision (AP), secondo la [definizione utilizzata da COCO](https://cocodataset.org/#detection-eval).''',
                style={'font-size': '0.9em'}),
            dcc.Markdown(read_local_file('evaluation_3.md', 'docs')),
            html.Img(src=read_local_file('ar.png', 'docs'), style={'height': '100%', 'width': '100%'}),
            dcc.Markdown('''**Figura 6.** Average Recall (AR), secondo COCO.''', style={'font-size': '0.9em'}),
            dcc.Markdown(read_local_file('future_development.md', 'docs')),
            dcc.Markdown(read_local_file('credits.md', 'docs')),
            dcc.Markdown(read_local_file('references.md', 'docs'))
        ], id="howto-md")),
        dbc.ModalFooter(dbc.Button(
            "Close", id="howto-close", className="howto-bn")),
    ],
    id="modal",
    size="lg",
)

# Header
header = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Img(
                            id="logo",
                            src=app.get_asset_url("dash-logo-new.png"),
                            height="30px",
                        ),
                        md="auto",
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H3("Neoplastic cells detection and segmentation in breast tissue"),
                                    html.P("Deep-learning based instance segmentation"),
                                ],
                                id="app-title",
                            )
                        ],
                        md=True,
                        align="center",
                    ),
                ],
                align="center",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.NavbarToggler(id="navbar-toggler"),
                            dbc.Collapse(
                                dbc.Nav(
                                    [
                                        dbc.NavItem(button_howto),
                                        dbc.NavItem(button_github),
                                    ],
                                    navbar=True,
                                ),
                                id="navbar-collapse",
                                navbar=True,
                            ),
                            modal_overlay,
                        ],
                        md=2,
                    ),
                ],
                align="center",
            ),
        ],
        fluid=True,
    ),
    dark=True,
    color="dark",
    sticky="top",
)

# Image Segmentation

gt_card = dbc.Card(
    [
        dbc.CardHeader("Ground truth"),
        dbc.CardBody(
            dcc.Graph(
                id='graph-gt',
                config={
                    'displayModeBar': False,
                    'showlegend': False}
            )
        ),
        dbc.CardFooter(dbc.Row(
            [dbc.Col([dbc.Button('Random tissue', id='submit-val', n_clicks=0, color="primary")], md=7),
             dbc.Col([html.P("Type:"),
                      dcc.Dropdown(
                          id='dropdown-visual-type',
                          options=[
                              {'label': 'SEGM', 'value': 'SEGM'},
                              {'label': 'BBOX', 'value': 'BBOX'}
                          ],
                          value='SEGM',
                          clearable=False)
                      ], md=5)],
        ),
            className="cfooter"
        ),
    ]
)

pred_card = dbc.Card(
    [
        dbc.CardHeader("Prediction"),
        dbc.CardBody(
            dcc.Graph(
                id='graph-pred',
                config={
                    'displayModeBar': False,
                    'showlegend': False}
            )
        ),
        dbc.CardFooter([
            html.P(id="out-info-threshold"),
            dcc.Slider(
                id="threshold-slider",
                min=0,
                max=1,
                value=0.5,
                marks={str(thr): thr for thr in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]},
                step=0.05
            ),
        ],
            className="cfooter"
        ),
    ]
)

distr_card = dbc.Card(
    [
        dbc.CardHeader("Distribution"),
        dbc.CardBody(
            [
                dcc.Graph(
                    id='distr-plot',
                    config={
                        'displayModeBar': False,
                        'showlegend': False}
                ),
            ],
        ),
    ],
)

summ_card = dbc.Card(
    [
        dbc.CardHeader("Summary"),
        dbc.CardBody(
            [
                dash_table.DataTable(
                    id='summary-data-table',
                    editable=True,
                    filter_action="native",
                    sort_action="native",
                    sort_mode="multi",
                    selected_columns=[],
                    export_format='xlsx',
                    export_headers='display',
                    merge_duplicate_headers=True,
                    page_action="native",
                    page_current=0,
                    page_size=10,
                    style_as_list_view=True,
                    style_header={'fontWeight': 'bold'},
                    style_table={'overflowX': 'auto'},
                ),
            ],
        ),
    ],
)

meta = [
    html.Div(
        id="no-display",
        children=[
            # Store for user created masks
            # data is a list of dicts describing shapes
            dcc.Store(id='image-id'),
            dcc.Store(id='pred-summary-data'),
            dcc.Store(id='gt-summary-data'),
            dcc.Store(id='base64-img-string'),
            dcc.Store(id='json-annotation-demo', data=read_local_file('json_annotation_demo.json', 'data')),
            dcc.Store(id='json-pred-annotation-demo', data=read_local_file('json_pred_annotation_demo.json', 'data'))
        ],
    ),
]

app.layout = html.Div(
    [
        header,
        dbc.Container(id='root-container',
                      children=[
                          dbc.Container(id="app-container",
                                        children=
                                        [
                                            dbc.Row(
                                                id="app-content",
                                                children=[dbc.Col(gt_card, md=4),
                                                          dbc.Col(pred_card, md=4),
                                                          dbc.Col(distr_card, md=4)],
                                            ),
                                            dbc.Row(id="summary-content",
                                                    children=[dbc.Col(summ_card, md=12),
                                                              dbc.Col(meta)
                                                              ],
                                                    ),
                                        ],
                                        fluid=True,

                                        ),
                      ],
                      fluid=True,
                      ),
    ]
)


def convert_bbox(bbox, type_ann):
    if type_ann == 'GT':
        x1 = round(bbox[0], 4)
        y1 = round(bbox[1], 4)
        width = round(bbox[2], 4)
        height = round(bbox[3], 4)

        x0 = x1
        y0 = y1 + height
        x2 = width + x1
        y2 = y1
        x3 = x2
        y3 = y0

        return [x0, x1, x2, x3, x0], [y0, y1, y2, y3, y0]

    elif type_ann == 'PRED':

        x1 = round(bbox[0], 4)
        y1 = round(bbox[1], 4)
        x3 = round(bbox[2], 4)
        y3 = round(bbox[3], 4)

        x0 = x1
        y0 = y3

        x2 = x3
        y2 = y1

        return [x0, x1, x2, x3, x0], [y0, y1, y2, y3, y0]


def get_summary(json_annotation, image_id, ann_type='GT'):
    summary = []

    for ann in json_annotation['annotations']:

        if ann['image_id'] == image_id:

            # Id detection
            id_detection = ann['id']

            # Predicted class
            classe = ann['category_id']

            # Description class
            classes_descr = DATA_CFG[classe]['desc']

            # Measuring region’s properties
            rle = ann['segmentation']
            mask = utils.RleToMask(rle)

            props = measure.regionprops_table(mask, properties=('area', 'perimeter', 'solidity'))

            # cell's area
            area = round(props['area'][0], 4)

            # cell's perimeter
            perimeter = round(props['perimeter'][0], 4)

            # cell's circularity
            try:
                circularity = round((4 * math.pi) * (area / pow(perimeter, 2)), 4)
            except:
                circularity = -1

            # cell's solidity
            solidity = round(props['solidity'][0], 4)

            # Detect contours from mask
            # mask = np.pad(mask, [(1,), (1,)], mode='constant')
            contours = measure.find_contours(mask, level=0.5)

            mask_contours = (contours[0][:, 1], contours[0][:, 0])

            bbox = convert_bbox(ann['bbox'], ann_type)

            if ann_type == 'GT':
                summary.append({"id": id_detection, "classe": classe, 'descr_class': classes_descr, "area": area,
                                "perimeter": perimeter, "circularity": circularity, "solidity": solidity,
                                "mask_contour": mask_contours, "bbox": bbox})
            elif ann_type == 'PRED':
                score = round(ann['score'], 4)

                summary.append(
                    {"id": id_detection, "classe": classe, 'descr_class': classes_descr, "score": score, "area": area,
                     "perimeter": perimeter, "circularity": circularity, "solidity": solidity,
                     "mask_contour": mask_contours, "bbox": bbox})

    return summary


@app.callback(Output('out-info-threshold', 'children'),
              Input('threshold-slider', 'value'))
def info_tresh(threshold_slider):
    return f"Confidence Threshold: {threshold_slider}"


@app.callback(Output('gt-summary-data', 'data'),
              [Input('image-id', 'data'),
               Input('json-annotation-demo', 'data')])
def detect_gt(image_id, json_annotation_demo):
    return get_summary(json_annotation_demo, image_id, ann_type='GT')


@app.callback(Output('pred-summary-data', 'data'),
              [Input('image-id', 'data'),
               Input('json-pred-annotation-demo', 'data')])
def detect_pred(image_id, json_pred_annotation_demo):
    return get_summary(json_pred_annotation_demo, image_id, ann_type='PRED')


@app.callback(Output('image-id', 'data'),
              [Input('submit-val', 'n_clicks'),
               Input('json-annotation-demo', 'data')])
def update_image_id(n_clicks, json_annotation_demo):
    img_ids = [i['image_id'] for i in json_annotation_demo['annotations']]
    return img_ids[random.randint(0, len(img_ids))]


@app.callback(Output('base64-img-string', 'data'),
              Input('image-id', 'data'))
def update_output(image_id):
    if image_id is not None:
        return read_local_file(f'{image_id:08}.png', 'data')


def draw_figure(base64_img_string, summary_data, ann_type, visual_type):
    data_descript = get_data_descript(ann_type, visual_type)

    fig = go.Figure(go.Image(source=base64_img_string))
    fig.update_traces(hoverinfo="skip", hovertemplate=None)

    for ann in summary_data:

        hover = ''

        if visual_type == 'SEGM':
            x_list = ann['mask_contour'][0]
            y_list = ann['mask_contour'][1]
        elif visual_type == 'BBOX':
            x_list = ann['bbox'][0]
            y_list = ann['bbox'][1]
        else:
            x_list = None
            y_list = None

        for dx in data_descript:
            hover = hover + f'<b>{data_descript[dx]["name"]}:</b> {ann[dx]}<br>'

        fig.add_trace(
            go.Scatter(
                x=x_list,
                y=y_list,
                fill='toself',
                mode='lines',
                fillcolor=DATA_CFG[ann['classe']]['color_tr'],
                line_color=DATA_CFG[ann['classe']]['color'],
                text=hover,
                name='',
                hoveron='fills'
            )
        )

    config = dict({
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'showlegend': False
    })

    fig.update_layout(height=300, margin=dict(l=10, r=10, b=10, t=10), template="simple_white")
    fig.update_xaxes(visible=False, range=[0, 256]).update_yaxes(visible=False, range=[256, 0])

    fig.update_layout(config)

    return fig


@app.callback(
    Output('graph-pred', 'figure'),
    [Input('base64-img-string', 'data'),
     Input('threshold-slider', 'value'),
     Input('pred-summary-data', 'data'),
     Input('dropdown-visual-type', 'value')])
def update_figure_pred(base64_img_string, threshold_slider, pred_summary_data, visual_type):
    summary = [ann for ann in pred_summary_data if ann['score'] >= threshold_slider]
    return draw_figure(base64_img_string, summary, 'PRED', visual_type)


@app.callback(
    Output('graph-gt', 'figure'),
    [Input('base64-img-string', 'data'),
     Input('gt-summary-data', 'data'),
     Input('dropdown-visual-type', 'value')])
def update_figure_gt(base64_img_string, summary_data, visual_type):
    return draw_figure(base64_img_string, summary_data, 'GT', visual_type)


@app.callback(
    Output('distr-plot', 'figure'),
    [Input('threshold-slider', 'value'),
     Input('pred-summary-data', 'data')])
def update_distr_plot(threshold_slider, summary_data):
    data = {1: 0,
            2: 0}

    for ann in summary_data:
        if ann['score'] >= threshold_slider:
            if ann['classe'] == 1:
                data[1] += 1
            elif ann['classe'] == 2:
                data[2] += 1

    tot = sum(data.values())

    fig = go.Figure(data=[go.Bar(
        x=[DATA_CFG[k]['desc'] for k in data.keys()],
        y=list(data.values()),
        text=['' if v == 0 else f'{str(round(v / tot * 100, 2))}%' for v in data.values()],
        marker_color=[DATA_CFG[k]['color'] for k in data.keys()],
        textposition='auto',
    )])

    fig.update_layout(xaxis_tickangle=45)

    config = dict({
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'showlegend': False,
        'margin': dict(l=20, r=20, t=20, b=20)
    })

    fig.update_layout(config)

    return fig


@app.callback(
    [Output('summary-data-table', 'columns'),
     Output('summary-data-table', 'data'),
     Output('summary-data-table', 'style_data_conditional')],
    [Input('threshold-slider', 'value'),
     Input('pred-summary-data', 'data')])
def update_table(threshold_slider, summary_data):
    data_descript = get_data_descript('PRED', 'SEGM')

    summary = []

    for ann in summary_data:
        if ann['score'] >= threshold_slider:
            del ann['mask_contour']
            del ann['bbox']
            summary.append(ann)

    columns = [{"name": data_descript[i]['name'], "id": i, "type": data_descript[i]['type'],
                "hideable": True} for i in data_descript.keys()]

    style_data_conditional = [{
        'if': {
            'filter_query': '{score} <= 0.5',
            'column_id': 'score'
        },
        'color': 'red',
        'fontWeight': 'bold'
    }]

    return columns, summary, style_data_conditional


# Callback for modal popup
@app.callback(
    Output("modal", "is_open"),
    [Input("howto-open", "n_clicks"), Input("howto-close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


if __name__ == "__main__":
    app.run_server()
