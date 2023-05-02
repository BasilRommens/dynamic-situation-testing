import dash
import plotly.graph_objects as go
from dash import html, dcc, Output, Input, no_update, State, dash_table, ALL, \
    MATCH
import dash_bootstrap_components as dbc

from flask_app.dashboard.layout import html_layout
from setup import data

click_shapes = list()


def serve_layout():
    # if there is the plot key, and it is set to false then no plot is drawn
    # add a link to dynamic page
    if 'plot' in data and not data['plot']:
        return html.Div([
            dbc.Container([
                dbc.Row([
                    dbc.Col(
                        html.H3('Discrimination Plot'),
                    )
                ], class_name='mt-3'),
                dbc.Row([
                    dbc.Col(
                        html.P('There is nothing to plot')
                    )
                ], class_name='mt-3'),
                dbc.Row([
                    dbc.Col(
                        html.A('Go to Dynamic Page',
                               href='/dynamic',
                               className='btn btn-primary mt-3'),
                        width=2
                    )
                ], class_name='my-3'),
            ]),
            dcc.Store(id='graph_store_layout'),
            dcc.Store(id='graph_store_style'),
        ])

    graph = [
        html.H3("Discrimination Plot"),
        dcc.Graph(
            id="graph",
            figure=data['fig'],
            clear_on_unhover=True,
            style={
                'display': 'inline-block'}
        )]

    contour_form = [
        html.H3("Add Contours", className='mt-2'),
        html.Form(
            data['contour_form'],
            id="form",
            action='/new_contour',
            method='POST',
            className='mb-3'
        ),
    ]
    contour_rm_btns = [
        html.Div(data['contour_html'],
                 id='contour_buttons'),
    ]
    table = [
        html.H3("Table of Discriminated Tuples"),
        dash_table.DataTable(data['table'],
                             [{"name": key, "id": key}
                              for key in
                              data['table'][0].keys()],
                             id='tbl')
        if len(data['table']) > 0 else
        dash_table.DataTable(id='tbl'),
    ]
    neighbor_table = [html.H3("Table of Neighboring Tuples")]
    if 'neighbor_tbl' in data:
        neighbor_table.append(
            html.Div(
                children=dash_table.DataTable(data['neighbor_tbl'],
                                              data['neighbor_cols'],
                                              id='neighbor_tbl'),
                id='neighbor_div',
                className='mb-3',
            )
        )
    else:
        # TODO clean this up by adding grey text
        neighbor_table.append(
            html.Div(
                'No neighboring tuples. Select before seeing a table of neighboring tuples.',
                id='neighbor_div'))

    return html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col(graph, width=8),
                dbc.Col([
                    dbc.Row(dbc.Col(contour_form)),
                    dbc.Row(dbc.Col()),
                    dbc.Row(dbc.Col(contour_rm_btns),
                            class_name="position-absolute bottom-0 start-0")
                ], width=4, class_name="position-relative",
                    style={'background-color': '#f8f9fa',
                           'corner-radius': '5px'})
            ], class_name='mt-3'),
            dbc.Row(dbc.Col(neighbor_table), className='mb-3'),
            dbc.Row(dbc.Col(table), class_name='mb-3'),
        ]),
        dcc.Store(id='graph_store_layout'),
        dcc.Store(id='graph_store_style'),
    ])


def init_dashboard(server):
    global data

    dash_app = dash.Dash(
        server=server,
        routes_pathname_prefix='/dashapp/',
        assets_folder='static/',
    )

    # custom HTML layout
    dash_app.index_string = html_layout

    # create Dash layout
    dash_app.layout = serve_layout
    init_callbacks(dash_app)

    return dash_app.server


def init_callbacks(dash_app):
    global data

    @dash_app.callback(
        Output('featureOptions', 'children'),
        Input('feature', 'value'),
        prevent_initial_call=True
    )
    def update_contour_form(value):
        global data

        contour_form_options = data['contour_form_options']
        for idx, contour_form_option in enumerate(contour_form_options):
            if data['features'][idx] == value:
                return contour_form_option

    @dash_app.callback(
        Output('graph_store_style', 'data'),
        Input('graph', 'restyleData')
    )
    def update_restyle_store(restyle_data):
        return restyle_data

    @dash_app.callback(
        Output('graph_store_layout', 'data'),
        Input('graph', 'relayoutData'),
    )
    def update_layout_store(relayout_data):
        return relayout_data

    @dash_app.callback(
        [Output("graph", "figure", allow_duplicate=True),
         Output("neighbor_div", "children"), ],
        [Input("graph", "clickData")],
        prevent_initial_call=True,
    )
    def update_click(clickData):
        global data

        if clickData is None:
            return no_update, no_update, no_update

        pt = clickData["points"][0]
        if 'hovertext' not in pt.keys():
            return no_update, no_update, no_update

        pt_coords = pt['x'], pt['y']
        pt_idx = data['data_pts'][(data['data_pts']['x'] == pt_coords[0]) & (
                data['data_pts']['y'] == pt_coords[1])].index[0]

        # find all the neighbors of the point, both protected and unprotected
        prot_pts = list()
        unprot_pts = list()
        for valid_tuple in data['valid_tuples']:
            if valid_tuple[0] == pt_idx:
                prot_pts = list(dict(valid_tuple[2]).keys())
                unprot_pts = list(dict(valid_tuple[3]).keys())

        # list containing all information of the neighbors
        neighbors = list()

        # information about the protected neighbors of the current
        for prot_pt in prot_pts:
            prot_pt_data = data['data_pts'].iloc[prot_pt].to_dict()
            del prot_pt_data['x']
            del prot_pt_data['y']
            prot_pt_data['is_protected'] = 'protected'
            neighbors.append(prot_pt_data)

        # information of all the unprotected neighbors
        for unprot_pt in unprot_pts:
            unprot_pt_data = data['data_pts'].iloc[unprot_pt].to_dict()
            del unprot_pt_data['x']
            del unprot_pt_data['y']
            unprot_pt_data['is_protected'] = 'unprotected'
            neighbors.append(unprot_pt_data)
        # add the current data point also to the neighbors
        data_pt = data['data_pts'].iloc[pt_idx].to_dict()
        del data_pt['x']
        del data_pt['y']
        data_pt['is_protected'] = 'selected point'
        neighbors.insert(0, data_pt)

        # set the neighbor table in the data dictionary
        data['neighbor_tbl'] = neighbors

        # set the columns in the data dictionary
        data['neighbor_cols'] = [{'name': i, 'id': i} for i in
                                 data['neighbor_tbl'][0].keys()]
        neighbor_tbl = dash_table.DataTable(data=data['neighbor_tbl'],
                                            columns=data['neighbor_cols'],
                                            id='neighbor_tbl')

        # if the point isn't a sensitive tuple, return the figure
        if not len(prot_pts) and not len(unprot_pts):
            return no_update, no_update, no_update

        # clear all the line shapes
        data['fig']['layout']['shapes'] = list()
        # find the coordinates of the protected points
        get_coords = lambda x: data['data_pts'].iloc[x][['x', 'y']].values
        prot_coords = list(map(get_coords, prot_pts))
        unprot_coords = list(map(get_coords, unprot_pts))

        # create the protected lines
        temp_fig = go.Figure()
        for prot_coord in prot_coords:
            temp_fig.add_shape(type="line",
                               x0=pt_coords[0], y0=pt_coords[1],
                               x1=prot_coord[0], y1=prot_coord[1],
                               line=dict(width=1, color='red'))
        # create the unprotected lines
        for unprot_coord in unprot_coords:
            temp_fig.add_shape(type="line",
                               x0=pt_coords[0], y0=pt_coords[1],
                               x1=unprot_coord[0], y1=unprot_coord[1],
                               line=dict(width=1, color='green'))

        # add the lines to the figure
        data['fig']['layout']['shapes'] = temp_fig['layout']['shapes']

        # add the shapes to the click shapes
        data['click_shapes'] = temp_fig['layout']['shapes']

        return data['fig'], neighbor_tbl

    @dash_app.callback(
        Output("graph", "figure"),
        [Input("graph", "hoverData"),
         State("graph_store_layout", "data"),
         State("graph_store_style", "data")],
    )
    def update_hover(hoverData, graph_store_layout, graph_store_style):
        global data

        # clear all the line shapes, apart from the ones created by clicking
        data['fig']['layout']['shapes'] = data['click_shapes']

        # keep the zoom level and everything when hovering using graph_store
        if graph_store_layout:
            if 'xaxis.autorange' in graph_store_layout or 'autosize' in graph_store_layout:
                data['fig']['layout']['xaxis']['autorange'] = True
            else:
                data['fig']['layout']['xaxis']['autorange'] = False
            if 'yaxis.autorange' in graph_store_layout or 'autosize' in graph_store_layout:
                data['fig']['layout']['yaxis']['autorange'] = True
            else:
                data['fig']['layout']['yaxis']['autorange'] = False

            for axis_type in ['x', 'y']:
                for axis_name in ['axis', 'axis2']:
                    if f'{axis_type}{axis_name}.range[0]' not in graph_store_layout:
                        continue
                    data['fig']['layout'][f'{axis_type}{axis_name}'][
                        'range'] = [
                        graph_store_layout[f'{axis_type}{axis_name}.range[0]'],
                        graph_store_layout[f'{axis_type}{axis_name}.range[1]']
                    ]

        # keep hidden traces hidden and show traces that weren't hidden
        if graph_store_style:
            for idx, trace_idx in enumerate(graph_store_style[1]):
                data['fig'].data[trace_idx].update(
                    visible=graph_store_style[0]['visible'][idx])

        if hoverData is None:
            return data['fig']

        pt = hoverData["points"][0]
        if 'hovertext' not in pt.keys():
            return data['fig']

        pt_coords = pt['x'], pt['y']
        try:
            pt_idx = \
                data['data_pts'][(data['data_pts']['x'] == pt_coords[0]) & (
                        data['data_pts']['y'] == pt_coords[1])].index[0]
        except IndexError:
            print('no point index found')
            return data['fig']

        # find all the neighbors of the point, both protected and unprotected
        prot_pts = list()
        unprot_pts = list()
        for valid_tuple in data['valid_tuples']:
            if valid_tuple[0] == pt_idx:
                prot_pts = list(dict(valid_tuple[2]).keys())
                unprot_pts = list(dict(valid_tuple[3]).keys())

        # if the point isn't a sensitive tuple, return the figure
        if not len(prot_pts) and not len(unprot_pts):
            return data['fig']

        # find the coordinates of the protected points
        get_coords = lambda x: data['data_pts'].iloc[x][['x', 'y']].values
        prot_coords = list(map(get_coords, prot_pts))
        unprot_coords = list(map(get_coords, unprot_pts))

        # create the protected lines
        for prot_coord in prot_coords:
            data['fig'].add_shape(type="line",
                                  x0=pt_coords[0], y0=pt_coords[1],
                                  x1=prot_coord[0], y1=prot_coord[1],
                                  line=dict(width=1,
                                            color='rgb(256, 170, 170)'))
        # create the unprotected lines
        for unprot_coord in unprot_coords:
            data['fig'].add_shape(type="line",
                                  x0=pt_coords[0], y0=pt_coords[1],
                                  x1=unprot_coord[0], y1=unprot_coord[1],
                                  line=dict(width=1,
                                            color='rgb(170, 256, 170)'))

        return data['fig']
