import dash
import plotly.graph_objects as go
from dash import html, dcc, Output, Input, no_update, State

from flask_app.dashboard.layout import html_layout
from setup import calc_german_credit_fig

click_shapes = list()


def init_dashboard(server):
    # calculate the figure to be used
    fig, data_pts, valid_tuples = calc_german_credit_fig()

    dash_app = dash.Dash(
        server=server,
        routes_pathname_prefix='/dashapp/',
    )

    # custom HTML layout
    dash_app.index_string = html_layout

    # create Dash layout
    dash_app.layout = html.Div(
        children=[
            dcc.Graph(
                id="graph",
                figure=fig,
                clear_on_unhover=True,
                style={'width': '100%', 'height': '100vh',
                       'display': 'inline-block'}),
            dcc.Store(id='graph_store_layout'),
            dcc.Store(id='graph_store_style')
        ]
    )
    init_callbacks(dash_app, fig, data_pts, valid_tuples)

    return dash_app.server


def init_callbacks(dash_app, fig, data_pts, valid_tuples):
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
        Output("graph", "figure", allow_duplicate=True),
        Input("graph", "clickData"),
        prevent_initial_call=True
    )
    def update_click(clickData):
        global click_shapes

        if clickData is None:
            return no_update

        pt = clickData["points"][0]
        if 'hovertext' not in pt.keys():
            return no_update

        pt_coords = pt['x'], pt['y']
        pt_idx = data_pts[(data_pts['x'] == pt_coords[0]) & (
                data_pts['y'] == pt_coords[1])].index[0]

        # find all the neighbors of the point, both protected and unprotected
        prot_pts = list()
        unprot_pts = list()
        for valid_tuple in valid_tuples:
            if valid_tuple[0] == pt_idx:
                prot_pts = list(dict(valid_tuple[2]).keys())
                unprot_pts = list(dict(valid_tuple[3]).keys())

        # if the point isn't a sensitive tuple, return the figure
        if not len(prot_pts) and not len(unprot_pts):
            return no_update

        # clear all the line shapes
        fig['layout']['shapes'] = list()
        # find the coordinates of the protected points
        get_coords = lambda x: data_pts.iloc[x][['x', 'y']].values
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
        fig['layout']['shapes'] = temp_fig['layout']['shapes']
        # add the lines to the figure
        click_shapes = temp_fig['layout']['shapes']

        return fig

    @dash_app.callback(
        Output("graph", "figure"),
        Input("graph", "hoverData"),
        [State("graph_store_layout", "data"),
         State("graph_store_style", "data")],
    )
    def update_hover(hoverData, graph_store_layout, graph_store_style):
        global click_shapes

        # clear all the line shapes, apart from the ones created by clicking
        fig['layout']['shapes'] = click_shapes

        # keep the zoom level and everything when hovering using graph_store
        if graph_store_layout:
            if 'xaxis.autorange' in graph_store_layout or 'autosize' in graph_store_layout:
                fig['layout']['xaxis']['autorange'] = True
            else:
                fig['layout']['xaxis']['autorange'] = False
            if 'yaxis.autorange' in graph_store_layout or 'autosize' in graph_store_layout:
                fig['layout']['yaxis']['autorange'] = True
            else:
                fig['layout']['yaxis']['autorange'] = False

            for axis_type in ['x', 'y']:
                for axis_name in ['axis', 'axis2']:
                    if f'{axis_type}{axis_name}.range[0]' not in graph_store_layout:
                        continue
                    fig['layout'][f'{axis_type}{axis_name}']['range'] = [
                        graph_store_layout[f'{axis_type}{axis_name}.range[0]'],
                        graph_store_layout[f'{axis_type}{axis_name}.range[1]']
                    ]

        # keep hidden traces hidden and show traces that weren't hidden
        if graph_store_style:
            for idx, trace_idx in enumerate(graph_store_style[1]):
                fig.data[trace_idx].update(
                    visible=graph_store_style[0]['visible'][idx])

        if hoverData is None:
            return fig

        pt = hoverData["points"][0]
        if 'hovertext' not in pt.keys():
            return fig

        pt_coords = pt['x'], pt['y']
        try:
            pt_idx = data_pts[(data_pts['x'] == pt_coords[0]) & (
                    data_pts['y'] == pt_coords[1])].index[0]
        except IndexError:
            print('no point index found')
            return fig

        # find all the neighbors of the point, both protected and unprotected
        prot_pts = list()
        unprot_pts = list()
        for valid_tuple in valid_tuples:
            if valid_tuple[0] == pt_idx:
                prot_pts = list(dict(valid_tuple[2]).keys())
                unprot_pts = list(dict(valid_tuple[3]).keys())

        # if the point isn't a sensitive tuple, return the figure
        if not len(prot_pts) and not len(unprot_pts):
            return fig

        # find the coordinates of the protected points
        get_coords = lambda x: data_pts.iloc[x][['x', 'y']].values
        prot_coords = list(map(get_coords, prot_pts))
        unprot_coords = list(map(get_coords, unprot_pts))

        # create the protected lines
        for prot_coord in prot_coords:
            fig.add_shape(type="line",
                          x0=pt_coords[0], y0=pt_coords[1],
                          x1=prot_coord[0], y1=prot_coord[1],
                          line=dict(width=1, color='rgb(256, 170, 170)'))
        # create the unprotected lines
        for unprot_coord in unprot_coords:
            fig.add_shape(type="line",
                          x0=pt_coords[0], y0=pt_coords[1],
                          x1=unprot_coord[0], y1=unprot_coord[1],
                          line=dict(width=1, color='rgb(170, 256, 170)'))

        return fig
