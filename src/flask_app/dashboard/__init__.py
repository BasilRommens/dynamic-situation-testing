import dash
import dash_dangerously_set_inner_html
import plotly.graph_objects as go
from dash import html, dcc, Output, Input, no_update, State, dash_table
import dash_bootstrap_components as dbc

from flask_app.dashboard.layout import html_layout
from setup import data

click_shapes = list()


def serve_layout():
    # if there is the plot key, and it is set to false then no plot is drawn
    # add a link to situation-testing-form page
    if 'plot' in data and not data['plot']:
        return html.Div([
            dbc.Container([
                dbc.Row([
                    dbc.Col(
                        html.H3(data['plot_name']),
                    )
                ], class_name='mt-3'),
                dbc.Row([
                    dbc.Col(
                        html.P('There is nothing to plot')
                    )
                ], class_name='mt-3'),
                dbc.Row([
                    dbc.Col(
                        html.A('Go to situation testing form',
                               href='/situation-testing-form',
                               className='btn btn-primary mt-3'),
                        width=4
                    )
                ], class_name='my-3'),
            ]),
            dcc.Store(id='graph_store_layout'),
            dcc.Store(id='graph_store_style'),
        ])

    html_points = '<i class="fas fa-circle" style="color: #e69f00;"></i>, <i class="fas fa-circle" style="color: #56b4e9;"></i>, <i class="fas fa-plus" style="color: #56b4e9;"></i> or <i class="fas fa-minus" style="color: #56b4e9;"></i>'
    plus = '<i class="fas fa-plus" style="color: #56b4e9;"></i>'
    minus = '<i class="fas fa-minus" style="color: #56b4e9;"></i>'
    red_square = '<i class="fas fa-square" style="color: #d55e00;"></i>'
    green_square = '<i class="fas fa-square" style="color: #009e73;"></i>'
    graph = [
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle(
                    dash_dangerously_set_inner_html.DangerouslySetInnerHTML(
                        '<i class="fas fa-info-circle"></i> Info About the Plot'
                    ))
                ),
                dbc.ModalBody(
                    [
                        dash_dangerously_set_inner_html.DangerouslySetInnerHTML(
                            f'<b>Hover</b> over a point ({html_points}) to see its details and its linked neighbors'),
                        dash_dangerously_set_inner_html.DangerouslySetInnerHTML(
                            f'<b>Click</b> on a point ({html_points}) to lock the {red_square} <b>red</b> and {green_square} <b>green</b> links to the neighbors'),
                        dash_dangerously_set_inner_html.DangerouslySetInnerHTML(
                            f'{minus} Negative discrimination means that you have a positive score for a negative class value.'
                            f' {plus} Positive discrimination means that your neighborhood of protected samples were mostly positevily evaluated.'
                            f' If your a sample is close to a <i class="fas fa-circle"></i> variable point then it probably has a high value for that variable.'
                        )
                    ]
                ),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close", id="close", className="ms-auto", n_clicks=0
                    )
                ),
            ],
            id="modal",
            is_open=False,
        ),
        dash_dangerously_set_inner_html.DangerouslySetInnerHTML(
            '<h4>Discrimination Plot'
        ),
        dbc.Row([
            dbc.Col(
                dbc.Button(
                    dash_dangerously_set_inner_html.DangerouslySetInnerHTML(
                        '<i class="fas fa-info-circle"></i> More Info'
                    ), id="open", color="dark", n_clicks=0),
            ),
            dbc.Col(
                dash_dangerously_set_inner_html.DangerouslySetInnerHTML(
                    f'''
                            <p>
                                {red_square} <b>Red</b> = Protected,
                                {green_square} <b>Green</b> = Not Protected
                            </p>
                        '''
                ), className="text-end align-middle",
            ),
        ]),
        dcc.Graph(
            id="graph",
            figure=data['fig'],
            clear_on_unhover=True,
            style={'display': 'block'}
        ),
    ]

    contour_form = [
        html.H4("Add Contours", className='mt-2'),
        html.P(
            "Contours are used to group tuples with similar values for a variable.",
            className='mb-2'),
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

    decision_attr_var = ''
    decision_attr_val = ''
    sensitive_attr_dict = dict()
    red_circle = '<i class="fas fa-circle" style="color: #d55e00;"></i>'
    green_circle = '<i class="fas fa-circle" style="color: #009e73;"></i>'
    table = [
        html.H4("Table of Discriminated Tuples", className='mt-3'),
        html.Div(
            dash_dangerously_set_inner_html.DangerouslySetInnerHTML(
                "The table below shows the tuples that have all the protected values "
                "and have the decision value. When clicking on a value in a row "
                f"the corresponding tuple will be highlighted in the plot with a {red_circle} red circle."
            ),
            className='mb-2',
        ),
        dash_table.DataTable(data['table'],
                             [{"name": key, "id": key}
                              for key in
                              data['table'][0].keys()],
                             id='tbl')
        if len(data['table']) > 0 else
        dash_table.DataTable(id='tbl'),
    ]

    neighbor_table = [html.H4("Table of Neighboring Tuples", className='mt-3')]
    if 'neighbor_tbl' in data:
        neighbor_table.append(
            html.Div(
                dash_dangerously_set_inner_html.DangerouslySetInnerHTML(
                    f"When clicking on a value in a row, the corresponding tuple will be highlighted in the plot with a {green_circle} green circle.",
                ),
                id='neighbor_txt',
                className='mb-2',
                style={'display': 'block'}
            )
        )
        neighbor_table.append(
            html.Div(
                children=dash_table.DataTable(data['neighbor_tbl'],
                                              data['neighbor_cols'],
                                              id='neighbor_tbl'),
                id='neighbor_div',
            )
        )
    else:
        neighbor_table.append(
            html.Div(
                dash_dangerously_set_inner_html.DangerouslySetInnerHTML(
                    f"When clicking on a value in a row, the corresponding tuple will be highlighted in the plot with a {green_circle} green circle.",
                ),
                id='neighbor_txt',
                style={'display': 'none'}
            )
        )
        # TODO clean this up by adding grey text
        neighbor_table.append(
            html.Div(
                'No neighboring tuples. Select a tuple before seeing a table of neighboring tuples.',
                id='neighbor_div',
                className='text-muted',
            )
        )

    return html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col(
                    html.H3(data['plot_name'], className='mt-3'),
                )
            ]),
            dbc.Row([
                dbc.Col([
                    html.H4("Situation Testing Variables & Values",
                            className='mt-3'),
                    data['situation_testing_html']
                ])
            ]),
            dbc.Row([
                dbc.Col(graph, width=8, className='mr-4'),
                dbc.Col([
                    dbc.Row(dbc.Col(contour_form)),
                    dbc.Row(dbc.Col()),
                    dbc.Row(dbc.Col(contour_rm_btns),
                            class_name="position-absolute bottom-0 start-0")
                ], width=4, class_name="position-relative",
                    style={'background-color': '#f8f9fa',
                           'border-radius': '5px'})
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
        routes_pathname_prefix='/plot/',
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

    # table of neigboring tuples
    @dash_app.callback(
        Output('graph', 'figure', allow_duplicate=True),
        Input('neighbor_tbl', 'active_cell'),
        prevent_initial_call=True
    )
    def click_neighbor_table(active_cell):
        global data

        if active_cell is None:
            return no_update

        # get the clicked point in the graph
        valid_tuple_click_pt = [(prot_tuples, unprot_tuples) for pt_idx, _, prot_tuples, unprot_tuples in data['valid_tuples'] if pt_idx == data['click_pt_idx']][0]

        # get k to determine if the tuple is protected or unprotected
        protected_len = len(valid_tuple_click_pt[0])

        # remove 1 from the row as the first row is the clicked point
        row = active_cell['row']

        # if the row is greater than the protected length, then it is an
        # unprotected tuple.
        if row >= protected_len:
            # Subtract the protected length to get the unprotected point
            row -= protected_len
            data_pts_row = valid_tuple_click_pt[1][row][0]  # unprotected point index
        else:
            data_pts_row = valid_tuple_click_pt[0][row][0]  # protected point index

        # get the x, y coordinates
        x = data['data_pts']['x'][data_pts_row]
        y = data['data_pts']['y'][data_pts_row]

        # update the figure with an added marker at the selected point
        fig = data['fig']

        # remove all the traces that contain marked as legend group
        fig['data'] = [trace for trace in fig['data']
                       if trace['legendgroup'] != 'marked_neighboring']

        # new trace
        new_fig = go.Figure(
            go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                marker=dict(
                    size=25,
                    color='#009E73',
                    symbol='circle',
                    line=dict(width=0)
                ),
                showlegend=False,
                legendgroup='marked_neighboring'
            )
        )

        # add a marked trace
        fig = go.Figure(data=new_fig['data'] + fig['data'],
                        layout=fig['layout'])

        data['fig'] = fig

        # maybe update more like the margins and the ticks, and the theme
        return fig

    # table of discriminated tuples
    @dash_app.callback(
        Output('graph', 'figure', allow_duplicate=True),
        Input('tbl', 'active_cell'),
        prevent_initial_call=True
    )
    def click_discriminated_table(active_cell):
        global data

        row = active_cell['row']
        # convert the row to a row index in the data pts to find the location
        # x, y coordinate
        data_pts_row = data['valid_tuples'][row][0]
        # get the x, y coordinates
        x = data['data_pts']['x'][data_pts_row]
        y = data['data_pts']['y'][data_pts_row]

        # update the figure with an added marker at the selected point
        fig = data['fig']

        # remove all the traces that contain marked as legend group
        fig['data'] = [trace for trace in fig['data']
                       if trace['legendgroup'] != 'marked']

        # new trace
        new_fig = go.Figure(
            go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                marker=dict(
                    size=25,
                    color='#D55E00',
                    symbol='circle',
                    line=dict(width=0)
                ),
                showlegend=False,
                legendgroup='marked'
            )
        )

        # add a marked trace
        fig = go.Figure(data=new_fig['data'] + fig['data'],
                        layout=fig['layout'])

        data['fig'] = fig

        # maybe update more like the margins and the ticks, and the theme
        return fig

    # MODAL
    @dash_app.callback(
        Output("modal", "is_open"),
        [Input("open", "n_clicks"), Input("close", "n_clicks")],
        [State("modal", "is_open")],
    )
    def toggle_modal(n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open

    # CONTOUR FORMS
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

    # UPDATE OF STORED DATA
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

    # CLICK ACTION
    @dash_app.callback(
        [Output("graph", "figure", allow_duplicate=True),
         Output("neighbor_div", "children"),
         Output("neighbor_div", "className"),
         Output("neighbor_txt", "style"),
         ],
        [Input("graph", "clickData")],
        prevent_initial_call=True,
    )
    def update_click(clickData):
        global data

        if clickData is None:
            return no_update, no_update, no_update, no_update

        pt = clickData["points"][0]
        if 'hovertext' not in pt.keys():
            return no_update, no_update, no_update, no_update

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
        for prot_pt_idx in prot_pts:
            prot_pt = data['data_pts'].iloc[prot_pt_idx].to_dict()
            del prot_pt['x']
            del prot_pt['y']
            del prot_pt['type']
            prot_pt['is_protected'] = 'protected'
            prot_pt['distance'] = f'{data["dist_mat"][pt_idx, prot_pt_idx]:.1f}'
            neighbors.append(prot_pt)

        # information of all the unprotected neighbors
        for unprot_pt_idx in unprot_pts:
            unprot_pt = data['data_pts'].iloc[unprot_pt_idx].to_dict()
            del unprot_pt['x']
            del unprot_pt['y']
            del unprot_pt['type']
            unprot_pt['is_protected'] = 'unprotected'
            unprot_pt[
                'distance'] = f'{data["dist_mat"][pt_idx, unprot_pt_idx]:.1f}'
            neighbors.append(unprot_pt)

        # add the current data point also to the neighbors
        data_pt = data['data_pts'].iloc[pt_idx].to_dict()
        del data_pt['x']
        del data_pt['y']
        del data_pt['type']
        data_pt['is_protected'] = 'selected point'
        data_pt['distance'] = None
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
            return no_update, no_update, no_update, no_update

        data['click_pt_idx'] = pt_idx

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
                               line=dict(width=1.5,
                                         color='#D55E00'))  # color is red

        # create the unprotected lines
        for unprot_coord in unprot_coords:
            temp_fig.add_shape(type="line",
                               x0=pt_coords[0], y0=pt_coords[1],
                               x1=unprot_coord[0], y1=unprot_coord[1],
                               line=dict(width=1.5,
                                         color='#009E73'))  # color is green

        # add the lines to the figure
        data['fig']['layout']['shapes'] = temp_fig['layout']['shapes']

        # add the shapes to the click shapes
        data['click_shapes'] = temp_fig['layout']['shapes']

        return data['fig'], neighbor_tbl, "mb-3", {'display': 'block'}

    # HOVER ACTION
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
                                  line=dict(width=1.5,
                                            color='#FF6F01'))
        # create the unprotected lines
        for unprot_coord in unprot_coords:
            data['fig'].add_shape(type="line",
                                  x0=pt_coords[0], y0=pt_coords[1],
                                  x1=unprot_coord[0], y1=unprot_coord[1],
                                  line=dict(width=1.5,
                                            color='#04FFBC'))

        return data['fig']
