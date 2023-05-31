import plotly.graph_objects as go
import seaborn as sns


def scatter_plot(data, hover_data=None, text=None, size=None,
                 text_position=None, symbol=None, symbol_map=None, color=None,
                 width=0, name=None):
    # create a scatter plot
    if hover_data is not None:
        hover_text = list()
        for row in data.iterrows():
            new_text = ''
            for col in hover_data:
                if col in ['x', 'y', 'type']:
                    continue
                new_text += f'{col}: {row[1][col]}<br>'
            hover_text.append(new_text)
        hover_data = hover_text

    if symbol is not None:
        symbol = [symbol_map[k] for k in symbol]

    fig = go.Figure()
    new_scatter = go.Scatter(x=data['x'], y=data['y'], mode='markers+text',
                             hoverinfo='text', hovertext=hover_data,
                             text=text, textposition=text_position,
                             name=name, opacity=1, legendgroup=name,
                             marker=dict(color=color, symbol=symbol, size=size,
                                         line=dict(width=width, color=color)))
    # if the symbol doesn't have a fill then add a thicker background with
    # another color
    if symbol is not None and symbol[0] in ['cross-thin', 'line-ew']:
        bg_scatter = go.Scatter(x=data['x'], y=data['y'], mode='markers+text',
                                hoverinfo='text', hovertext=hover_data,
                                text=text, textposition=text_position,
                                name=name, opacity=1,
                                marker=dict(symbol=symbol, size=size,
                                            line=dict(width=2 * width, color='white')),
                                showlegend=False, legendgroup=name)
        fig.add_trace(bg_scatter)

    fig.add_trace(new_scatter)

    return fig


def kde_segment(data, segment_dict, color, segment_name, weights=None):
    # bound the data points using the segment dictionary
    for col, vals in segment_dict.items():
        if type(vals) == list or type(vals) == tuple:
            lo_bound, hi_bound = vals
            data = data[(lo_bound <= data[col]) & (data[col] <= hi_bound)]
        else:
            data = data[data[col] == vals]
    if weights is not None:
        idx_val = data.index.values
        weights = weights[idx_val]

    # create the seaborn's kde plot to create the plotly figure from
    og_fig = sns.kdeplot(data=data, x='x', y='y', weights=weights, levels=2,
                         thresh=0.01, bw_method=0.1)

    # create the plotly figure
    figure = go.Figure()
    # go over all paths and combine them into on
    for i, path in enumerate(og_fig.collections[-2].get_paths()):
        vertices = path.vertices  # the vertices of the path
        figure = figure.add_trace(
            go.Scatter(x=vertices[:, 0], y=vertices[:, 1],
                       fill="toself", fillcolor=color,
                       line=dict(width=0), opacity=0.2,
                       showlegend=not i, name=segment_name))

    figure.update_traces(legendgroup=segment_name)
    return figure


def combine_plots(plot1, plot2):
    # combine two plots
    combined = go.Figure(data=plot1.data + plot2.data)
    return combined


def set_theme(plot, template):
    # set the theme of a plot
    plot.update_layout(template=template)
    return plot


if __name__ == '__main__':
    # fig = px.parallel_coordinates(og_df, color="Class",
    #                               color_continuous_midpoint=2)
    # fig.show()
    pass
