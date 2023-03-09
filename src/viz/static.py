if __name__ == '__main__':
    # TODO: fix that this works
    # # seaborn version of scatter plot
    # sns.scatterplot(x=dim_red_samples[:n_d_pts, 0],
    #                 y=dim_red_samples[:n_d_pts, 1],
    #                 hue=og_df['origin'].values,
    #                 legend=False, linewidth=1.5)
    # sns.scatterplot(x=dim_red_samples[n_d_pts:, 0],
    #                 y=dim_red_samples[n_d_pts:, 1],
    #                 hue=5, legend=False)

    # find the protected samples
    # sns.scatterplot(x=dim_red_samples[:base, 0], y=dim_red_samples[:base, 1],
    #                 hue=class_col, style=all_tuple_markers, legend=False,
    #                 markers={'negative discrimination': '_', 'neutral': '4',
    #                          'positive discrimination': '+', 'sensitive': '4'},
    #                 linewidth=1.5, zorder=1)
    # non_neutral_pts = dim_red_samples[sensitive_tuple_idxs]
    # non_neutral_tuple_markers = np.array(all_tuple_markers)[sensitive_tuple_idxs]
    # non_neutral_class_col = np.array(class_col)[sensitive_tuple_idxs]
    #
    # marker = matplotlib.markers.MarkerStyle('o', fillstyle='full')
    # sns.scatterplot(x=non_neutral_pts[:, 0], y=non_neutral_pts[:, 1],
    #                 color='black', style=non_neutral_tuple_markers,
    #                 legend=False,
    #                 markers={'negative discrimination': marker,
    #                          'positive discrimination': marker,
    #                          'neutral': marker,
    #                          'sensitive': marker},
    #                 zorder=0)
    # sns.scatterplot(x=dim_red_samples[base:, 0], y=dim_red_samples[base:, 1],
    #                 color='green', legend=False, zorder=-1)

    # # add feature names
    # for col_idx, col in enumerate(df.columns):
    #     col_base = n_d_pts + col_idx
    #     pt = dim_red_samples[col_base]
    #     x = pt[0] + 0.03
    #     y = pt[1] + 0.03
    #     plt.text(x=x, y=y, s=col, fontdict=dict(color='black', size=10),
    #              bbox=dict(color='white', alpha=0.3))
    #
    # plt.show()
    pass
