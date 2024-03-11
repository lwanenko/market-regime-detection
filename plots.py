
from plotly.subplots import make_subplots
import plotly.graph_objs as go


def plot_results(test_df, true_states, predicted_states, state_probabilities, state_names, png_path=None):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02, subplot_titles=('1. Close Price with True State Markers',
                                                              '2. Close Price with Predicted State Markers',
                                                              '3. State Probabilities Over Time'))

    colors = ['red', 'green', 'blue', 'orange', 'purple']

    for state, name in enumerate(state_names):
        fig.add_trace(go.Scatter(
            x=test_df.index[true_states == state],
            y=test_df['close'][true_states == state],
            mode='markers',
            name=f'True State: {name}',
            marker=dict(color=colors[state % len(colors)], symbol=state)
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=test_df.index,
        y=test_df['close'],
        mode='lines',
        name='Close Price',
        line=dict(color='grey', width=1),
        showlegend=False
    ), row=1, col=1)

    for state, name in enumerate(state_names):
        fig.add_trace(go.Scatter(
            x=test_df.index[predicted_states == state],
            y=test_df['close'][predicted_states == state],
            mode='markers',
            name=f'Predicted State: {name}',
            marker=dict(color=colors[state % len(colors)], symbol=state),
            showlegend=False
        ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=test_df.index,
        y=test_df['close'],
        mode='lines',
        name='Close Price',
        line=dict(color='grey', width=1),
        showlegend=False
    ), row=2, col=1)

    for state, name in enumerate(state_names):
        fig.add_trace(go.Scatter(
            x=test_df.index,
            y=state_probabilities[:, state],
            mode='lines',
            name=f'State {name} Probability',
            line=dict(color=colors[state % len(colors)], width=2),
            showlegend=False
        ), row=3, col=1)

    fig.update_layout(height=1200, title_text='HMM States and Probabilities')
    fig.update_yaxes(title_text="Close Price", row=1, col=1)
    fig.update_yaxes(title_text="Close Price", row=2, col=1)
    fig.update_yaxes(title_text="Probability", row=3, col=1)
    
    if png_path is None:
        fig.show()
    else:
        fig.to_html(png_path)
