import plotly.graph_objects as go
import numpy as np
import pandas as pd

def plot_function(data_country, data_forecast_country, user_input):

    first_date = data_country['year'].min()

    last_date = pd.to_datetime(data_country['year'].max()) + pd.offsets.MonthBegin(1)

    country_name = data_country["country"].unique()[0]

    t = pd.date_range(last_date, periods = user_input, freq='MS')
    forecast_date = t.max()
    out_forecast_df = pd.DataFrame([[x, y] for x, y in zip(t, data_forecast_country)], columns=["year", "Forecast"])

    out_forecast_df['Forecast'] = out_forecast_df['Forecast'].apply(lambda x: np.ravel(x)[0])

    config = {'displayModeBar': False}
    fig = go.Figure()
    df_short = data_country.loc[data_country["year"] >= "2015-01-01"]

    fig.add_trace(go.Scatter(x = df_short["year"],
                             y = df_short['ccpi'],
                             line_color = "blue",
                             name = "Core CPI",
                             mode = "lines"))

    fig.add_trace(go.Scatter(x = out_forecast_df["year"],
                             y = out_forecast_df['Forecast'],
                             name = "Forecasted Core CPI",
                             line_dash = "dash",
                             line_color = "black",
                             mode = "lines"))

    fig.add_vrect(x0 = last_date,
                  x1 = forecast_date,
                  fillcolor = "grey",
                  opacity = 0.25,
                  line_width = 0)

    fig.update_layout(
        title = f"Out-of-sample forecast until {forecast_date.strftime('%B %Y')} for {country_name}",
        xaxis_title = "",
        yaxis_title = "Monthly y-o-y percentage change",
        autosize=False,
        hoverlabel_namelength=-1,
        width=890,
        height=600,
      legend = dict(
            xanchor = "center",
            yanchor = "top",
            y = -0.2,
            x = 0.5,
            orientation = 'h'
      ),
      xaxis=dict(
          dtick='M12',
          tickangle=45,
          tickfont=dict(size=14)
        ),
      yaxis=dict(
          tickfont=dict(size=14)
        )
    )

    fig.update_layout(
    plot_bgcolor='white')

    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey')

    #fig.show(config=config)
    return fig
