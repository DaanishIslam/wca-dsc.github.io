from networkx.algorithms import community
from networkx.utils import groups
import networkx as nx

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.figure_factory as ff


# Define the folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
save_url = os.path.join(BASE_DIR, 'static', 'Outputs')


"""def daily_msg_plot_1(daily_msg_count_filtered):
    # Plotting the data after filtering
    plt.figure(figsize=(18, 10))
    plt.plot(daily_msg_count_filtered.index,
             daily_msg_count_filtered.values, label='Daily Messages')

    # Set x-axis major and minor ticks
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    # plt.gca().xaxis.set_minor_locator(mdates.WeekdayLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))

    # Set y-axis ticks
    plt.yticks(range(0, int(daily_msg_count_filtered.max()) + 500, 100))

    # Annotate the top 10 peaks
    top_10_peaks = daily_msg_count_filtered.nlargest(5)

    # Define the initial offset and decrement
    initial_offset = 100
    decrement = 10
    for i, (peak_date, peak_value) in enumerate(top_10_peaks.items()):
        offset = initial_offset - (decrement * i)
        if offset < 0:
            offset = -offset
        plt.annotate(f'{str(peak_date), peak_value}', xy=(peak_date, peak_value), xytext=(peak_date, peak_value + offset),
                     arrowprops=dict(facecolor='red', shrink=0.01, alpha=0.5),
                     horizontalalignment='left', alpha=0.9)

    # Add labels and title
    plt.xlabel('Date')
    plt.xticks(rotation=90, ha='left')
    plt.ylabel('Message Count')
    plt.title('Daily Message Count Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(save_url)
    plt.close()
"""


def daily_msg_plot_1(daily_msg_count_filtered):

    # Convert the index to a list for Plotly
    x_values = daily_msg_count_filtered.index.tolist()
    y_values = daily_msg_count_filtered.values.tolist()

    # Create the plot
    fig = go.Figure()

    # Add the line trace
    fig.add_trace(go.Scatter(x=x_values, y=y_values,
                  mode='lines', name='Daily Messages'))

    # Define the top 10 peaks
    top_10_peaks = daily_msg_count_filtered.nlargest(5)

    # Define the initial offset and decrement
    initial_offset = 50
    decrement = 5
    for i, (peak_date, peak_value) in enumerate(top_10_peaks.items()):
        offset = initial_offset - (decrement * i)
        if offset < 0:
            offset = -offset
        fig.add_annotation(
            x=peak_date,
            y=peak_value,
            text=f'{peak_date.strftime("%Y-%m-%d")}, {peak_value}',
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-offset,
            arrowcolor='red',
            font=dict(color="black"),
            align='left'
        )
    # Generate monthly tick values from the start to the end of the data
    start_date = x_values[0].replace(day=1)
    end_date = x_values[-1]
    step = 200
    if max(y_values)//step > 15:
        step = 500
    monthly_ticks = pd.date_range(start=start_date, end=end_date, freq='MS')
    y_ticks = list(range(0, max((y_values)) + 100, int(step)))

    # Update x and y axes
    fig.update_xaxes(
        tickvals=monthly_ticks,
        tickformat='%b\n%Y',
        tickmode='array',
        tickangle=-90,
        tickfont=dict(size=10),
    )

    fig.update_yaxes(
        tickvals=y_ticks,
        tickmode='array',
        tickfont=dict(size=10),
    )

    # Update layout to set fixed container size and make it scrollable
    fig.update_layout(
        autosize=False,
        width=1300,  # Set your desired width
        height=500,  # Set your desired height
        margin=dict(l=10, r=10, t=50, b=10),
        # Allow scrolling horizontally
        xaxis=dict(fixedrange=False),
        # Allow scrolling vertically
        yaxis=dict(fixedrange=False),
        title='1. Daily Message Count Over Time - line graph',
        xaxis_title='Date',
        yaxis_title='Message Count'
    )

    # Save the plot as HTML file
    save_plot = save_url + r"\graph_01.html"
    fig.write_html(save_plot)


# Assuming `daily_msg_count_filtered` is your DataFrame with 'Date' as index and 'Message' as values
"""def daily_msg_logT_plot_2(daily_msg_count_filtered):
    # Log transformation
    daily_msg_count_log = np.log1p(daily_msg_count_filtered)  # Use log1p to handle zero values

    plt.figure(figsize=(18, 6))
    plt.plot(daily_msg_count_log.index, daily_msg_count_log.values, label='Daily Messages (Log Transformed)')

    # Set x-axis major and minor ticks
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator())
    plt.gca().xaxis.set_minor_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))

    # Rotate and align x-axis labels for better readability
    plt.xticks(rotation=90, ha='left')

    # Set y-axis ticks
    plt.yticks(range(0, int(daily_msg_count_log.max()) + 1, 1))

    # Annotate the top 10 peaks
    top_10_peaks = daily_msg_count_log.nlargest(10)
    # Define the initial offset and decrement
    initial_offset =5
    decrement = 0.5
    for i, (peak_date, peak_value) in enumerate(top_10_peaks.items()):
        offset = initial_offset - (decrement * i)
        if offset < 0:
            offset = -offset
        plt.annotate(f'{str(peak_date)}: {np.expm1(peak_value):.0f}', xy=(peak_date, peak_value), xytext=(peak_date, peak_value + offset),
                    arrowprops=dict(facecolor='red', shrink=0.01, alpha=0.5),
                    horizontalalignment='left', alpha=0.9)

    # Add labels and title
    plt.xlabel('Date')
    plt.ylabel('Log Transformed Message Count')
    plt.title('Daily Message Count Over Time (Log Transformed)')
    plt.legend()
    plt.grid(True)

    plt.show()
    plt.savefig(save_url)
    plt.close()"""


def daily_msg_logT_plot_2(daily_msg_count_filtered):
    # Log transformation
    # Use log1p to handle zero values
    daily_msg_count_log = np.log1p(daily_msg_count_filtered)

    # Convert the index to a list for Plotly
    x_values = daily_msg_count_log.index.tolist()
    y_values = daily_msg_count_log.values.tolist()

    # Create the plot
    fig = go.Figure()

    # Add the line trace
    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines',
                  name='Daily Messages (Log Transformed)'))

    # Define the top 10 peaks
    top_10_peaks = daily_msg_count_log.nlargest(5)

    # Define the initial offset and decrement
    initial_offset = 50
    decrement = 10
    for i, (peak_date, peak_value) in enumerate(top_10_peaks.items()):
        offset = initial_offset - (decrement * (i//2))
        if offset < 0:
            offset = -offset
        fig.add_annotation(
            x=peak_date,
            y=peak_value,
            # peak_date.strftime("%Y-%m-%d")},
            text=f'{np.expm1(peak_value):.0f}',
            showarrow=True,
            arrowhead=2,
            arrowwidth=1,
            ax=0,
            ay=-offset,
            arrowcolor='red',
            font=dict(color="black"),
            align='left'
        )

    # Generate monthly tick values from the start to the end of the data
    start_date = x_values[0].replace(day=1)
    end_date = x_values[-1]
    monthly_ticks = pd.date_range(start=start_date, end=end_date, freq='MS')

    # Create custom y-axis ticks
    y_ticks = list(range(0, int(daily_msg_count_log.max()) + 1, 1))

    # Update x and y axes
    fig.update_xaxes(
        tickvals=monthly_ticks,
        tickformat='%b\n%Y',
        tickmode='array',
        tickangle=-90,
        tickfont=dict(size=10),
    )

    fig.update_yaxes(
        tickvals=y_ticks,
        tickmode='array',
        tickfont=dict(size=10),
    )

    # Update layout to set fixed container size and make it scrollable
    fig.update_layout(
        autosize=False,
        width=1300,  # Set your desired width
        height=500,  # Set your desired height
        margin=dict(l=10, r=10, t=50, b=10),
        # Allow scrolling horizontally
        xaxis=dict(fixedrange=False),
        # Allow scrolling vertically
        yaxis=dict(fixedrange=False),
        title='2. Daily Message Count Over Time - Line graph (Log Transformed)',
        xaxis_title='Date',
        yaxis_title='Log Transformed Message Count'
    )

    # Save the plot as HTML file
    save_plot = save_url + r"\graph_02.html"
    fig.write_html(save_plot)


"""def daily_msg_hist_plot_3(total_msg_per_day):
    plt.figure(figsize=(12, 6))
    plt.hist(total_msg_per_day.values, bins=100, edgecolor='k', alpha=0.7)

    # Set x-axis labels
    plt.xlabel('Daily Message Count')
    plt.xticks(np.arange(0, total_msg_per_day.max() + 1, step=150))  # Adjust step as needed

    # Set y-axis labels
    plt.ylabel('Frequency')
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure y-axis labels are integers
    plt.yticks(np.arange(0, plt.gca().get_ylim()[1] + 1, step=10))  # Adjust step as needed

    # Add grid for better readability
    plt.grid(True)


    plt.xlabel('Daily Message Count')
    plt.ylabel('Frequency')
    plt.title('Distribution of Daily Message Counts')
    plt.grid(True)
    plt.show()
    plt.savefig(save_url)
    plt.close()"""


def daily_msg_hist_plot_3(total_msg_per_day):
    # Create histogram data
    hist_data = total_msg_per_day.values

    # Define the bins
    bins = np.histogram(hist_data, bins=100)[1]

    # Create the plot
    fig = go.Figure()

    # Add the histogram trace
    fig.add_trace(go.Histogram(
        x=hist_data,
        xbins=dict(
            start=bins[0],
            end=bins[-1],
            size=(bins[-1] - bins[0]) / 100  # Adjust the bin size
        ),
        autobinx=False,
        marker=dict(color='blue', line=dict(color='black', width=1)),
        opacity=0.7,
        name='Frequency'
    ))

    # Update layout to set fixed container size and make it scrollable
    fig.update_layout(
        autosize=False,
        width=1300,  # Set your desired width
        height=500,  # Set your desired height
        margin=dict(l=20, r=20, t=50, b=50),
        title='3. Distribution of Daily Message Counts - Histogram Plot',
        xaxis=dict(
            title='Daily Message Count',
            tickvals=np.arange(0, total_msg_per_day.max() + 1,
                               step=100),  # Adjust step as needed
            fixedrange=False,  # Allow scrolling horizontally
            range=[-5, bins[-1] + 5]
        ),
        yaxis=dict(
            title='Frequency',
            tickmode='array',
            tickvals=np.arange(0, np.histogram(hist_data, bins=100)[
                               0].max() + 1, step=50),  # Adjust step as needed
            fixedrange=False  # Allow scrolling vertically
        )
    )

    # Save the plot as HTML file
    save_plot = save_url + r"\graph_03.html"
    fig.write_html(save_plot)


# Assuming daily_msg_count_filtered is your pandas Series with dates as index and message counts as values
"""def daily_msg_box_plot_4(daily_msg_count_filtered):

    # Set up the plot
    plt.figure(figsize=(12, 5))

    # Create the box plot with additional details
    boxprops = dict(facecolor='yellow', color='blue')
    medianprops = dict(color='red', linewidth=2)
    meanprops = dict(marker='o', markerfacecolor='green', markeredgecolor='black')

    boxplot = plt.boxplot(daily_msg_count_filtered.values, vert=False, patch_artist=True,
                        boxprops=boxprops, medianprops=medianprops, meanline=True,
                        showmeans=True, meanprops=meanprops, flierprops=dict(marker='o', color='orange', markersize=5))

    # Set x-axis labels
    plt.xlabel('Daily Message Count')

    # Add grid for better readability
    plt.grid(True)

    # Add title
    plt.title('Box Plot of Daily Message Counts')

    # Annotate outliers with adjusted positions
    annotated_values = set()
    for i, flier in enumerate(boxplot['fliers']):
        for j, outlier in enumerate(flier.get_xdata()):
            annotate_outlier = True
            for value in annotated_values:
                if abs(outlier - value) < 120:
                    annotate_outlier = False
                    break
            if annotate_outlier:
                annotated_values.add(outlier)
                if i % 2 == 0:
                    plt.annotate(f'{outlier}', xy=(outlier, 1), xytext=(outlier, 1.1),
                                arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=5),
                                horizontalalignment='center')
                else:
                    plt.annotate(f'{outlier}', xy=(outlier, 1), xytext=(outlier, 0.9),
                                arrowprops=dict(facecolor='blue', shrink=0.05, width=1, headwidth=5),
                                horizontalalignment='center')

    # Annotate quartiles and median with adjusted positions
    quartiles = np.percentile(daily_msg_count_filtered.values, [25, 50, 75])
    for i, q in enumerate(quartiles):

        if i == 0:
            offset = 0.10
        elif i == 1:
            offset = -0.15
        elif i == 2:
            offset = -0.18
        else:
            offset = 0.10

        plt.annotate(f'Q{i+1} = {q}', xy=(q, 1), xytext=(q, 1 + offset),
                    horizontalalignment='center', color='blue')

    plt.show()
    plt.savefig(save_url)
    plt.close()"""


def daily_msg_box_plot_4(daily_msg_count_filtered):
    # Create the box plot
    fig = go.Figure()

    # Calculate quartiles and identify outliers
    q1 = np.percentile(daily_msg_count_filtered.values, 25)
    q3 = np.percentile(daily_msg_count_filtered.values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = daily_msg_count_filtered[(daily_msg_count_filtered.values < lower_bound) |
                                        (daily_msg_count_filtered.values > upper_bound)]

    # Add the box trace
    fig.add_trace(go.Box(
        x=daily_msg_count_filtered.values,
        boxpoints='outliers',  # Show only outliers
        jitter=0.5,  # Spread out the points for visibility
        pointpos=0,  # Position of points (aligned with the box plot)
        fillcolor='yellow',
        marker=dict(color='orange', size=5),
        line=dict(color='blue'),
        name='Daily Messages'
    ))

    # Annotate quartiles and median
    quartiles = np.percentile(daily_msg_count_filtered.values, [25, 50, 75])
    plot_height = 500  # This should be the same as the height specified in update_layout

    for i, q in enumerate(quartiles):
        if i == 0:
            offset = 0.35
        elif i == 1:
            offset = 0.30
        elif i == 2:
            offset = 0.25
        else:
            offset = 0.2

        dynamic_ay = -0.1 * plot_height  # Dynamic offset as a fraction of plot height

        fig.add_annotation(
            x=q,
            y=offset,  # Align with the box plot
            text=f'Q{i+1} = {q}',
            showarrow=True,
            arrowhead=3,
            ax=0,
            ay=dynamic_ay,  # Adjusted dynamically
            arrowwidth=1,
            arrowcolor='red',
            font=dict(color="blue"),
            align='center'

        )

    # Annotate top 5 outliers
    top_outliers = outliers.sort_values(ascending=False).head(5)
    inv = 1
    for outlier in top_outliers.values:
        # Dynamic offset as a fraction of plot height
        dynamic_ay = -0.1 * plot_height * inv
        fig.add_annotation(
            x=outlier,
            y=0.01 * inv,  # Align with the outlier point
            text=f'{outlier}',
            showarrow=True,
            arrowhead=3,
            ax=0,
            ay=dynamic_ay,  # Adjusted dynamically
            arrowwidth=1,
            arrowcolor='blue',
            font=dict(color="red"),
            align='center'
        )
        inv *= -1

    # Add title and labels
    fig.update_layout(
        title='4. Daily Message Counts - Box Plot',
        xaxis_title='Daily Message Count',
        yaxis_title='Frequency',
        xaxis=dict(
            tickmode='array',
            # Adjust step as needed
            tickvals=np.arange(
                0, daily_msg_count_filtered.max() + 1, step=150),
            # Add space before the first bin and after the last bin
            range=[-5, daily_msg_count_filtered.max() + 5],
            zeroline=True  # Show x-axis line
        ),
        yaxis=dict(
            showticklabels=True,  # Show y-axis labels
            zeroline=True  # Show y-axis line
        ),
        autosize=False,
        width=1300,  # Set your desired width
        height=plot_height,  # Set your desired height
        margin=dict(l=20, r=20, t=50, b=50),
        # plot_bgcolor='rgba(0,0,0,0)',  # Set plot background color to transparent

    )

    # Center the plot
    fig.update_xaxes(title_standoff=20)
    fig.update_yaxes(title_standoff=20)

    # Save the plot as an HTML file
    save_plot = save_url + r"\graph_04.html"
    fig.write_html(save_plot)


"""Plotting Box Plot for Hourly Messages"""

# Plotting for Hourly Messages
""""def hourly_barLine_plot_5(hourly_msg_count):

    plt.figure(figsize=(18, 10))

    # Plotting the line plot
    plt.plot(hourly_msg_count.index, hourly_msg_count.values, label='Hourly Messages', color='blue')

    # Plotting the bar plot
    plt.bar(hourly_msg_count.index, hourly_msg_count.values, color='green', alpha=0.6)

    # Add annotations for all 24-hour points
    for hour, count in zip(hourly_msg_count.index, hourly_msg_count.values):
        plt.annotate(f'{count}', xy=(hour, count), xytext=(hour, count + 150),
                    arrowprops=dict(facecolor='red', shrink=0.01, alpha=0.5),
                    horizontalalignment='center', fontsize=8)

    # Set x-axis ticks to display all hours from 0 to 24
    plt.xticks(range(24))

    # Set y-axis ticks in increments of 100
    plt.yticks(range(0, int(hourly_msg_count.max()) + 100, 100))

    # Add labels and title
    plt.xlabel('Hours (24 hours)')
    plt.ylabel('Message Count')
    plt.title('Hourly Message Count Over Entire Dataset Grouped')
    plt.legend()
    plt.grid(True)

    plt.show()
    plt.savefig(save_url)
    plt.close()"""


def hourly_barLine_plot_5(hourly_msg_count):
    fig = go.Figure()

    # Add the line trace
    fig.add_trace(go.Scatter(
        x=hourly_msg_count.index,
        y=hourly_msg_count.values,
        mode='lines+markers',
        name='Hourly Messages',
        line=dict(color='blue')
    ))

    # Add the bar trace
    fig.add_trace(go.Bar(
        x=hourly_msg_count.index,
        y=hourly_msg_count.values,
        name='Hourly Messages',
        marker=dict(color='green', opacity=0.6)
    ))

    # Add annotations for all 24-hour points
    for hour, count in zip(hourly_msg_count.index, hourly_msg_count.values):
        fig.add_annotation(
            x=hour,
            y=count,
            text=f'{count}',
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-30,
            arrowcolor='red',
            font=dict(size=8),
            align='center'
        )

    # Update layout
    fig.update_layout(
        title='5. Hourly Message Count Over Entire Dataset Grouped - Bar graph with Line plot',
        xaxis_title='Hours (24 hours)',
        yaxis_title='Message Count',
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(24))
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(0, int(hourly_msg_count.max()) + 100, 200))
        ),
        autosize=False,
        width=1300,
        height=500,
        margin=dict(l=10, r=10, t=50, b=50)
    )

    # Save the plot as HTML file
    save_plot = save_url + r"\graph_05.html"
    fig.write_html(save_plot)


"""Plotting Box Plot for Hourly Messages"""

"""def hourly_msg_box_plot_6(hourly_msg_count):
    plt.figure(figsize=(12, 8))
    plt.boxplot(hourly_msg_count.values, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))

    # Add annotations for quartiles
    for quartile, value in zip([25, 50, 75], np.percentile(hourly_msg_count, [25, 50, 75])):
        plt.annotate(f'Q{quartile}: {value:.0f}', xy=(value, 1), xytext=(value, 1.2),
                    arrowprops=dict(facecolor='blue', shrink=0.01, alpha=0.5),
                    horizontalalignment='center', fontsize=8)

    # Add annotations for outliers
    outliers = hourly_msg_count.index[hourly_msg_count.values > np.percentile(hourly_msg_count, 75) + 1.5 * (np.percentile(hourly_msg_count, 75) - np.percentile(hourly_msg_count, 25))]
    for outlier in outliers:
        plt.annotate(f'{hourly_msg_count[outlier]}', xy=(hourly_msg_count[outlier], 1), xytext=(hourly_msg_count[outlier], 1.2),
                    arrowprops=dict(facecolor='red', shrink=0.01, alpha=0.5),
                    horizontalalignment='center', fontsize=8)

    # Add labels and title
    plt.xlabel('Message Count')
    plt.title('Hourly Message Count Box Plot')
    plt.grid(True)

    plt.show()
    plt.savefig(save_url)
    plt.close()
"""


def hourly_msg_box_plot_6(hourly_msg_count):
    fig = go.Figure()

    # Calculate quartiles and identify outliers
    q1 = np.percentile(hourly_msg_count.values, 25)
    q3 = np.percentile(hourly_msg_count.values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = hourly_msg_count[(hourly_msg_count.values < lower_bound) |
                                (hourly_msg_count.values > upper_bound)]

    # Add the box trace
    fig.add_trace(go.Box(
        x=hourly_msg_count.values,
        boxpoints='outliers',  # Show only outliers
        jitter=0.5,  # Spread out the points for visibility
        pointpos=0,  # Position of points
        fillcolor='lightblue',
        marker=dict(color='orange', size=5),
        line=dict(color='blue'),
        name='Hourly Messages'
    ))

    # Annotate quartiles
    quartiles = np.percentile(hourly_msg_count.values, [25, 50, 75])
    plot_height = 500  # This should be the same as the height specified in update_layout

    for i, q in enumerate(quartiles):
        dynamic_ay = -0.1 * plot_height  # Dynamic offset as a fraction of plot height
        fig.add_annotation(
            x=q,
            y=0.3,  # Align with the box plot
            text=f'Q{i+1} = {q:.0f}',
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=dynamic_ay,  # Adjusted dynamically
            arrowcolor='blue',
            # font=dict(size=8),
            align='center'
        )

    # Annotate outliers
    top_outliers = outliers.sort_values(ascending=False).head(5)
    inv = 1
    for outlier in top_outliers.values:
        # Dynamic offset as a fraction of plot height
        dynamic_ay = -0.1 * plot_height * inv
        fig.add_annotation(
            x=outlier,
            y=0.01 * inv,  # Align with the outlier point
            text=f'{outlier}',
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=dynamic_ay,  # Adjusted dynamically
            arrowcolor='red',
            font=dict(size=8),
            align='center'
        )
        inv *= -1

    # Update layout
    fig.update_layout(
        title='6. Hourly Message Count - Box Plot',
        xaxis_title='Message Count',
        yaxis=dict(
            showticklabels=False  # Hide y-axis labels
        ),
        autosize=False,
        width=1300,
        height=plot_height,
        margin=dict(l=20, r=20, t=50, b=50)
    )

    # Save the plot as HTML file
    save_plot = save_url + r"\graph_06.html"
    fig.write_html(save_plot)


"""Plotting for Monthly Messages"""
"""def monthly_barLine_plot_7(monthly_msg_count):
    plt.figure(figsize=(18, 10))

    # Plotting the line plot
    plt.plot(monthly_msg_count.index, monthly_msg_count.values, label='Monthly Messages', color='green')

    # Plotting the bar plot
    plt.bar(monthly_msg_count.index, monthly_msg_count.values, color='pink', alpha=0.7)

    # Add annotations for all 24-hour points
    for hour, count in zip(monthly_msg_count.index, monthly_msg_count.values):
        plt.annotate(f'{count}', xy=(hour, count), xytext=(hour, count + 500),
                    arrowprops=dict(facecolor='red', shrink=0.01, alpha=0.5),
                    horizontalalignment='center', fontsize=8)

    # Set custom month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # Set custom month names for months with available data
    available_months = [month_names[i-1] for i in monthly_msg_count.index]

    # Set x-axis ticks to display custom month names for available months
    plt.xticks(monthly_msg_count.index, available_months)

    # Set y-axis ticks in increments of 100
    plt.yticks(range(0, int(monthly_msg_count.max()) + 1, 1000))

    # Add labels and title
    plt.xlabel('Months')
    plt.ylabel('Message Count')
    plt.title('Monthly Message Count Over Entire Dataset Grouped')
    plt.legend()
    plt.grid(True)

    plt.show()
    plt.savefig(save_url)
    plt.close()"""


def monthly_barLine_plot_7(monthly_msg_count):
    fig = go.Figure()

    # Add the line trace
    fig.add_trace(go.Scatter(
        x=monthly_msg_count.index,
        y=monthly_msg_count.values,
        mode='lines+markers',
        name='Monthly Messages',
        line=dict(color='green')
    ))

    # Add the bar trace
    fig.add_trace(go.Bar(
        x=monthly_msg_count.index,
        y=monthly_msg_count.values,
        name='Monthly Messages',
        marker=dict(color='pink', opacity=0.7)
    ))

    # Add annotations for all months
    for month, count in zip(monthly_msg_count.index, monthly_msg_count.values):
        fig.add_annotation(
            x=month,
            y=count,
            text=f'{count}',
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-30,
            arrowcolor='red',
            font=dict(size=8),
            align='center'
        )

    # Set custom month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May',
                   'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    available_months = [month_names[i-1] for i in monthly_msg_count.index]

    # Update layout
    fig.update_layout(
        title='7. Monthly Message Count Over Entire Dataset Grouped - Bar graph with Line Plot',
        xaxis_title='Months',
        yaxis_title='Message Count',
        xaxis=dict(
            tickmode='array',
            tickvals=monthly_msg_count.index,
            ticktext=available_months
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(0, int(monthly_msg_count.max()) + 1000, 1000))
        ),
        autosize=False,
        width=1300,
        height=500,
        margin=dict(l=10, r=10, t=50, b=50)
    )

    # Save the plot as HTML file
    save_plot = save_url + r"\graph_07.html"
    fig.write_html(save_plot)


"""Plotting Box Plot for Monthly Messages"""
"""def monthly_msg_box_plot_8(monthly_msg_count):

    plt.figure(figsize=(12, 8))
    plt.boxplot(monthly_msg_count.values, vert=False, patch_artist=True, boxprops=dict(facecolor='lightgreen'))

    # Add annotations for quartiles
    for quartile, value in zip([25, 50, 75], np.percentile(monthly_msg_count, [25, 50, 75])):
        plt.annotate(f'Q{quartile}: {value:.0f}', xy=(value, 1), xytext=(value, 1.2),
                    arrowprops=dict(facecolor='blue', shrink=0.01, alpha=0.5),
                    horizontalalignment='center', fontsize=8)

    # Add annotations for outliers
    outliers = monthly_msg_count.index[monthly_msg_count.values > np.percentile(monthly_msg_count, 75) + 1.5 * (np.percentile(monthly_msg_count, 75) - np.percentile(monthly_msg_count, 25))]
    for outlier in outliers:
        plt.annotate(f'{monthly_msg_count[outlier]}', xy=(monthly_msg_count[outlier], 1), xytext=(monthly_msg_count[outlier], 1.2),
                    arrowprops=dict(facecolor='red', shrink=0.01, alpha=0.5),
                    horizontalalignment='center', fontsize=8)

    # Add labels and title
    plt.xlabel('Message Count')
    plt.title('Monthly Message Count Box Plot')
    plt.grid(True)

    plt.show()
    plt.savefig(save_url)
    plt.close()"""


def monthly_msg_box_plot_8(monthly_msg_count):
    fig = go.Figure()

    # Calculate quartiles and identify outliers
    q1 = np.percentile(monthly_msg_count.values, 25)
    q3 = np.percentile(monthly_msg_count.values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = monthly_msg_count[(monthly_msg_count.values < lower_bound) |
                                 (monthly_msg_count.values > upper_bound)]

    # Add the box trace
    fig.add_trace(go.Box(
        x=monthly_msg_count.values,
        boxpoints='outliers',  # Show only outliers
        jitter=0.5,  # Spread out the points for visibility
        pointpos=0,  # Position of points
        fillcolor='lightgreen',
        marker=dict(color='orange', size=5),
        line=dict(color='blue'),
        name='Monthly Messages'
    ))

    # Annotate quartiles
    quartiles = np.percentile(monthly_msg_count.values, [25, 50, 75])
    plot_height = 500  # This should be the same as the height specified in update_layout

    for i, q in enumerate(quartiles):
        dynamic_ay = -0.1 * plot_height  # Dynamic offset as a fraction of plot height
        fig.add_annotation(
            x=q,
            y=0.3,  # Align with the box plot
            text=f'Q{i+1} = {q:.0f}',
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=dynamic_ay,  # Adjusted dynamically
            arrowcolor='blue',
            # font=dict(size=8),
            align='center'
        )

    # Annotate outliers
    top_outliers = outliers.sort_values(ascending=False).head(5)
    inv = 1
    for outlier in top_outliers.values:
        # Dynamic offset as a fraction of plot height
        dynamic_ay = -0.1 * plot_height * inv
        fig.add_annotation(
            x=outlier,
            y=0.01 * inv,  # Align with the outlier point
            text=f'{outlier}',
            showarrow=True,
            arrowhead=2,
            ax=20,
            ay=dynamic_ay,  # Adjusted dynamically
            arrowcolor='red',
            font=dict(size=8),
            align='center'
        )
        inv *= -1

    # Update layout
    fig.update_layout(
        title='8. Monthly Message Count - Box Plot',
        xaxis_title='Message Count',
        yaxis=dict(
            showticklabels=False  # Hide y-axis labels
        ),
        autosize=False,
        width=1300,
        height=plot_height,
        margin=dict(l=20, r=20, t=50, b=50)
    )

    # Save the plot as HTML file
    save_plot = save_url + r"\graph_08.html"
    fig.write_html(save_plot)


"""Bar Graph for most Active Users Sorted"""

"""def most_active_user_plot_9(most_active_users_list):
    active_user = most_active_users_list.sort_values(ascending=False)

    plt.figure(figsize=(20, 10))

    # Plotting the line plot
    plt.plot(active_user.index,active_user.values,  label='Active Users', color='green')

    # Plotting the bar plot
    plt.bar(active_user.index, active_user.values,  color='skyblue', alpha=0.7)

    for name, count in zip(active_user.index, active_user.values):
        plt.annotate(f'{count}', xy=(name, count), xytext=(0, 3), textcoords='offset points',
                    ha='left', rotation=65)

    # Set y-axis ticks in increments of 100
    plt.yticks(range(0, int(active_user.max()) + 100, 100))

    # Add labels and title
    plt.xlabel('Names')
    plt.xticks(rotation=90, ha='left')
    plt.ylabel('Message Count')
    plt.title('Most Active Users in Chats')
    plt.legend()
    plt.grid(True)

    plt.show()
    plt.savefig(save_url)
    plt.close()"""


def most_active_user_plot_9(most_active_users_list):
    active_user = most_active_users_list.sort_values(ascending=False)

    fig = go.Figure()

    # Add the bar trace
    fig.add_trace(go.Bar(
        x=active_user.index,
        y=active_user.values,
        marker_color='skyblue',
        opacity=0.7,
        name='Active Users'
    ))

    # Add annotations for each bar
    for name, count in zip(active_user.index, active_user.values):
        fig.add_annotation(
            x=name,
            y=count+50,
            text=f'{count}',
            showarrow=False,
            font=dict(size=10),
            align='right',
            textangle=-45

        )

    # Update layout
    fig.update_layout(
        title='9. Most Active Users in Chats - Bar Graph',
        xaxis_title='Names',
        yaxis_title='Message Count',
        xaxis=dict(
            tickangle=-90,
            tickfont=dict(size=10),
            tickmode='array',
            tickvals=active_user.index
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(0, int(active_user.max()) + 100, 200))
        ),
        autosize=False,
        width=1300,
        height=500,
        margin=dict(l=20, r=20, t=50, b=50)
    )

    # Save the plot as HTML file
    save_plot = save_url + r"\graph_09.html"
    fig.write_html(save_plot)


"""Bar graph for finding Peak hour in a day of an entire month"""
"""def peak_msg_ToD_plot_10(tdays,thours,tmessage_counts):
    

    # Plotting the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(tdays, tmessage_counts, color='skyblue', label='Message Count')
    plt.xlabel('Day')
    plt.ylabel('Message Count')
    plt.title('Peak Messaging Time per Day')
    plt.xticks(tdays)  # Set x-axis ticks to display all days
    plt.legend()

    # Adding annotations for each bar
    for day, hour, count in zip(tdays, thours, tmessage_counts):
        plt.text(day, count, f'{hour}:00', ha='center', va='bottom', rotation=45)

    plt.grid(True)
    plt.show()
    plt.savefig(save_url)
    plt.close()"""


def peak_msg_ToD_plot_10(tdays, thours, tmessage_counts):
    fig = go.Figure()

    # Add the bar trace
    fig.add_trace(go.Bar(
        x=tdays,
        y=tmessage_counts,
        marker_color='skyblue',
        opacity=0.7,
        name='Message Count'
    ))

    # Add annotations for each bar
    for day, hour, count in zip(tdays, thours, tmessage_counts):
        fig.add_annotation(
            x=day,
            y=count,
            text=f'{hour}:00',
            showarrow=False,
            font=dict(size=10),
            align='center',
            yshift=5
        )

    # Update layout
    fig.update_layout(
        title='10. Peak Messaging Time per Day - Bar graph',
        xaxis_title='Day',
        yaxis_title='Message Count',
        xaxis=dict(
            tickmode='array',
            tickvals=tdays
        ),
        autosize=False,
        width=1300,
        height=500,
        margin=dict(l=20, r=20, t=50, b=50)
    )

    # Save the plot as HTML file
    save_plot = save_url + r"\graph_10.html"
    fig.write_html(save_plot)


"""ScatterPlot for intersection of day-hour for Message Count"""

"""def interDH_scatter_plot_11(tdays,thours,tmessage_counts):
    # Extracting data for plotting

    # Plotting the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(tdays, thours, s=tmessage_counts, c='lightgreen', alpha=0.7, label='Message Count')
    plt.xlabel('Day')
    plt.ylabel('Hour')
    plt.title('Peak Messaging Time per Day in a month')
    plt.xticks(tdays)  # Set x-axis ticks to display all days
    plt.legend()

    # Adding annotations for each point
    for day, hour, count in zip(tdays, thours, tmessage_counts):
        plt.text(day, hour, f'{count}', ha='center', va='bottom')


    # Set y-axis ticks in increments of 100
    plt.yticks(range(0, 24, 1))

    plt.grid(True)
    plt.show()
    plt.savefig(save_url)
    plt.close()"""


def interDH_scatter_plot_11(tdays, thours, tmessage_counts):
    fig = go.Figure()

    # Add the scatter trace
    fig.add_trace(go.Scatter(
        x=tdays,
        y=thours,
        mode='markers',
        marker=dict(
            size=tmessage_counts * 0.1,
            color='lightgreen',
            opacity=0.7,
            line=dict(width=0.5, color='DarkSlateGrey')
        ),
        name='Message Count'

    ))

    # Add annotations for each point
    for day, hour, count in zip(tdays, thours, tmessage_counts):
        fig.add_annotation(
            x=day,
            y=hour,
            text=f'{count}',
            showarrow=False,
            font=dict(size=10),
            align='center'
        )

    # Update layout
    fig.update_layout(
        title='11. Peak Messaging Time per Day in a Month - Scatter Plot',
        xaxis_title='Day',
        yaxis_title='Hour',
        xaxis=dict(
            tickmode='array',
            tickvals=tdays
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(0, 24, 1))
        ),
        autosize=False,
        width=1300,
        height=500,
        margin=dict(l=20, r=20, t=50, b=50)
    )

    # Save the plot as HTML file
    save_plot = save_url + r"\graph_11.html"
    fig.write_html(save_plot)


"""Weekly Message analysis"""
"""def peak_msg_DPweek_plot_12(week_df):

    # Define a color map for the days of the week
    color_map = {
        'Monday': 'red',
        'Tuesday': 'blue',
        'Wednesday': 'green',
        'Thursday': 'purple',
        'Friday': 'orange',
        'Saturday': 'pink',
        'Sunday': 'cyan'
    }

    # Assign colors to each bar based on the day
    bar_colors = [color_map[day] for day in week_df['most_messaged_day']]

    # Plotting the bar plot
    plt.figure(figsize=(14, 8))
    bars = plt.bar(week_df['week_range'], week_df['most_messaged_day_count'], color=bar_colors, alpha=0.6)

    # Add labels and title
    plt.xlabel('Week Range')
    plt.ylabel('Message Count')
    plt.title('Most Messaged Day Count per Week Range')
    plt.grid(True)

    # Annotate the bars with the day names and counts
    for bar, day, count in zip(bars, week_df['most_messaged_day'], week_df['most_messaged_day_count']):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, f'{day}\n{count}', ha='center', rotation=45, fontsize=10, color='black')

    plt.show()
    plt.savefig(save_url)
    plt.close()"""
"""Error in x axis and something else"""


def peak_msg_DPweek_plot_12(week_df):
    try:
        # Extract x_values and y_values from the DataFrame
        # Use range for x-axis values
        x_values = list(range(1, len(week_df) + 1))
        y_values = week_df['most_messaged_day_count'].tolist()
        week_labels = week_df['week_range'].tolist()
        days = week_df['most_messaged_day'].tolist()

        # Define a color map for the days of the week
        color_map = {
            'Monday': 'red',
            'Tuesday': 'blue',
            'Wednesday': 'green',
            'Thursday': 'purple',
            'Friday': 'orange',
            'Saturday': 'pink',
            'Sunday': 'cyan'
        }

        # Assign colors to each bar based on the day
        bar_colors = [color_map[day]
                      for day in week_df['most_messaged_day'].tolist()]

        fig = go.Figure()

        # Add the bar trace
        fig.add_trace(go.Bar(
            x=x_values,
            y=y_values,
            marker_color=bar_colors,
            opacity=0.6,
            name='Message Count',
            # hovertemplate='<b>Week Range: %{text}</b><br>Message Count: %{y}<extra></extra>',
        ))

        # Add annotations for each bar
        for i, value in enumerate(y_values):
            fig.add_annotation(
                x=x_values[i],
                y=value,
                text=f'{value} <br>({days[i]})',
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-20 + (-value*0.01),
                arrowcolor='black',
                font=dict(size=10, color='black'),
                align='center'
            )

        # Update layout
        fig.update_layout(
            title='12. Most Messaged Day Count per Week Range - Colored Bar Graph',
            xaxis_title='Week Range',
            yaxis_title='Message Count',
            xaxis=dict(
                tickmode='array',
                tickvals=x_values,
                ticktext=week_labels,
                tickangle=45
            ),
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(0, int(max(y_values)) + 100, 100))
            ),
            autosize=False,
            width=1300,
            height=500,
            margin=dict(l=20, r=20, t=50, b=50)
        )

        # Save the plot as HTML file
        fig.write_html(save_url + r"\graph_12.html")
        # fig.show()

        print("Plot successfully created and saved.")

    except Exception as e:
        print(f"Error occurred: {e}")


"""Weekday / Weekend Plots"""
"""def wdwe_activity_level_plot_13(wde_data_list):

    # unpacking the list:
    weekday_high,weekday_moderate,weekday_low,weekend_high,weekend_moderate,weekend_low = \
        wde_data_list[0],wde_data_list[1],wde_data_list[2],wde_data_list[3],wde_data_list[4],wde_data_list[5]
        
    # Plotting the comparison bar chart
    plt.figure(figsize=(6,4))

    # Define the positions and width for the bars
    bar_width = 0.3
    bar_positions = np.arange(3)

    # Plot weekday data
    wdbar1 = plt.bar(bar_positions - bar_width / 2, [weekday_high.sum(), weekday_moderate.sum(), weekday_low.sum()],
                    width=bar_width, color='blue', label='Weekday')

    # Plot weekend data
    webar1 = plt.bar(bar_positions + bar_width / 2, [weekend_high.sum(), weekend_moderate.sum(), weekend_low.sum()],
                    width=bar_width, color='green', label='Weekend')

    # Set x-ticks to be in the center of the grouped bars
    plt.xticks(bar_positions, ['High', 'Moderate', 'Low'])

    # Add labels and title
    plt.xlabel('Activity Level')
    plt.ylabel('Message Count')
    plt.title('Weekday vs Weekend Message Count by Activity Level')
    plt.legend()

    # Annotate weekday bars
    for bar in wdbar1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 50, f'{height}', ha='center', va='bottom', fontsize=8)

    # Annotate weekend bars
    for bar in webar1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 50, f'{height}', ha='center', va='bottom', fontsize=8)

    plt.grid(True)
    plt.show()
    plt.savefig(save_url)
    plt.close()"""


def wdwe_activity_level_plot_13(wde_data_list):
    # Unpacking the list
    weekday_high, weekday_moderate, weekday_low, weekend_high, weekend_moderate, weekend_low = \
        wde_data_list[0], wde_data_list[1], wde_data_list[2], wde_data_list[3], wde_data_list[4], wde_data_list[5]

    fig = go.Figure()

    # Define the bar positions
    bar_positions = ['High', 'Moderate', 'Low']
    week_data = [weekday_high.sum(), weekday_moderate.sum(), weekday_low.sum()]
    weekend_data = [weekend_high.sum(), weekend_moderate.sum(),
                    weekend_low.sum()]

    # Add weekday bars
    fig.add_trace(go.Bar(
        x=bar_positions,
        y=week_data,
        name='Weekday',
        marker_color='blue',
        opacity=0.7
    ))

    # Add weekend bars
    fig.add_trace(go.Bar(
        x=bar_positions,
        y=weekend_data,
        name='Weekend',
        marker_color='green',
        opacity=0.7
    ))

    # Update layout
    fig.update_layout(
        title='13. Weekday vs Weekend Message Count by Activity Level - Grouped Bar Graph',
        xaxis_title='Activity Level',
        yaxis_title='Message Count',
        xaxis=dict(
            tickfont=dict(size=10),
            tickmode='array',
            tickvals=bar_positions
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(
                range(0, max(max(week_data), max(weekend_data)) + 100, 1000))
        ),
        barmode='group',
        autosize=False,
        width=1200,
        height=550,
        margin=dict(l=20, r=20, t=50, b=50)
    )

    # Add annotations
    for bar in range(len(bar_positions)):
        # Calculate the positions for annotations
        weekday_xpos = bar - 0.2
        weekend_xpos = bar + 0.2

        # Weekday annotations
        fig.add_annotation(
            x=weekday_xpos,
            y=week_data[bar],
            text=f'{week_data[bar]}',
            showarrow=False,
            # font=dict(size=10),
            align='center',
            yshift=10
        )

        # Weekend annotations
        fig.add_annotation(
            x=weekend_xpos,
            y=weekend_data[bar],
            text=f'{weekend_data[bar]}',
            showarrow=False,
            # font=dict(size=10),
            align='center',
            yshift=10
        )

    # Save the plot as HTML file
    save_plot = save_url + r"\graph_13.html"
    fig.write_html(save_plot)


"""Top 5 weekend/weekday holders"""
"""WEEKDAY Plot of user activity"""
"""def wdwe_top5_active_plot_14_15(wde_data_list):
    # unpack the compiled list
    # Define the top 5 users for each activity level in weekdays and weekends
    top_weekday_high_users = wde_data_list[0].nlargest(5)
    top_weekday_moderate_users = wde_data_list[1].nlargest(5)
    top_weekday_low_users = wde_data_list[2].nlargest(5)

    top_weekend_high_users = wde_data_list[3].nlargest(5)
    top_weekend_moderate_users = wde_data_list[4].nlargest(5)
    top_weekend_low_users = wde_data_list[5].nlargest(5)



    # Plotting the comparison bar chart
    plt.figure(figsize=(10,6))


    bar_width = 0.6

    # Plotting the bars
    bars_high = plt.bar(top_weekday_high_users.index, top_weekday_high_users.values, width=bar_width, color='wheat', label='High Activity Users')
    bars_moderate = plt.bar(top_weekday_moderate_users.index, top_weekday_moderate_users.values, width=bar_width, color='aquamarine', label='Moderate Activity Users')
    bars_low = plt.bar(top_weekday_low_users.index, top_weekday_low_users.values, width=bar_width, color='peru', label='Low Activity Users')

    # Annotate the bars with counts
    for bars in [bars_high, bars_moderate, bars_low]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 10, f'{height}', ha='center', va='bottom', fontsize=10)

    
    # Set the main x-axis labels
    plt.xticks(rotation=45, ha='right')
    # plt.ylim(-200, top_weekday_high_users.max() + 200)

    # Add labels and title
    plt.xlabel('Activity Level')
    plt.ylabel('Message Count')
    plt.title('Weekday Message Count by top 5 Active User in Each Category')
    plt.legend()


    plt.grid(True)
    plt.show()
    plt.savefig(save_url)
    plt.close()

    # 
    # Plotting the comparison bar chart
    plt.figure(figsize=(10,8))


    bar_width1 = 0.4
    # x = np.arange(len(top_weekday_high_users)+len(top_weekday_moderate_users)+len(top_weekday_low_users))


    webars_high1 = plt.bar(top_weekend_high_users.index, top_weekend_high_users.values, color='pink', label='High Activity Users')
    webars_moderate1 = plt.bar(top_weekend_moderate_users.index, top_weekend_moderate_users.values, color='lightblue', label='Moderate Activity Users')
    webars_low1 = plt.bar(top_weekend_low_users.index, top_weekend_low_users.values, color='lightgreen', label='Low Activity Users')

    # Annotate the bars with counts
    for webars1 in [webars_high1, webars_moderate1, webars_low1]:
        for webar1 in webars1:
            height1 = webar1.get_height()
            plt.text(webar1.get_x() + webar1.get_width() / 2, height1 + 10, f'{height1}', ha='center', va='bottom', fontsize=10)

    # Set the main x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Set y-axis ticks in increments of 100
    plt.yticks(range(0, int(top_weekend_high_users.max()) + 100, 50))

    # Add labels and title
    plt.xlabel('Activity Level')
    plt.ylabel('Message Count')
    plt.title('Weekend Message Count by top 5 Active User in Each Category')
    plt.legend()


    plt.grid(True)
    plt.show()
    plt.savefig(save_url)
    plt.close()"""


def wdwe_top5_active_plot_14(wde_data_list):
    # Unpack the compiled list
    top_weekday_high_users = wde_data_list[0].nlargest(5)
    top_weekday_moderate_users = wde_data_list[1].nlargest(5)
    top_weekday_low_users = wde_data_list[2].nlargest(5)

    top_weekend_high_users = wde_data_list[3].nlargest(5)
    top_weekend_moderate_users = wde_data_list[4].nlargest(5)
    top_weekend_low_users = wde_data_list[5].nlargest(5)

    # Plot weekday activity
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=top_weekday_high_users.index,
        y=top_weekday_high_users.values,
        name='High Activity Users',
        marker_color='wheat'
    ))
    fig1.add_trace(go.Bar(
        x=top_weekday_moderate_users.index,
        y=top_weekday_moderate_users.values,
        name='Moderate Activity Users',
        marker_color='aquamarine'
    ))
    fig1.add_trace(go.Bar(
        x=top_weekday_low_users.index,
        y=top_weekday_low_users.values,
        name='Low Activity Users',
        marker_color='peru'
    ))

    # Add annotations
    for x, y in zip(top_weekday_high_users.index, top_weekday_high_users.values):
        fig1.add_annotation(
            x=x,
            y=y,
            text=str(y),
            showarrow=False,
            font=dict(size=10),
            yshift=10,
            xanchor='center'
        )
    for x, y in zip(top_weekday_moderate_users.index, top_weekday_moderate_users.values):
        fig1.add_annotation(
            x=x,
            y=y,
            text=str(y),
            showarrow=False,
            font=dict(size=10),
            yshift=10,
            xanchor='center'
        )
    for x, y in zip(top_weekday_low_users.index, top_weekday_low_users.values):
        fig1.add_annotation(
            x=x,
            y=y,
            text=str(y),
            showarrow=False,
            font=dict(size=10),
            yshift=10,
            xanchor='center'
        )

    # Update layout
    fig1.update_layout(
        title='14.1. Weekday Message Count by Top 5 Active Users in Each Category',
        xaxis_title='Users',
        yaxis_title='Message Count',
        barmode='group',
        width=1300,
        height=500,
        margin=dict(l=20, r=20, t=50, b=50)
    )
    fig1.update_xaxes(tickangle=-90)

    # Save the plot as an HTML file
    fig1.write_html(save_url + r"\graph_14_1.html")

    # Plot weekend activity
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=top_weekend_high_users.index,
        y=top_weekend_high_users.values,
        name='High Activity Users',
        marker_color='pink'
    ))
    fig2.add_trace(go.Bar(
        x=top_weekend_moderate_users.index,
        y=top_weekend_moderate_users.values,
        name='Moderate Activity Users',
        marker_color='lightblue'
    ))
    fig2.add_trace(go.Bar(
        x=top_weekend_low_users.index,
        y=top_weekend_low_users.values,
        name='Low Activity Users',
        marker_color='lightgreen'
    ))

    # Add annotations
    for x, y in zip(top_weekend_high_users.index, top_weekend_high_users.values):
        fig2.add_annotation(
            x=x,
            y=y,
            text=str(y),
            showarrow=False,
            font=dict(size=10),
            yshift=10,
            xanchor='center'
        )
    for x, y in zip(top_weekend_moderate_users.index, top_weekend_moderate_users.values):
        fig2.add_annotation(
            x=x,
            y=y,
            text=str(y),
            showarrow=False,
            font=dict(size=10),
            yshift=10,
            xanchor='center'
        )
    for x, y in zip(top_weekend_low_users.index, top_weekend_low_users.values):
        fig2.add_annotation(
            x=x,
            y=y,
            text=str(y),
            showarrow=False,
            font=dict(size=10),
            yshift=10,
            xanchor='center'
        )

    # Update layout
    fig2.update_layout(
        title='14.2. Weekend Message Count by Top 5 Active Users in Each Category',
        xaxis_title='Users',
        yaxis_title='Message Count',
        barmode='group',
        width=1300,
        height=500,
        margin=dict(l=20, r=20, t=50, b=50)
    )
    fig2.update_xaxes(tickangle=-90)

    # Save the plot as an HTML file
    fig2.write_html(save_url + r"\graph_14_2.html")


"""Resampled D,W,M plot variation"""
# Plot the Resampled D, W, M message counts
"""def resampled_DWM_plot_16(df):
    dfn = df
    dfn.head(2)
    dfn.set_index('DateTime', inplace=True)

    # Handle missing values (if any)
    # dfn.fillna(method='ffill', inplace=True)
    dfn.ffill(inplace=True)
    # Resample data to monthly frequency and calculate the mean

    # Resample the data by day to aggregate the counts
    daily_messages = dfn['Message'].resample('D').count() # D, W, M
    monthly_messages = dfn['Message'].resample('ME').count() # D, W, M
    weekly_messages = dfn['Message'].resample('W').count() # D, W, M


    # Merged plot
    plt.figure(figsize=(14,6))
    plt.plot(daily_messages, label='daily Messages')
    plt.plot(weekly_messages, label='Weekly Messages')
    plt.plot(monthly_messages, label='Monthly Messages')

    plt.title('Daily, Weekly, Monthly Message Counts Over Time')
    plt.xlabel('Date')
    plt.ylabel('Message Count')
    plt.legend()
    plt.show()
    plt.savefig(save_url)
    plt.close()

    # Individual Plots
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 12), sharex=True)

    axes[0].plot(daily_messages.index, daily_messages.values, label='Daily Messages', color='blue')
    axes[0].set_title('Daily Messages')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].plot(weekly_messages.index, weekly_messages.values, label='Weekly Messages', color='green')
    axes[1].set_title('Weekly Messages')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    axes[2].plot(monthly_messages.index, monthly_messages.values, label='Monthly Messages', color='red')
    axes[2].set_title('Monthly Messages')
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.savefig(save_url)
    plt.close()"""


def resampled_DWM_plot_15(df):
    dfn = df
    dfn.set_index('DateTime', inplace=True)

    # Handle missing values (if any)
    dfn.ffill(inplace=True)

    # Resample data to monthly frequency and calculate the mean
    daily_messages = dfn['Message'].resample('D').count()  # D, W, M
    monthly_messages = dfn['Message'].resample('ME').count()  # D, W, M
    weekly_messages = dfn['Message'].resample('W').count()  # D, W, M

    # Merged plot
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=daily_messages.index,
                   y=daily_messages.values, mode='lines', name='Daily Messages'))
    fig1.add_trace(go.Scatter(x=weekly_messages.index,
                   y=weekly_messages.values, mode='lines', name='Weekly Messages'))
    fig1.add_trace(go.Scatter(x=monthly_messages.index,
                   y=monthly_messages.values, mode='lines', name='Monthly Messages'))

    fig1.update_layout(
        title='15.1. Daily, Weekly, Monthly Message Counts Over Time - Merged',
        xaxis_title='Date',
        yaxis_title='Message Count',
        width=1300,
        height=500,
        margin=dict(l=20, r=20, t=50, b=50)
    )

    # Save the plot as an HTML file
    fig1.write_html(save_url + r"\graph_15_1.html")

    # Individual plots
    fig2 = sp.make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=[
                            'Daily Messages', 'Weekly Messages', 'Monthly Messages'])

    fig2.add_trace(go.Scatter(x=daily_messages.index, y=daily_messages.values,
                   mode='lines', name='Daily Messages'), row=1, col=1)
    fig2.add_trace(go.Scatter(x=weekly_messages.index, y=weekly_messages.values,
                   mode='lines', name='Weekly Messages'), row=2, col=1)
    fig2.add_trace(go.Scatter(x=monthly_messages.index, y=monthly_messages.values,
                   mode='lines', name='Monthly Messages'), row=3, col=1)

    fig2.update_layout(

        width=1300,
        height=500,
        title_text='15.2. Daily, Weekly, Monthly Message Counts Over Time',
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=50)
    )

    # Update x-axis labels
    fig2.update_xaxes(title_text="Date", row=3, col=1)
    fig2.update_xaxes(tickangle=-90)

    # Save the plot as an HTML file
    fig2.write_html(save_url + r"\graph_15_2.html")


"""Plot A Word Cloud"""
"""def wordCloud_plot_17(word_freq):
        
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(6, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Message Frequency')
    plt.show()
    plt.savefig(save_url)
    plt.close()"""


"""To create a Graph of sender -  mention using AD"""

"""def basic_graph_spring_plot_18(Ga):

    # Remove edges where the target mention is empty
    Ga.remove_edges_from([(source, target) for source, target in Ga.edges() if target == ''])



    # Visualize the entire graph
    plt.figure(figsize=(25, 15))  # Increase the figure size
    pos = nx.spring_layout(Ga, seed=20, k=1.5)  # Adjust the spring layout parameter k

    # Draw the graph
    nx.draw_networkx_nodes(Ga, pos, node_size=150, node_color='lightblue', alpha=0.9)
    nx.draw_networkx_edges(Ga, pos, edge_color='gray', alpha=0.5)
    nx.draw_networkx_labels(Ga, pos, font_size=12, font_color='black')

    plt.title("Entire Network Graph")
    plt.show()
    plt.savefig(save_url)
    plt.close()
"""


def basic_graph_spring_plot_16(Ga):
    # Remove edges where the target mention is empty
    Ga.remove_edges_from([(source, target)
                         for source, target in Ga.edges() if target == ''])

    # Extract node positions using spring layout
    pos = nx.spring_layout(Ga, seed=150, k=1.8)

    # Extract node and edge information
    node_trace = go.Scatter(
        x=[pos[node][0] for node in Ga.nodes()],
        y=[pos[node][1] for node in Ga.nodes()],
        mode='markers+text',
        marker=dict(
            showscale=True,
            colorscale='Viridis',  # Set the colorscale to Viridis
            reversescale=True,
            size=10,
            color=[Ga.degree(node) for node in Ga.nodes()],
            colorbar=dict(
                thickness=15,
                title='Node Degree',
                xanchor='left',
                titleside='right'
            ),
            opacity=0.6
        ),
        text=[f'{node}' for node in Ga.nodes()],
        name='Nodes'
    )

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='gray'),
        hoverinfo='none',
        mode='lines'
    )

    for edge in Ga.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)

    # Create a figure with nodes and edges
    fig = go.Figure(data=[edge_trace, node_trace])

    # Update layout
    fig.update_layout(
        title='16. Entire Network Graph',
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=1300,
        height=500,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Save the plot as an HTML file
    fig.write_html(save_url + r"\graph_16.html")
    print("Graph Created Successfully")


"""Create the bar plot for centrality"""
"""def grouped_centrality_bar_plot_20(degree_centrality,betweenness_centrality,closeness_centrality,eigenvector_centrality):
        
    
    dkeys = list(degree_centrality.keys())
    dvalues = list(degree_centrality.values())

    ckeys = list(closeness_centrality.keys())
    cvalues = list(closeness_centrality.values())

    bkeys = list(betweenness_centrality.keys())
    bvalues = list(betweenness_centrality.values())

    ekeys = list(eigenvector_centrality.keys())
    evalues = list(eigenvector_centrality.values())


    # Ensure all keys are the same for plotting purposes
    all_keys = dkeys  # Assuming all centrality measures have the same keys
    index = np.arange(len(all_keys))

    # Grouped adjacent bar plot for each node
    plt.figure(figsize=(20, 8))

    bar_width = 0.4

    plt.bar(index, dvalues, color='red', width=bar_width, alpha=0.8, label='Degree Centrality')
    plt.bar(index + bar_width, cvalues, color='green', width=bar_width, alpha=0.8, label='Closeness Centrality')
    plt.bar(index + 2 * bar_width, bvalues, color='blue', width=bar_width, alpha=0.8, label='Betweenness Centrality')
    plt.bar(index + 3 * bar_width, evalues, color='orange', width=bar_width, alpha=0.8, label='Eigenvector Centrality')

    plt.xlabel('User Nodes')
    plt.ylabel('Centrality Scores')
    plt.title('Centrality Measures for Nodes')
    plt.xticks(index + 1.5 * bar_width, all_keys, rotation=90)

    # Set y-axis ticks with a step increment of 0.5
    plt.yticks(np.arange(0, 0.6, 0.1))

    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(save_url)
    plt.close()"""

"""Error in yticks and gaps between"""


def grouped_centrality_bar_plot_17(degree_centrality, betweenness_centrality, closeness_centrality, eigenvector_centrality):
    dkeys = list(degree_centrality.keys())
    dvalues = list(degree_centrality.values())
    ckeys = list(closeness_centrality.keys())
    cvalues = list(closeness_centrality.values())
    bkeys = list(betweenness_centrality.keys())
    bvalues = list(betweenness_centrality.values())
    ekeys = list(eigenvector_centrality.keys())
    evalues = list(eigenvector_centrality.values())

    # Create subplot figure with 4 rows and 1 column
    fig = sp.make_subplots(
        rows=4, cols=1,
        subplot_titles=('Degree Centrality', 'Closeness Centrality',
                        'Betweenness Centrality', 'Eigenvector Centrality'),
        shared_xaxes=False,  # Set shared_xaxes to False
        vertical_spacing=0.2  # Adjust the spacing between subplots
    )

    bar_width = 0.6  # Adjust the bar width as needed

    # Add Degree Centrality bar trace
    fig.add_trace(go.Bar(
        x=dkeys, y=dvalues, name='Degree Centrality',
        marker_color='red', opacity=0.8, width=bar_width
    ), row=1, col=1)

    # Add Closeness Centrality bar trace
    fig.add_trace(go.Bar(
        x=ckeys, y=cvalues, name='Closeness Centrality',
        marker_color='green', opacity=0.8, width=bar_width
    ), row=2, col=1)

    # Add Betweenness Centrality bar trace
    fig.add_trace(go.Bar(
        x=bkeys, y=bvalues, name='Betweenness Centrality',
        marker_color='blue', opacity=0.8, width=bar_width
    ), row=3, col=1)

    # Add Eigenvector Centrality bar trace
    fig.add_trace(go.Bar(
        x=ekeys, y=evalues, name='Eigenvector Centrality',
        marker_color='orange', opacity=0.8, width=bar_width
    ), row=4, col=1)

    # Update layout
    fig.update_layout(
        title_text='17. Centrality Measures for Nodes',
        xaxis_tickangle=-90,
        width=1800,  # Increased width
        height=1200,  # Increased height for better spacing
        margin=dict(l=50, r=50, t=50, b=100)
    )

    # Update x-axis for all subplots
    for i in range(1, 5):
        fig.update_xaxes(title_text='User Nodes', row=i, col=1)

    # Update y-axis for all subplots with step size of 0.1
    max_y_value = max(max(dvalues), max(cvalues), max(bvalues), max(evalues))
    # Generate ticks from 0 to max value with step 0.1
    y_ticks = np.arange(0, np.ceil(max_y_value * 10) / 10, 0.01)

    for i in range(1, 5):
        fig.update_yaxes(title_text='Centrality Scores', row=i, col=1)
        # Set ticks with step size of 0.1
        fig.update_yaxes(tickvals=y_ticks, row=i, col=1)

    # Save the plot as an HTML file
    fig.write_html(save_url + r"\graph_17.html")


"""Create the line plot for centrality"""
"""def grouped_centrality_line_plot_21(degree_centrality,betweenness_centrality,closeness_centrality,eigenvector_centrality):
    
    dkeys = list(degree_centrality.keys())
    dvalues = list(degree_centrality.values())

    ckeys = list(closeness_centrality.keys())
    cvalues = list(closeness_centrality.values())

    bkeys = list(betweenness_centrality.keys())
    bvalues = list(betweenness_centrality.values())

    ekeys = list(eigenvector_centrality.keys())
    evalues = list(eigenvector_centrality.values())

    all_keys = dkeys  # Assuming all centrality measures have the same keys
    index = np.arange(len(all_keys))

    plt.figure(figsize=(20, 14))

    plt.plot(index, dvalues, 'x-', color='blue', label='Degree Centrality', markerfacecolor='blue')
    plt.plot(index, cvalues, 'o-', color='green', label='Closeness Centrality', markerfacecolor='green')
    plt.plot(index, bvalues, '^-', color='pink', label='Betweenness Centrality', markerfacecolor='pink')
    plt.plot(index, evalues, 's-', color='red', label='Eigenvector Centrality', markerfacecolor='red')

    plt.xlabel('User Nodes')
    plt.ylabel('Centrality Scores')
    plt.title('Centrality Measures for Nodes (Line Plot)')
    plt.xticks(index, all_keys, rotation=90)

    # Set y-axis ticks from 0 to the length of the largest dvalues
    plt.yticks(np.arange(-0.01, 0.6, 0.01))

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(save_url)
    plt.close()"""


def grouped_centrality_line_plot_18(degree_centrality, betweenness_centrality, closeness_centrality, eigenvector_centrality):
    try:
        dkeys = list(degree_centrality.keys())
        dvalues = list(degree_centrality.values())
        ckeys = list(closeness_centrality.keys())
        cvalues = list(closeness_centrality.values())
        bkeys = list(betweenness_centrality.keys())
        bvalues = list(betweenness_centrality.values())
        ekeys = list(eigenvector_centrality.keys())
        evalues = list(eigenvector_centrality.values())

        all_keys = dkeys  # Assuming all centrality measures have the same keys

        # Calculate y-axis range and step size
        all_values = dvalues + cvalues + bvalues + evalues

        # Create figure
        fig = go.Figure()

        # Add line traces
        # Add line traces with updated line style and marker style
        fig.add_trace(go.Scatter(
            x=all_keys, y=dvalues, mode='lines+markers',
            name='Degree Centrality',
            line=dict(color='blue', dash='dot', width=4),  # Change line style
            marker=dict(symbol='circle-open-dot', size=6)  # Change marker size
        ))
        fig.add_trace(go.Scatter(
            x=all_keys, y=cvalues, mode='lines+markers',
            name='Closeness Centrality',
            line=dict(color='green', dash='dash',
                      width=3),  # Change line style
            marker=dict(symbol='cross-thin', size=6)  # Change marker size
        ))
        fig.add_trace(go.Scatter(
            x=all_keys, y=bvalues, mode='lines+markers',
            name='Betweenness Centrality',
            line=dict(color='pink', dash='solid',
                      width=2),  # Change line style
            marker=dict(symbol='star-open-dot', size=6)  # Change marker size
        ))
        fig.add_trace(go.Scatter(
            x=all_keys, y=evalues, mode='lines+markers',
            name='Eigenvector Centrality',
            line=dict(color='red', dash='dashdot'),  # Change line style
            marker=dict(symbol='octagon-dot', size=6)  # Change marker size
        ))

        # Calculate y-axis tick values
        y_max = max(dvalues + cvalues + bvalues + evalues)
        y_ticks = [round(i * 0.05, 2) for i in range(int(y_max / 0.05) + 1)]

        # Update layout
        fig.update_layout(
            title='18. Centrality Measures for Nodes (Line Plot)',
            xaxis_title='User Nodes',
            yaxis_title='Centrality Scores',
            xaxis_tickangle=-90,
            width=1300,
            height=500,
            margin=dict(l=40, r=40, t=40, b=40),
            # Set y-axis tick values
            yaxis=dict(tickmode='array', tickvals=y_ticks),

        )

        # Save the plot as an HTML file
        fig.write_html(save_url + r"\graph_18.html")

    except Exception as e:
        print(f"An error occurred: {e}")


"""user in each Commmunity bar plot """
"""def user_per_Community_plot_22(community_sizes,community_labels):
    plt.figure(figsize=(8, 6))
    plt.bar(community_labels, community_sizes, color='skyblue')

    plt.xlabel('Community')
    plt.ylabel('Number of Users')
    plt.title('Number of Users in Each Community')
    plt.xticks(rotation=90)
    # plt.grid(axis='y')
    plt.grid(axis='y', linestyle='--', alpha=0.7)


    plt.show()
    plt.savefig(save_url)
    plt.close()"""


def user_per_Community_plot_19(community_sizes, community_labels):
    # Create figure
    fig = go.Figure(data=[go.Bar(
        x=community_labels, y=community_sizes,
        marker_color='skyblue', opacity=0.7
    )])

    # Get the indices of the top 5 community sizes
    top_indices = sorted(range(len(community_sizes)),
                         key=lambda i: community_sizes[i], reverse=True)[:5]

    # Add annotations for all community sizes
    for idx, size in enumerate(community_sizes):
        if idx in top_indices:
            # Annotate top 5 bars with arrows
            fig.add_annotation(
                x=community_labels[idx],
                y=community_sizes[idx],
                text=f'{community_sizes[idx]}',
                xanchor='center',
                yanchor='bottom',
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40,
                font=dict(size=10, color='black')
            )
        else:
            # Annotate the rest without arrows
            fig.add_annotation(
                x=community_labels[idx],
                y=community_sizes[idx],
                text=f'{community_sizes[idx]}',
                xanchor='center',
                yanchor='bottom',
                showarrow=False,
                font=dict(size=10, color='black')
            )

    # Update layout
    fig.update_layout(
        title='19. Number of Users in Each Community',
        xaxis_title='Community',
        yaxis_title='Number of Users',
        xaxis_tickangle=-90,
        width=1300,
        height=500,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Save the plot as an HTML file
    fig.write_html(save_url + r"\graph_19.html")


"""user in each Commmunity graph plot """
"""def user_per_Community_graph_plot_23(Ga, communities):

    # Assign colors to communities
    cmap = plt.get_cmap('RdYlBu_r', len(communities))
    community_colors = {i: cmap(i) for i in range(len(communities))}

    # Create a list of node colors based on their community
    node_colors = []
    for node in Ga.nodes():
        for i, community in enumerate(communities):
            if node in community:
                node_colors.append(community_colors[i])
                break

    # Draw the graph with nodes colored by their community
    plt.figure(figsize=(10,6))
    pos = nx.spring_layout(Ga, seed=100)  # You can choose any layout you prefer

    nx.draw(Ga, pos, with_labels=False, node_size=50, font_size=8, node_color=node_colors, edge_color='gray', arrows=True, cmap='RdYlBu_r')

    # Create legend handles
    legend_handles = [mpatches.Patch(color=community_colors[i], label=f'Community {i + 1}') for i in range(len(communities))]

    # Add legend to the plot
    plt.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.1, 1.05))
    plt.title("Graph with Communities (Louvain Communities)")
    plt.axis('off')
    plt.show()
    plt.savefig(save_url)
    plt.close()"""


def user_per_Community_graph_plot_20(Ga, communities):
    # Assign colors to communities
    cmap = plt.get_cmap('RdYlBu_r', len(communities))
    community_colors = {
        i: f'rgb({cmap(i)[0]*255}, {cmap(i)[1]*255}, {cmap(i)[2]*255})' for i in range(len(communities))}

    # Create a list of node colors based on their community
    node_colors = []
    for node in Ga.nodes():
        for i, community in enumerate(communities):
            if node in community:
                node_colors.append(community_colors[i])
                break

    # You can choose any layout you prefer
    pos = nx.spring_layout(Ga, seed=100)

    edge_x = []
    edge_y = []
    for edge in Ga.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in Ga.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            colorscale='RdYlBu_r',
            color=[],
            size=10,
            line_width=2))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(Ga.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'Node {node} has {len(adjacencies[1])} connections')

    node_trace.marker.color = node_colors
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='20. Graph with Communities (Louvain Communities)',
                        titlefont_size=16,
                        showlegend=True,  # Show the legend
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        width=1300,  # Set the width
                        height=500,  # Set the height

                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper")],

                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    # Add legend for each community
    for i, color in community_colors.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color, line=dict(width=2)),
            showlegend=True,
            name=f'Community {i}'
        ))

    # Save the plot as an HTML file
    fig.write_html(save_url + r"\graph_20.html")


"""Create a matrix to count edges between communities"""

"""def inter_community_heatmap_plot_24(Ga, community_indices,communities):
        
    matrix = np.zeros((len(communities), len(communities)))

    for u, v in Ga.edges():
        if u in community_indices and v in community_indices:
            matrix[community_indices[u], community_indices[v]] += 1

    plt.figure(figsize=(6,4))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap='coolwarm')
    plt.xlabel('Community')
    plt.ylabel('Community')
    plt.title('Inter-Community Edge Count')
    plt.show()
    plt.savefig(save_url)
    plt.close()"""

# import logging


def inter_community_heatmap_plot_21(Ga, community_indices, communities):
    try:
        # Create a matrix to count edges between communities
        matrix = np.zeros((len(communities), len(communities)))

        for u, v in Ga.edges():
            if u in community_indices and v in community_indices:
                matrix[community_indices[u], community_indices[v]] += 1

        # Generate the heatmap data
        z = matrix
        x_labels = [f'Community {i+1}' for i in range(len(communities))]
        y_labels = [f'Community {i+1}' for i in range(len(communities))]

        # Prepare annotation text, removing diagonal values
        annotation_text = z.astype(int).astype(str)
        for i in range(len(communities)):
            annotation_text[i, i] = "*"  # Clear diagonal values

        # Define a custom colorscale
        colorscale = [
            [0, 'lightgrey'],       # Light color for 0 values
            # Slightly darker for non-zero
            [0.002, 'rgba(255, 255, 255, 0.8)'],
            [1.0, 'blue']           # Darker color for higher values
        ]

        # Create annotated heatmap figure
        fig = ff.create_annotated_heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale,
            annotation_text=annotation_text,
            showscale=True
        )

        fig.update_layout(
            title='21. Inter-Community Edge Count - Heat Map',
            xaxis_title='Community',
            yaxis_title='Community',
            xaxis=dict(side='bottom'),  # Move x-axis labels to bottom
            width=1300,
            height=500,
            margin=dict(l=40, r=40, t=50, b=50)
        )

        # Save the plot as an HTML file
        fig.write_html(save_url + r"\graph_21.html")

        print("Plot successfully created and saved.")

    except Exception as e:
        print(f"Error occurred: {e}")


"""Heat Map to show intersection users between communitites"""

"""def user_intersect_comm_plot_27(compiled_data):
        
    # Prepare data for the matrix plot
    users = list(compiled_data.keys())

    mcommunities = list(set(community for comms in compiled_data.values() for community in comms))
    matrix = np.zeros((len(users), len(mcommunities)))

    for i, user in enumerate(users):
        for community in compiled_data[user]:
            j = mcommunities.index(community)
            matrix[i, j] = 1


    # Prepare data for the heatmap
    heatmap_data = pd.DataFrame(matrix, index=users, columns=mcommunities)

    plt.figure(figsize=(8,6))
    sns.heatmap(heatmap_data, cmap='Blues', linewidths=0.5, annot=True, fmt='g')
    plt.xlabel('Communities')
    plt.ylabel('Users')
    plt.title('User Membership in Communities')
    plt.show()
    plt.savefig(save_url)
    plt.close()"""


def user_intersect_comm_plot_22(compiled_data):
    # Prepare data for the matrix plot
    users = list(compiled_data.keys())
    mcommunities = list(
        set(community for comms in compiled_data.values() for community in comms))
    matrix = np.zeros((len(users), len(mcommunities)))

    for i, user in enumerate(users):
        for community in compiled_data[user]:
            j = mcommunities.index(community)
            matrix[i, j] = 1

    # Prepare data for the heatmap
    heatmap_data = pd.DataFrame(matrix, index=users, columns=mcommunities)

    # Create heatmap figure
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Blues',
        showscale=True,
        # hoverinfo='z',
        text=heatmap_data.values.astype(int).astype(str),
        texttemplate="%{text}",
        textfont={"size": 12},
        xgap=1.5,  # gap between x-axis cells
        ygap=1.5   # gap between y-axis cells
    ))

    # Adding cell borders and annotations
    annotations = []
    for i in range(len(users)):
        for j in range(len(mcommunities)):
            value = heatmap_data.iat[i, j]
            if value > 0:
                annotations.append(
                    dict(
                        x=mcommunities[j],
                        y=users[i],
                        text=str(int(value)),
                        showarrow=False,
                        font=dict(color="white")
                    )
                )

    fig.update_layout(
        title='22. User Membership in Communities - Heat Map',
        width=1300,
        height=500,
        xaxis_tickangle=45,
        margin=dict(l=60, r=60, t=60, b=60),
        annotations=annotations
    )

    # Save the plot as an HTML file
    fig.write_html(save_url + r"\graph_22.html")

    print("Plot successfully created and saved.")



"""Plot MediaCount vs Name"""
"""def Total_media_per_user_plot_29(Emoji_DF):
        
    plt.figure(figsize=(20, 6))
    bars = plt.bar(Emoji_DF['Name'], Emoji_DF['MediaCount'], color='lightgreen')

    # Add annotations for each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height+10, str(int(height)), ha='center', va='bottom')

    # Mark the top 5 largest bars
    top_5_indices = Emoji_DF['MediaCount'].nlargest(5).index
    for i, bar in enumerate(bars):
        if i in top_5_indices:
            bar.set_color('orange')

    plt.xlabel('Name')
    plt.ylabel('Media Count')
    plt.yticks(range(0, int(Emoji_DF['MediaCount'].max()) + 10, 50))
    plt.title('Media Count by User')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.show()
    plt.savefig(save_url)
    plt.close()"""


def Total_media_per_user_plot_23(Emoji_DF):
    try:
        # Create a bar chart with Plotly using graph_objects
        fig = go.Figure()

        # Add bar trace
        fig.add_trace(go.Bar(
            x=Emoji_DF['Name'],
            y=Emoji_DF['MediaCount'],
            marker=dict(color=['orange' if count in Emoji_DF['MediaCount'].nlargest(
                5).values else 'lightgreen' for count in Emoji_DF['MediaCount']])
        ))

        # Update layout
        fig.update_layout(
            title='23. Media Count by User - Bar Graph',
            xaxis_title='Name',
            yaxis_title='Media Count',
            yaxis=dict(tickmode='array', tickvals=list(
                range(0, int(Emoji_DF['MediaCount'].max()) + 100, 200))),
            width=1300,
            height=500,
            margin=dict(l=40, r=40, t=50, b=50)
        )

        # Rotate x-axis labels
        fig.update_xaxes(tickangle=-90)

        # Add annotations for each bar
        for name, count in zip(Emoji_DF['Name'], Emoji_DF['MediaCount']):
            fig.add_annotation(
                x=name,
                y=count,
                text=f'{count}',
                showarrow=True,
                arrowhead=3,
                ax=0,
                ay=-15,
                align='center'
            )

        # Save the plot as an HTML file
        fig.write_html(save_url + r"\graph_23.html")

        print("Plot successfully created and saved.")

    except Exception as e:
        print(f"Error occurred: {e}")


"""Plot Total_Emojis_Count vs Name"""

"""def Total_emoji_per_user_plot_28(Emoji_DF):
        
    plt.figure(figsize=(20, 6))
    bars = plt.bar(Emoji_DF['Name'], Emoji_DF['Total_Emojis_Count'], color='skyblue')

    # Add annotations for each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height+10, str(int(height)), ha='center', va='bottom')

    # Mark the top 5 largest bars
    top_5_indices = Emoji_DF['Total_Emojis_Count'].nlargest(5).index
    for i, bar in enumerate(bars):
        if i in top_5_indices:
            bar.set_color('orange')

    plt.xlabel('Name')
    plt.ylabel('Total Emojis Count')
    plt.yticks(range(0, int(Emoji_DF['Total_Emojis_Count'].max()) + 10, 50))
    plt.title('Total Emojis Count by User')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.show()
    plt.savefig(save_url)
    plt.close()"""


def Total_emoji_per_user_plot_24(Emoji_DF):
    try:
        # Create a bar chart with Plotly using graph_objects
        fig = go.Figure()

        # Add bar trace
        fig.add_trace(go.Bar(
            x=Emoji_DF['Name'],
            y=Emoji_DF['Total_Emojis_Count'],
            marker=dict(color=['red' if count in Emoji_DF['Total_Emojis_Count'].nlargest(
                5).values else 'skyblue' for count in Emoji_DF['Total_Emojis_Count']])
        ))

        # Update layout
        fig.update_layout(
            title='24. Total Emojis Count by User - Bar Graph',
            xaxis_title='Name',
            yaxis_title='Total Emojis Count',
            yaxis=dict(tickmode='array', tickvals=list(
                range(0, int(Emoji_DF['Total_Emojis_Count'].max()) + 100, 200))),
            width=1300,
            height=500,
            margin=dict(l=40, r=40, t=50, b=50)
        )

        # Rotate x-axis labels
        fig.update_xaxes(tickangle=-90)

        # Add annotations for each bar
        for name, count in zip(Emoji_DF['Name'], Emoji_DF['Total_Emojis_Count']):
            fig.add_annotation(
                x=name,
                y=count,
                text=f'{count}',
                showarrow=True,
                arrowhead=3,
                ax=0,
                ay=-15,
                # font=dict(size=10),
                align='center'
            )

        # Save the plot as an HTML file
        fig.write_html(save_url + r"\graph_24.html")

        print("Plot successfully created and saved.")

    except Exception as e:
        print(f"Error occurred: {e}")



"""Filter the overall emoji counts to include only those with a count greater than 10"""

"""def count_distinct_emojis_plot_30(filtered_emoji_counts, symbola_font_path):
        
    # Prepare the data for plotting
    emojis = list(filtered_emoji_counts.keys())
    counts = list(filtered_emoji_counts.values())
    
    plt.figure(figsize=(12, 6))

    # Create a bar chart
    bars = plt.bar(emojis, counts, color='skyblue')

    # Add labels and title
    plt.xlabel('Emojis', fontsize=11)
    plt.ylabel('Count', fontsize=14)
    plt.title('Emoji Usage Count (Count > 15)', fontsize=16)

    # Use both Symbola and Noto Emoji fonts for rendering emojis
    symbola_prop = fm.FontProperties(fname=symbola_font_path)
    # noto_emoji_prop = fm.FontProperties(fname=noto_emoji_font_path)

    # Add emojis as labels on x-axis
    plt.xticks(emojis, labels=[emoji for emoji in emojis], fontsize=12, fontproperties=symbola_prop)
    # Adjust margins and spacing (combined for efficiency)
    plt.subplots_adjust(left=0.1, right=0.9, wspace=20.5)  # Adjust margins and bar spacing
    # plt.yticks(range(0, int(Emoji_DF['MediaCount'].max()) + 10, 50))
    plt.yticks(range(0,(max(counts)+1),50))

    # Add counts above the bars
    for i in range(len(emojis)):
        plt.text(i, counts[i] + 1, str(counts[i]), ha='center', va='bottom', fontsize=9, font='arial')

    plt.tight_layout()
    plt.grid(True)
    plt.show()
    plt.savefig(save_url)
    plt.close()"""


def count_distinct_emojis_plot_25(filtered_emoji_counts):
    try:
        # Prepare the data for plotting
        emojis = list(filtered_emoji_counts.keys())
        counts = list(filtered_emoji_counts.values())

        # Create a bar chart with Plotly
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=emojis,
            y=counts,
            marker=dict(color='skyblue'),
            # width=  # Adjust the bar width here
        ))

        # Update layout
        fig.update_layout(
            title='25. Emoji Usage Count (Count > 15) - Bar Graph',
            xaxis_title='Emojis',
            yaxis_title='Count',
            xaxis=dict(
                tickmode='array',
                tickvals=emojis,
                ticktext=emojis,
                tickangle=0  # Ensure ticks are not rotated
            ),
            yaxis=dict(tickmode='array', tickvals=list(
                range(-200, max(counts) + 1, 200))),
            width=1300,
            height=500,
            margin=dict(l=40, r=0, t=50, b=50)
        )

        # Add annotations for each bar
        for emoji, count in zip(emojis, counts):
            fig.add_annotation(
                x=emoji,
                y=count,
                text=str(count),
                textangle=-60, 
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay = -25 - (0 if count * 0.01 > 50 else count * 0.02),
                align='center'
            )

        # Save the plot as an HTML file
        fig.write_html(save_url + r"\graph_25.html")

        print("Plot successfully created and saved.")

    except Exception as e:
        print(f"Error occurred: {e}")


"""Top Emoji per user"""
"""def top_emoji_per_user_plot_31(top_emojis_per_person,symbola_font_path):
    # Prepare the data for plotting
    names = [item[0] for item in top_emojis_per_person]
    top_emojis = [item[1] for item in top_emojis_per_person]
    epcounts = [item[2] for item in top_emojis_per_person]
    # print(names[:3])
    # Set the figure size
    plt.figure(figsize=(20, 10))

    # Create a bar chart
    bars = plt.bar(names, epcounts, color='skyblue')

    # Add labels and title
    plt.xlabel('Users')
    plt.ylabel('Top Emoji Count')
    plt.title('Top Emoji Used by Each User')
    plt.xticks(rotation=90, font='calibri')
    # Add top emojis as labels on x-axis , fontsize=12, fontproperties=noto_emoji_prop
    plt.xticks(ticks=range(len(names)), labels=names)
    plt.yticks(range(0,(max(epcounts)+50),10))
    plt.margins(0.01)


    symbola_prop = fm.FontProperties(fname=symbola_font_path)
    # noto_emoji_prop = fm.FontProperties(fname=noto_emoji_font_path)

    # Add counts and emojis above the bars
    for i in range(len(names)):
        emoji = top_emojis[i]
        count = epcounts[i]
        plt.text(i, count + 1, f"{count}\n\n{emoji}", ha='center', va='bottom', font=symbola_prop)

    # Adjust margins and spacing
    plt.tight_layout()
    plt.grid(True,alpha=0.6)

    # Show the plot
    plt.show()
    plt.savefig(save_url)
    plt.close()"""


def top_emoji_per_user_plot_26(top_emojis_per_person):
    try:
        # Prepare the data for plotting
        names = [item[0] for item in top_emojis_per_person]
        top_emojis = [item[1] for item in top_emojis_per_person]
        epcounts = [item[2] for item in top_emojis_per_person]

        # Create a bar chart with Plotly
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=names,
            y=epcounts,
            marker=dict(color='skyblue'),
            width=0.5  # Adjust the bar width here
        ))

        # Update layout
        fig.update_layout(
            title='26. Top Emoji Used by Each User - Bar Graph',
            xaxis_title='Users',
            yaxis_title='Top Emoji Count',
            xaxis=dict(
                tickmode='array',
                tickvals=names,
                ticktext=names,
                tickangle=-90  # Ensure ticks are not rotated
            ),
            yaxis=dict(tickmode='array', tickvals=list(
                range(0, max(epcounts) + 50, 200))),
            width=1300,
            height=500,
            margin=dict(l=40, r=0, t=50, b=50)
        )

        # Add annotations for each bar
        for i, (name, emoji, count) in enumerate(zip(names, top_emojis, epcounts)):
            fig.add_annotation(
                x=name,
                y=count,
                text=f"{emoji}" if emoji is not None else "0",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-25,
                align='center'
            )

        # Save the plot as an HTML file
        fig.write_html(save_url + r"\graph_26.html")

        print("Plot successfully created and saved.")

    except Exception as e:
        print(f"Error occurred: {e}")

