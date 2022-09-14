import wandb
import numpy as np 
import plotly.express as px

# Initialize a new run
run = wandb.init(project="wandb-test", name="plotly_html")

# Create a table
table = wandb.Table(columns = ["plotly_figure"])

for num in range(10):
    # Create path for Plotly figure
    path_to_plotly_html = "./plotly_figure_{}.html".format(num)

    # Example Plotly figure
    fig = px.scatter(x = [0, 1, 2, 3, 4], y = np.array([0, 1, 4, 9, 16])*np.sqrt(num))

    # Write Plotly figure to HTML
    fig.write_html(path_to_plotly_html, auto_play = False) # Setting auto_play to False prevents animated Plotly charts from playing in the table automatically

    # Add Plotly figure as HTML file into Table
    table.add_data(wandb.Html(path_to_plotly_html))

# Log Table
run.log({"test_table": table})
wandb.finish()