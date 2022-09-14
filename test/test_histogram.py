import plotly.express as px
import wandb 
import numpy as np 
import pandas as pd 

run = wandb.init(project="wandb-test", name="histogram-test")

x = np.arange(1, 11)
for step in range(4):
    # frames = []
    # y = step * x + step
    # plt.title("Matplotlib Demo")
    # plt.xlabel("x axis caption")
    # plt.ylabel("y axis caption")
    # plt.plot(x, y)
    # wandb.log({"plt":wandb.Plotly(plt.gcf())},step=step)


    df = px.data.tips()
    fig = px.histogram(df, x="total_bill", color="sex")
    wandb.log({"plt":wandb.Plotly(fig)},step=step)
# fig.show()