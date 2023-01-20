import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
import json
import os


import plotly.graph_objects as go

if __name__ == "__main__":


    for disable_type in ["swing", "stance"]:
        t2m = {
            "swing" : "flexor",
            "stance": "extensor"
        }

        dmg_speed_count = json.load(open("dmg_speed_count_25_may.json"))
        x = sorted(set([e["speed"] for e in dmg_speed_count]))
        y = np.array(sorted(set([e["disable_count"] for e in dmg_speed_count])))/3
        y = y[:-5]

        def get_error(x, y):
            y = int(y*3)
            example = [e for e in dmg_speed_count 
                        if e["speed"] == x and e["disable_count"] == y and e["disable_phase"]==disable_type][0]
            return example["error_phase"]

        xv, yv = np.meshgrid(x, y)


        vf = np.vectorize(get_error)
        zv = vf(xv, yv)


        fig = go.Figure(data =
            go.Contour(
                z=zv,
                x=x,
                y=y,
                colorscale='thermal_r',
                colorbar=dict(
                    title='Phase duration error, au', # title here
                    titleside='right',
                )
            )
        )

        fig.update_layout(
            title={
                'text' : f"{t2m[disable_type]} damage",
                'x':0.5,
                'xanchor': 'center',
            },
            xaxis_title="CPG input",
            yaxis_title="Damaged neurons, %",
            font=dict(
                family="Arial",
                size=26
            )
        )

        # fig.show()

        os.makedirs("./images", exist_ok=True)
        name = f"{t2m[disable_type]}_dmg_input_contour"
        fig.write_image(f"images/{name}.png")
        fig.write_image(f"images/{name}.pdf")