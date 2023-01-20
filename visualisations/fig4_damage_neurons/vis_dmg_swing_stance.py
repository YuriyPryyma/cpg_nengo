import json
import numpy as np
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    dmg_swing_stance = json.load(open('../data_gen/dmg_swing_stance_15.json', 'r'))

    d_swing = defaultdict(list)
    d_stance = defaultdict(list)

    for e in dmg_swing_stance:
        x = round((e["disable_count"]/300)*100, 1)
        if x >= 3.5:
            continue
        y = e["error"]
        if e["disable_phase"] == "swing":
            d_swing['x'].append(x)
            d_swing['y'].append(y)
        elif e["disable_phase"] == "stance":
            d_stance['x'].append(x)
            d_stance['y'].append(y)

    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    fig = plt.figure(figsize=(10, 10))

    ax = plt.gca()

    print(len(d_swing["x"]))
    df_swing = pd.DataFrame(data=d_swing)
    df_stance = pd.DataFrame(data=d_stance)

    sns.lineplot(ax = ax,
             data = df_swing,
             x = 'x',
             y = 'y',
             ci = 95,
             linewidth=2.5,
             label='Swing',
             color="#0081D9")

    sns.lineplot(ax = ax,
             data = df_stance,
             x = 'x',
             y = 'y',
             ci = 95,
             linewidth=2.5,
             label='Stance',
             color="#F8550D")

    ax.set_xlabel('Damaged neurons, %', labelpad=10, fontsize=20)
    ax.set_xticks([0.5, 1, 1.5, 2, 2.5, 3])
    ax.set_xlim([0, 3.5])

    ax.set_ylabel('Error', labelpad=10, fontsize=20)
    ax.set_yticks([i*2 for i in range(5)])
    ax.set_ylim([0, 8])

    ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=20)

    fig.tight_layout()

    os.makedirs("./images", exist_ok=True)
    f_name = f"./images/dmg_swing_stance"
    plt.savefig(f_name + ".pdf", format="pdf", dpi=200, bbox_inches="tight", transparent=True)
    plt.savefig(f_name + ".png", dpi=200, bbox_inches="tight")

    plt.show()
