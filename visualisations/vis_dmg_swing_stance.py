import json
import numpy as np
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == "__main__":


    dmg_swing_stance = json.load(open('dmg_swing_stance.json', 'r'))

    data = defaultdict(list)

    for phase in ["swing", "stance", "all"]:
        for i in range(1, 15):
            runs = [e for e in dmg_swing_stance if e["disable_count"]==i and e["disable_phase"]==phase]
            # errors = [r["error_phase"] for r in runs]
            errors = [r["error"] for r in runs]

            # print(i, np.median(errors))

            data[phase].append(np.mean(errors))
            # break

        # break



    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    fig = plt.figure(figsize=(10, 10))

    ax = plt.gca()

    x = list(range(1, 15))

    ax.plot(x, data["swing"], color="#F8550D", linewidth=2, label='Swing')
    ax.plot(x, data["stance"], color="#0081D9", linewidth=2, label='Stance')
    ax.plot(x, data["all"], color="black", linewidth=2, label='Combined')

    ax.set_xlabel('Damaged neurons', labelpad=10, fontsize=20)
    ax.set_xticks([i*2 for i in range(9)])
    ax.set_xlim([0, 16])

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
