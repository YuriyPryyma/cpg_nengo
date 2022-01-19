import sys
sys.path.insert(0, "../src")

import os
import numpy as np
import json
import matplotlib.pyplot as plt
import optimize
import tune_optimize_utils as utils
from sklearn.metrics import r2_score
from tqdm import tqdm
tau = 0.01

from p_tqdm import p_map

"""
[0.7267948251594097, 0.7106241737320016, 0.7289156854854392, 0.7158763882611964, 0.7267948251594097, 0.7267948251594097, 0.717292495081487, 0.7267948251594097, 0.5506025983010363, 0.5559164437509951, 1.3641456880545848, 1.3643741632936695, 1.3705549747487002, 1.3606414196799308, 1.3629809553318402, 1.3705549747487002, 1.3678934603882058, 1.3577080749163972, 1.3643741632936695, 1.3643741632936695, 2.2190343543512907, 2.2143057086993947, 2.1776261228600933, 2.18183624079501, 2.167919214017188, 2.167919214017188, 2.2190343543512907, 2.167919214017188, 2.271044920361499, 2.2234725717928554, 4.672234787965424, 4.58739168388706, 4.605641673356303, 4.605641673356303, 4.587491364760136, 4.641466895517306, 4.647112582609994, 4.592995578124929, 4.639868157980079, 4.672234787965424, 3.597635387834009, 3.5893558889061503, 3.605196128492733, 3.6013095148339405, 3.5893558889061503, 3.5899297258912513, 3.6114367584206333, 3.5899297258912513, 3.6013095148339405, 3.608873451505775, 3.942420747499778, 3.925765482503004, 3.889572530569012, 3.889572530569012, 4.0192987831590115, 4.031480702994726, 3.896520289494475, 3.9004506629235216, 3.925765482503004, 4.031480702994726, 3.6909560642148023, 3.724500682173447, 3.6909560642148023, 3.7462949035694892, 3.6909560642148023, 3.7104890582907872, 3.7140647688284654, 3.734590293589654, 3.6909560642148023, 3.7140647688284654, 3.5665581944606175, 3.5659222525020056, 3.5659222525020056, 3.5621296059565477, 3.5621296059565477, 3.5621296059565477, 3.5760137292082654, 3.5659222525020056, 3.5659222525020056, 3.5834475299651154, 3.534681238073149, 3.5352831998486263, 3.534681238073149, 3.541203941044356, 3.5432207031681453, 3.534681238073149, 3.534681238073149, 3.5400314935558943, 3.5361287825970704, 3.540997361828661, 3.581630261533217, 3.5742593054329292, 3.575479971480516, 3.5768108384211077, 3.5760593742231843, 3.575479971480516, 3.5817840417302236, 3.5742593054329292, 3.575479971480516, 3.581630261533217]
"""

if __name__ == "__main__":
    # os.environ["PYOPENCL_CTX"] = '0'

    # def error_with_disable(percentage):
    #     os.environ["PYOPENCL_CTX"] = '0'
    #     state_neurons = 2000
    #     disable = int((percentage/100)*state_neurons)
    #     res = optimize.simulation_error(params=utils.best_params[0], 
    #                                   progress_bar=False,
    #                                   state_neurons=state_neurons,
    #                                   disable=disable)
        
    #     return res[1]

    # percentages = []

    # for i in range(9, 11):
    #     for _ in range(5):
    #         percentages.append(i)

    
    # errors = p_map(error_with_disable, percentages)

    # print(errors)
 
    # errors_dict = {}

    # for i in range(len(percentages)):
    #     if percentages[i] not in errors_dict:
    #         errors_dict[percentages[i]] = []

    #     errors_dict[percentages[i]].append(errors[i])


    # json.dump(errors_dict, open("errors_dict.json", 'w'))

    errors_dict = json.load(open("errors_dict.json"))

    plt.style.use('seaborn-pastel')
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2

    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()

    fig.suptitle(f"Model error with damage", fontsize=24)

    x = sorted([ int(val) for val in list(errors_dict.keys())])

    y = [min(9, np.mean(errors_dict[str(per)])) for per in x]

    ax.plot(x, y, color='black', linewidth=2)

    ax.set_xlabel('Percentage of damage neurons, %', labelpad=10, fontsize=20)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 15, 20])
    ax.set_xlim([0, 20])

    ax.set_ylabel('Model error', labelpad=10, fontsize=20)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_ylim([0, 10])

    ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=20)

    fig.tight_layout()

    plt.show()
    # plt.savefig("dmg_neurons.pdf", format="pdf", dpi=200, bbox_inches="tight")
    # plt.savefig("dmg_neurons.png", dpi=200, bbox_inches="tight")

