from glob import glob
import pandas as pd

from tqdm import tqdm
import os

import argparse
import ast

RELOAD = True


parser = argparse.ArgumentParser()

parser.add_argument("-f", "--fields", type=ast.literal_eval, default='["nalu", "dist", "param", "extp"]')
parser.add_argument("-n", "--name", type=str, default="1_nalu_minimal_arith")
parser.add_argument("-p", "--pdf", action="store_true", default=False)
group = parser.add_mutually_exclusive_group()

group.add_argument("-i", "--int", action='store_true')
group.add_argument("-e", "--ext", action='store_true')

args = parser.parse_args()

fields = args.fields

if args.int:
    sel = "int"
else:
    sel = "ext"

if not RELOAD and os.path.exists("%s_results_%s.csv" % (args.name, sel)):
    df = pd.read_csv("%s_results_%s.csv" % (args.name, sel))
else:

    resultsfiles = [f for f in glob("results/*.csv")]

    results = []

    for f in tqdm(resultsfiles):


        if "exponential" in f:
            if "0.2" not in f and "0.8" not in f:
                  continue
        with open(f, "r") as csv:
            csv = csv.readlines()[-1].strip()
            if "nan" in csv.lower():
                print(csv)
                continue
            csv = csv.split("\t")

            if len(csv) != 12:
                print("Something possibly went wrong...")
                print(csv, f, len(csv))
                continue

            if "int.csv" in f:
                csv.append("int")
                if sel == "int":
                    results.append(csv)
            elif "ext.csv" in f:
                csv.append("ext")
                if sel == "ext":
                    results.append(csv)




#184     64000   nalu_simple_arith       naluv   normal  (5, 5)  (-10, 1)        103     div     6.18393E+01     6.14920E+00

    df = pd.DataFrame(results, columns=["epoch", "steps", "name", "nalu", "dist", "param", "extp", "seed", "op", "trainloss", "loss", "reinits", "exp"])
    df.to_csv("%s_results_%s.csv" % (args.name, sel))

df["loss"] = pd.to_numeric(df["loss"], errors="coerce")

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import cm
from matplotlib import pyplot as plt
fields.append("op")

import numpy as np


fig, ax1 = plt.subplots(2,2,figsize=(10, 7), sharey=True, sharex=True)

naludict = {}

naludict["nalu"] = "NALU"
naludict["nalu"] = "NALU"
naludict["nalui1"] = "iNALU (sw)"
naludict["nalui2"] = "iNALU (iw)"

def cf2lbl(config):
    config = config.replace("nalu__paperm", "NALU (m)")
    config = config.replace("nalu__paperv", "NALU (v)")
    config = config.replace("nalui1", "iNALU (sw)")
    config = config.replace("nalui2", "iNALU (iw)")
    cfields = config.split("_")

    dist = "U" if cfields[1] == "uniform" else "N" if cfields[1] == "normal" else "E" if cfields[1] == "exponential" else "?"
    if sel == "int":
        par = cfields[2]
    else:
        par = cfields[2] + ", " +cfields[3]
    return cfields[0] +" " + dist + par

if sel == "int":  
    fields.remove('extp')

ax = [item for sublist in ax1 for item in sublist]
succdict = {}
for i, o in enumerate(["add", "sub", "mul", "div"]):

    df["config"] = df[fields].apply(lambda x: '_'.join(x), axis=1)
    dff = df[df["exp"]==sel]
    dff = dff[dff["op"].apply(str.lower) == o].filter(fields + ["loss", "config"]).sort_values(by="config")




    labels = dff["config"].unique()
    from collections import defaultdict
    import string
    alph = list(string.ascii_lowercase)
    labelrefs = ["(%s) %s" % (x,cf2lbl(l)) for x, l in enumerate(labels)]
    nalus = [x.split("_")[0] for x in labels]
    nalucolors = {}
    color=iter(plt.get_cmap("Dark2")(np.linspace(0,1,len(nalus))))
    for n in nalus:
        c=next(color)
        nalucolors[n] = c
    
    values = {}

    for c in dff["config"].unique():
        values[c] = dff[dff["config"] == c]["loss"].values

    
    plt.sca(ax[i])
    ax[i].set_yscale("log", nonposy='clip')
    ax[i].set_title(o, y=0.88)
    plt.grid(color='silver', linestyle=':', linewidth=1)
    plt.axhline(y=1E-4, color='r', linestyle=":")
    
    if i % 2 == 0:
       plt.ylabel("mean squared error")


    for k, l in enumerate(labels):
        nalu = l.split("_")[0]
        xjitter = np.random.randn(len(values[l])) / 10 + k
        ax[i].scatter(xjitter,values[l], s=10, color=nalucolors[nalu])#, showfliers=False)

        succ = 0
        fail = 0
        for v in values[l]:
            if o == "add":
                print(v)
            if v < 1E-5:
                succ += 1    
            else:
                fail += 1
        if l.replace(o, "") not in succdict: 
            succdict[l.replace(o, "")] = defaultdict(int)
        succdict[l.replace(o, "")][i] = succ / (succ + fail)
    if i > 1:
        plt.xticks(range(0,len(labels)),labelrefs, rotation=90)
    else:
        plt.xticks(range(0,len(labels)),["" for r in labelrefs], rotation=90)

    ax[i].set_ylim([1E-16,1E13])
    for n in nalus:
        for t in ax[i].get_xticklabels():
            if naludict[n] in t.get_text():

                plt.setp(t, color=nalucolors[n])




fig.tight_layout()

plt.subplots_adjust(wspace=0)

if args.pdf:
   plt.savefig("%s_%s_scatter.pdf"%(args.name, sel))
else:
   plt.savefig("%s_%s_scatter.png"%(args.name, sel))
   plt.savefig("%s_%s_scatter.pdf"%(args.name, sel))
with open("%s_%s_legend.txt" % (args.name, sel), "w") as f:
    for k, l in enumerate(labels):
        f.write("("+labelrefs[i] +") " + l + ": " + str(succdict[l.replace(o, "")]) + "\n")

