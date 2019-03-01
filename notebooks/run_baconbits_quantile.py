#!/usr/bin/env python
from __future__ import print_function, division
from collections import defaultdict, OrderedDict
import concurrent.futures
import gzip
import pickle
import json
import time
import numexpr
import pdb, traceback, sys
import uproot
import numpy as np
from fnal_column_analysis_tools import hist, lookup_tools

with open("metadata/datadef_qcd.json") as fin:
    datadef = json.load(fin)

extractor = lookup_tools.extractor()
extractor.add_weight_sets(["* * correction_files/n2ddt_transform_2017MC.root"])
extractor.finalize()
evaluator = extractor.make_evaluator()
n2ddt_rho_pt = evaluator[b"Rho2D"]

gpar = np.array([1.00626, -1.06161, 0.0799900, 1.20454])
cpar = np.array([1.09302, -0.000150068, 3.44866e-07, -2.68100e-10, 8.67440e-14, -1.00114e-17])
fpar = np.array([1.27212, -0.000571640, 8.37289e-07, -5.20433e-10, 1.45375e-13, -1.50389e-17])

def msd_weight(pt, eta):
    genw = gpar[0] + gpar[1]*np.power(pt*gpar[2], -gpar[3])
    ptpow = np.power.outer(pt, np.arange(cpar.size))
    cenweight = np.dot(ptpow, cpar)
    forweight = np.dot(ptpow, fpar)
    weight = np.where(np.abs(eta)<1.3, cenweight, forweight)
    return weight

# [pb]
dataset_xs = {k: v['xs'] for k,v in datadef.items()}

lumi = 1000.  # [1/pb]

dataset = hist.Cat("dataset", "Primary dataset")

gencat = hist.Bin("AK8Puppijet0_isHadronicV", "Matched", 4, 0., 4)
# one can relabel intervals, although process mapping obviates this
titles = ["QCD", "V(light) matched", "V(c) matched", "V(b) matched"]
for i,v in enumerate(gencat.identifiers()):
    setattr(v, 'label', titles[i])

jetpt = hist.Bin("AK8Puppijet0_pt", "Jet $p_T$", [450, 500, 550, 600, 675, 800, 1000])
jetpt_coarse = hist.Bin("AK8Puppijet0_pt", "Jet $p_T$", [450, 800])

jetmass = hist.Bin("AK8Puppijet0_msd", "Jet $m_{sd}$", 23, 40, 201)
jetmass_coarse = hist.Bin("AK8Puppijet0_msd", "Jet $m_{sd}$", [40, 100, 140, 200])
jetrho = hist.Bin("jetrho", r"Jet $\rho$", 52, -6, -2.1)
doubleb = hist.Bin("AK8Puppijet0_deepdoubleb", "Double-b", 40, 0., 1)
doublec = hist.Bin("AK8Puppijet0_deepdoublec", "Double-c", 20, 0., 1.)
doublecvb = hist.Bin("AK8Puppijet0_deepdoublecvb", "Double-cvb", 20, 0., 1.)
doubleb_coarse = [1., 0.93, 0.92, 0.89, 0.85, 0.7]
doubleb_coarse = hist.Bin("AK8Puppijet0_deepdoubleb", "Double-b", doubleb_coarse[::-1])
doublec_coarse = [0.87, 0.84, 0.83, 0.79, 0.69, 0.58]
doublec_coarse = hist.Bin("AK8Puppijet0_deepdoublec", "Double-c", doublec_coarse[::-1])
doublecvb_coarse = [0.93, 0.91, 0.86, 0.76, 0.6, 0.17, 0.12]
doublecvb_coarse = hist.Bin("AK8Puppijet0_deepdoublecvb", "Double-cvb", doublecvb_coarse[::-1])
n2ddt = hist.Bin("AK8Puppijet0_N2sdb1_ddt", "N2 DDT", 20, -0.25, 0.25)
n2ddt_coarse = hist.Bin("AK8Puppijet0_N2sdb1_ddt", "N2 DDT", [-0.1, 0.])
doublecsv = hist.Bin("AK8Puppijet0_doublecsv", "doublecsv", 80, -1., 1.)




hists = {}
#hists['hjetpt'] = hist.Hist("Events", dataset, gencat, jetmass, hist.Bin("AK8Puppijet0_pt", "Jet $p_T$", 60, 400, 1000),hist.Bin("N2quantile", "N2quantile", 50, 0, 1), doubleb, dtype='f')
hists['hjetpt'] = hist.Hist("Events", dataset, gencat, jetmass, hist.Bin("AK8Puppijet0_pt", "Jet $p_T$", 60, 400, 1000),hist.Bin("N2quantile", "N2quantile", 50, 0, 1), doublecsv, dtype='f')

with gzip.open("n2quantile_QCD_finern2.pkl.gz") as fin:
    n2hist = pickle.load(fin)
# n2hist being a 3d hist of pt, rho, n2
# which has its values replaced by the appropriate cumsum
n2q_array = n2hist.values()[()]
bins = tuple(ax.edges() for ax in n2hist.axes())
#n2ax = bins[2]
#n2ax = n2ax[::-1]
#bins_list = list(bins)
#bins_list[2] = n2ax
#bins = tuple(bins_list)

evaluator._functions['N2quantile'] = lookup_tools.dense_lookup.dense_lookup(n2q_array, bins)

#q = evaluator["N2quantile"](pt, rho, n2)
  

branches = [
    "AK8Puppijet0_pt",
    "AK8Puppijet0_eta",
    "AK8Puppijet0_msd",
    "AK8Puppijet0_isHadronicV",
    "AK8Puppijet0_deepdoubleb",
    "AK8Puppijet0_deepdoublec",
    "AK8Puppijet0_deepdoublecvb",
    "AK8Puppijet0_N2sdb1",
    "AK8Puppijet0_doublecsv",
]

tstart = time.time()


for h in hists.values(): h.clear()
nevents = defaultdict(lambda: 0.)

#np.set_printoptions(threshold=np.nan)

def processfile(dataset, file):
    # Many 'invalid value encountered in ...' due to pt and msd sometimes being zero
    # This will just fill some NaN bins in the histogram, which is fine
    tree = uproot.open(file)["Events"]
    arrays = tree.arrays(branches, namedecode='ascii')
    arrays["AK8Puppijet0_msd"] *= msd_weight(arrays["AK8Puppijet0_pt"], arrays["AK8Puppijet0_eta"])
    arrays["jetrho"] = 2*np.log(arrays["AK8Puppijet0_msd"]/arrays["AK8Puppijet0_pt"])
    #arrays["AK8Puppijet0_N2sdb1_ddt"] = arrays["AK8Puppijet0_N2sdb1"] - n2ddt_rho_pt(arrays["jetrho"], arrays["AK8Puppijet0_pt"])
    arrays["N2quantile"] = evaluator["N2quantile"](arrays["AK8Puppijet0_pt"],arrays["jetrho"],arrays["AK8Puppijet0_N2sdb1"])
    hout = {}
    for k in hists.keys():
        h = hists[k].copy(content=False)
        h.fill(dataset=dataset, **arrays)
        hout[k] = h
    return dataset, tree.numentries, hout


nworkers = 10
#fileslice = slice(None, 5)
fileslice = slice(None)
#with concurrent.futures.ThreadPoolExecutor(max_workers=nworkers) as executor:
with concurrent.futures.ProcessPoolExecutor(max_workers=nworkers) as executor:
    futures = set()
    for dataset, info in datadef.items():
        futures.update(executor.submit(processfile, dataset, file) for file in info['files'][fileslice])
    try:

        total = len(futures)
        processed = 0
        while len(futures) > 0:
            finished = set(job for job in futures if job.done())
            for job in finished:
                dataset, nentries, hout = job.result()
                nevents[dataset] += nentries
                for k in hout.keys():
                    hists[k] += hout[k]
                processed += 1
                print("Processing: done with % 4d / % 4d files" % (processed, total))
            futures -= finished
        del finished
    except KeyboardInterrupt:
        print("Ok quitter")
        for job in futures: job.cancel()
    except:
        for job in futures: job.cancel()
        raise

scale = dict((ds, lumi * dataset_xs[ds] / nevents[ds]) for ds in nevents.keys())
for h in hists.values(): h.scale(scale, axis="dataset")

dt = time.time() - tstart
print("%.2f us*cpu/event" % (1e6*dt*nworkers/sum(nevents.values()), ))
nbins = sum(sum(arr.size for arr in h._sumw.values()) for h in hists.values())
nfilled = sum(sum(np.sum(arr>0) for arr in h._sumw.values()) for h in hists.values())
print("Processed %.1fM events" % (sum(nevents.values())/1e6, ))
print("Filled %.1fM bins" % (nbins/1e6, ))
print("Nonzero bins: %.1f%%" % (100*nfilled/nbins, ))

# Pickle is not very fast or memory efficient, will be replaced by something better soon
with gzip.open("hists_quantile_qcd_doublecsv.pkl.gz", "wb") as fout:
    pickle.dump(hists, fout)

