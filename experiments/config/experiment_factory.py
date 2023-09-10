# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 11:18:04 2016

@author: finn
"""

import datetime as dt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype']=42
matplotlib.rcParams['ps.fonttype']=42
from src.algorithms import GeneralCausal,  ParallelCausal, SuccessiveRejects, AlphaUCB, RandomArm, ThompsonSampling
import pickle
import os

print("LOADING EXPERIMENT CONFIG MODULE AGAIN")

def now_string():
    return dt.datetime.now().strftime('%Y%m%d_%H%M')

# TODO : Run debug and check each parameters types
  
class ExperimentFactory(object):
    def __init__(self, experiment_id: int):
        self.started = now_string()
        self.experiment_id = experiment_id
        self.REGRET_LABEL = "Regret"
        self.HORIZON_LABEL = "T"
        self.M_LABEL = "m(q)"
        self.markers = ['s', 'o', 'D', '*', "^", "p"]
        self.colors = ["red", "green", "blue", "purple", "cyan", "orange"]
        self.algorithms = [ParallelCausal, GeneralCausal, SuccessiveRejects, AlphaUCB, RandomArm, ThompsonSampling]
        
        try:
            os.makedirs("results")
        except OSError:
            if not os.path.isdir("results"):
                raise
        for indx, a in enumerate(self.algorithms):
            a.marker = self.markers[indx]
            a.color = self.colors[indx]
              
    def plot_regret(self, regret , xvals , xlabel , algorithms , legend_loc="upper right", legend_extra=[], log=True):
        s_axis = regret.ndim - 1 # assumes the last axis is over simulations
        simulations = regret.shape[-1]
        mu = regret.mean(s_axis)
        error = 3 * regret.std(s_axis) / np.sqrt(simulations)
        fig, ax = plt.subplots()
        for indx, alg in enumerate(algorithms):    
            ax.errorbar(xvals, mu[indx,:], yerr=error[indx,:], label=alg.label, linestyle="", marker=alg.marker, color=alg.color)
        
        for alg in legend_extra: # trick matplotlib into adding legend for additional algorithms
            ax.errorbar(-1,-1, yerr=.0001, label=alg.label, color=alg.color, marker=alg.marker, linestyle="")

        upper=np.nanmax(mu+error)
            
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(self.REGRET_LABEL)
        ax.set_ylim(bottom=0, top=upper)
        ax.set_xlim(left=0, right=max(xvals))
        
        if legend_loc is not None:
            ax.legend(loc=legend_loc, numpoints=1)
        fig_name = "results/experiment{0}_{1}.pdf".format(self.experiment_id, self.started)
        fig.savefig(fig_name, bbox_inches="tight")
        
        if log:
            self.log_regret(regret, xvals, xlabel, algorithms)
        
    def log_code(self):
        out = "results/experiment{0}_{1}_settings.txt".format(self.experiment_id, self.started)
        experiment_file = "experiment{0}.py".format(self.experiment_id)
        with open(experiment_file,"r") as f, open(out,"w") as o:
            o.write(f.read())
    
    def log_regret(self, regret, xvals, xlabel, algorithms):
        filename = "results/experiment{0}_{1}.pickle".format(self.experiment_id, self.started)
        with open(filename, mode='wb') as out:
            tpl = (regret, xvals, xlabel, algorithms)
            pickle.dump(tpl, out)
            
    def read_data(self, filename):
        with open(filename, mode="rb") as f:
            tpl = pickle.load(f)
        return tpl
        
   
        

                

            
            

            
        
        
    
        

            


 

    

    

    