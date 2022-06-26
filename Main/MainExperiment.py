# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 18:02:07 2022

Fernando Acosta Perez
Jomayra Torres Caro
Carmelo Velez Santiago
Yasiel Valentin Acevedo
Grupo 7

Lab #5 Exponentially Weighted Moving Average
Instructor: Bryan Ortiz Torres

"""

from ControlCharts import ShewhartControlModel, EWMAControlModel
import numpy as np
import pandas as pd
from scipy.stats import norm

# Set random seed for numpy
np.random.seed(7)

def MainExperimentShewhart(delta, k, miu, sigma, n, m):
    """

    Parameters
    ----------
    delta : A list/array type with the variations in miu as a function of the standard deviation 
    that we want to consider when running the experiment.
    
    k : The value for k when evaluating the shewart control charts in the experiment
    
    miu : miu for generating data
    
    sigma : Standard deviation for generating data
    
    n : sample size
    
    m: total runs

    Returns
    -------
    results: A dataframe with the results of the experiment

    """
    
    results= {'Delta': [], 'Empirical ARL': [], 'Theoretical ARL': [], 'Theoretical RL Stdev': []}
    model= ShewhartControlModel(k)
    ucl, lcl= model.fit(miu= miu, sigma=sigma)
    
    for d in delta:
        empirical_arl_list= []
        miu_i= miu + d*sigma
        x= np.random.normal(loc=miu_i, scale= sigma, size= (n, m))
        power= 1-(norm.cdf(ucl, miu_i, sigma)-norm.cdf(lcl, miu_i, sigma))
        theoretical_arl= 1/power
        theoretical_stdev= np.sqrt((1-power)/(power**2))
        
        for i in range(m): # iterate over each run
            out_of_control= model.predict(x[:, i])
        
            try: # this part may lead to an error if there are no points out of control
                empirical_arl_list.append(min(out_of_control['Point Index']))
            
            except: # continue the code in the case that there are no points out of control
                continue
        
        # Store Results
        results['Delta'].append(d)
        results['Empirical ARL'].append(np.average(empirical_arl_list))
        results['Theoretical ARL'].append(theoretical_arl)
        results['Theoretical RL Stdev'].append(theoretical_stdev)
    
    results= pd.DataFrame(results)
    
    return results
    
def MainExperimentEWMA(delta, l, lbd, miu, sigma, n, m):
    """

    Parameters
    ----------
    delta : A list/array type with the variations in miu as a function of the standard deviation 
    that we want to consider when running the experiment
    
    l: An array type with parameter L, it controls the width of the control limits
    
    lbd : An array type containing the parameter lambda (lbd)
    
    miu : miu for generating data
    
    sigma : Standard deviation for generating data
    
    n : sample size
    
    m: total runs
    
    Returns
    -------
    results: A dataframe with the results of the experiment

    """
    
    results= {'Delta': [], 'L': [], 'Lambda': [], 'Empirical ARL': []}
    
    for L in range(len(l)):
        for lambd in range(L, L+2):
            model= EWMAControlModel(l=l[L], lbd= lbd[lambd])
            ucl, lcl= model.fit(m=n, miu= miu, sigma=sigma)
            
            for d in delta:
                empirical_arl_list= []
                miu_i= miu + d*sigma
                x= np.random.normal(loc=miu_i, scale= sigma, size= (n, m))
                
                for i in range(m): # iterate over each run
                    out_of_control= model.predict(x[:, i])
                
                    try: # this part may lead to an error if there are no points out of control
                        empirical_arl_list.append(min(out_of_control['Point Index After Steady State']))
                    
                    except: # continue the code in the case that there are no points out of control
                        continue
                
                # Store Results
                results['Delta'].append(d)
                results['L'].append(l[L])
                results['Lambda'].append(lbd[lambd])
                results['Empirical ARL'].append(np.average(empirical_arl_list))
        
    results= pd.DataFrame(results)
    
    return results
    



print('Running Experiments Shewhart...')
results_shewhart= MainExperimentShewhart(delta=[0, 0.25, 0.50, 0.75, 1, 1.50, 2], 
                                miu= 48, 
                                sigma=0.50,
                                n= 10000,
                                m=500,
                                k= 3)

print('Running Experiments EWMA...')
results_ewma= MainExperimentEWMA(delta=[0, 0.25, 0.50, 0.75, 1, 1.50, 2],
                                     l= [3.054, 2.998, 3],
                                     lbd= [0.40, 0.25, 0.10, 0.05],
                                     miu= 48, 
                                     sigma=0.50,
                                     n= 10000,
                                     m=500)


