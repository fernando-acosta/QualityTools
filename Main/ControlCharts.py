# -*- coding: utf-8 -*-
"""
Spyder Editor
Quality Tools: A Modular Framework for Statistical Process Control Using Python
Author: Fernando A. Acosta Perez
This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ShewhartControlModel:
    
    def __init__(self, k):
        """
        
        This class is cpable of training and predicting data using the Shewart control model.
        Currently, the programmed structure assumes that the analyst knows the parameters of the data (miu and sigma),
        and that the quality characteristic to be modeled is univariate and normally distributed. 

        Parameters
        ----------
        k : integer type
            How many standard deviation should the control limmits be. 

        Returns
        -------
        None.

        """
        self.k= k
    
    def fit(self, x, miu= None, sigma= None):
        """
        
        Description:
            
            This functions takes as input a set of data in vector form and calculates the control limits. 
            This particular control chart assumes that the population mean and variance are known.
        
        Assumptions: 
            
            The quality characteristic is normally distributed. 

        Parameters:
        ----------
        
        miu : float or integer type 
            Population Mean
        
        sigma : float or integer type 
            Population Standard Deviation

        Returns
        -------
        
        ucl: float or integer type 
            upper control limit
            
        lcl:float or integer type 
            lower control limit

        """
        
        # If user does not provide miu, estimate miu from the data
        if miu==None:
            miu= np.average(x)
            
        # If user does not provide sigma, estimate from the data
        n= len(x)
        if sigma== None:
            if n<=25:
                qcc= pd.read_csv('BiasControlConstants/quality_control_constants.csv')
                c4= qcc['c4'].loc[n-2]
                sigma= (np.std(x)*c4)/np.sqrt(n)
            else:
                sigma= (np.std(x))/np.sqrt(n)
        
        # Estimate control limits
        self.ucl= miu + self.k*sigma
        self.cl= miu
        self.lcl= miu - self.k*sigma
        
        return self.ucl, self.lcl
    
    def predict(self, x):
        
        """
        
        Description:

            This functions predict what points are outside the control limits.
            It returns an array with the out of control data points and their index.
        
        Parameters
        ----------
        x : numpy array
            Vector containing quality characteristic data

        Returns
        -------
        
        output_df: pandas dataframe
            A dataframe with the points that are out of control in the vector x, and their respective index.

        """
        
        m= len(x)
        idx= np.arange(1, m+1, 1)
        out_of_control_mask= ((x>self.ucl) | (x<self.lcl))
        output= {'Point Index': idx[out_of_control_mask], 'Data Point': x[out_of_control_mask]}
        output_df= pd.DataFrame(output)
        self.m=m
        
        return output_df
    
    def plot(self, x, dpi):
        """
        
        The purpose of this function is to visualize the points taht are out of control. It plots the control chart. 

        Parameters
        ----------
        x : numpy array
            A vector containing the quality characteristc data

        Returns
        -------
        None.

        """
        
        plt.figure(dpi=dpi)
        plt.plot(x)
        plt.plot(np.full(shape= (len(x),), fill_value= self.ucl), color= 'orange')
        plt.plot(np.full(shape= (len(x),), fill_value= self.cl), color= 'black', linestyle= 'dashed')
        plt.plot(np.full(shape= (len(x),), fill_value= self.lcl), color= 'orange')
        plt.xlabel('Sample Run')
        plt.ylabel('Data Point')
        plt.show()
        plt.close()
        
class EWMAControlModel:
    
    def __init__(self, l, lbd):
        
        self.l= l
        self.lbd= lbd
        
    def fit(self, m, miu= None, sigma= None):
        
        """
        This functions takes as input a set of data in vector form and calculates the control limits using the EWMA approach. It also calculates the statistic z. 
        It calculates the limits for the steady steady state and non steady state cases. It take the parameters miu and sigma as input. The model assume they are known.

        Parameters
        ----------
        x : numpy array
            A vector containing the quality characteristc data

        Returns
        -------
        ucl: float or integer type 
            Upper control limit
            
        lcl: float or integer type 
            Lower control limit

        """
    
        idx= np.arange(1,m+1,1)
        
        # Step 1: Non Steady State Limits
        self.ucl_nss= miu+self.l*sigma*np.sqrt((self.lbd/(2-self.lbd))*(1-(1-self.lbd)**(2*idx)))
        self.cl_nss= miu
        self.lcl_nss= miu-self.l*sigma*np.sqrt((self.lbd/(2-self.lbd))*(1-(1-self.lbd)**(2*idx)))
        
        # Step 2: Steady State Limits
        self.ucl= miu+self.l*sigma*np.sqrt((self.lbd/(2-self.lbd)))
        self.cl= miu
        self.lcl= miu-self.l*sigma*np.sqrt((self.lbd/(2-self.lbd)))
        
        self.m= m
        
        return self.ucl, self.lcl
    
    def predict(self, x):
        
        """
        
        Description:

            This functions predict what points are outside the control limits using the steady state limits.
            It returns an array with the out of control data points and their index.
        
        Parameters
        ----------
        x : numpy array
            Vector containing quality characteristic data

        Returns
        -------
        
        output_df: pandas dataframe
            A dataframe with the points that are out of control in the vector x, and their respective index.

        """
        # Part 1: Calculate Z, to be able to plot it
        m= len(x)
        z= np.empty(shape= (m,))
        z[0]= self.cl
        for i in range(1, m):
            z[i]= self.lbd*x[i]+(1-self.lbd)*z[i-1]
            
        # Step 2: Estimate Stabilization Index
        estab_index= np.round(np.log(0.00001)/(2*np.log(1-self.lbd)), 0)
        
        idx= np.arange(1, m+1, 1)
        out_of_control_mask= ((z>self.ucl) | (z<self.lcl))
        output= {'Point Index': idx[out_of_control_mask], 'Point Index After Steady State': idx[out_of_control_mask]-estab_index, 'Data Point': x[out_of_control_mask], 'Data Point Z': z[out_of_control_mask]}
        output_df= pd.DataFrame(output)
        
        return output_df
    
    def plot(self, x, ss=True):
        """
        
        The purpose of this function is to visualize the points taht are out of control. It uses steady state limits by default. When ss is False,
        the function plots the non steady state limits.

        Parameters
        ----------
        x : numpy array
            A vector containing the quality characteristc data

        Returns
        -------
        None.

        """
        
        z= np.empty(shape= (self.m,))
        z[0]= self.cl
        for i in range(1, self.m):
            z[i]= self.lbd*x[i]+(1-self.lbd)*z[i-1]
        
        if ss:
            plt.figure(dpi=500)
            plt.plot(z)
            plt.plot(np.full(shape= (self.m,), fill_value= self.ucl), color= 'orange')
            plt.plot(np.full(shape= (self.m,), fill_value= self.cl), color= 'black', linestyle= 'dashed')
            plt.plot(np.full(shape= (self.m,), fill_value= self.lcl), color= 'orange')
            plt.xlabel('Sample Run')
            plt.ylabel('Data Point')
            plt.show()
            plt.close()
        
        else:
            plt.figure(dpi=500)
            plt.plot(z)
            plt.plot(np.full(shape= (self.m,), fill_value= self.ucl_nss), color= 'orange')
            plt.plot(np.full(shape= (self.m,), fill_value= self.cl_nss), color= 'black', linestyle= 'dashed')
            plt.plot(np.full(shape= (self.m,), fill_value= self.lcl_nss), color= 'orange')
            plt.xlabel('Sample Run')
            plt.ylabel('Data Point')
            plt.show()
            plt.close()
            




