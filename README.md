# QualityTools
 Quality tools is a framework for statistical process control using python. You will find a modlular repository that will help you for your analysis in taks like: calculating control limits and visualizing a process across time to ensure that your process is under control.
 
 ## What can you do?
 
 It currently has two main models. The Shewhart control model and the Exponentially Weighted Moving Average control model. For both models Quality Tools help you to complete tasks such as: calculating control limits and plotting time series data. The framework modular design, was inspired by popular machine learning libraries that function using the fit and predict methods. In Quality Tools, the fit method enables you to establish your control limits and the predict method returns the points that are out of control with their respective index. 
 
 ## Assumptions
 Currently, the models assume that the quality characteristic to track is normally distributed and that the analyst knows the parameters beforehand. Future versions may include the ability of estimating the parameters from the data and relaxing the normality assumption. 
 
 ## A simple example using an EWMA Control Model
 
 The following code block generates a set of 200 points, and it returns a dataframe with the points that are out of control and it plots the series with the limits. 
   
   from ControlCharts import EWMAControlModel
   import numpy as np
   
   x= np.random.normal(loc=48, scale= 0.50, size= (500,))
   model= EWMAControlModel(l=3, lbd=0.10)
   model.fit(10000, miu=48, sigma=0.50)
   ooc= model.predict(x)
   model.plot(x, ss=True)

### Results
  
  #### Time Series Plot
  <p align="center">
  <img src="https://github.com/fernando-acosta/QualityTools/blob/main/Examples/TimeSeriesExample1.png" />
  </p>
  
  #### Out of Control Output
  <p align="center">
  <img src="https://github.com/fernando-acosta/QualityTools/blob/main/Examples/OutofControlExample1.PNG" />
  </p>


 
