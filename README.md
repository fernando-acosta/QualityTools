# QualityTools
 Quality tools is a framework for statistical process control using python. You will find a modlular repository that will help you in tasks like: calculating control limits and visualizing a process across time to ensure control.
 
 ## What can you do?
 
 It currently has two main models. The Shewhart control model and the Exponentially Weighted Moving Average control model. For both models Quality Tools help you to complete tasks such as: calculating control limits and plotting time series data. The framework modular design, was inspired by popular machine learning libraries that function using the fit and predict methods. In Quality Tools, the fit method enables you to establish your control limits and the predict method returns the points that are out of control with their respective index. 
 
 ## Assumptions
 Currently, the models assume that the quality characteristic to track is normally distributed and that the analyst knows the parameters beforehand. Future versions may include the ability to estimate the parameters from the data and relaxing the normality assumption. 
 
 ## A simple example using a Shewhart Control Model
 
 The following code block generates a set of 500 points, and it returns a dataframe with the points that are out of control and a ploted control chart. 
   
   ```
   from ControlCharts import ShewhartControlModel
   import numpy as np
   
   # Step 1: Set the Parameters
   size= 250
   miu_train= 48
   miu_test= 48.50
   sigma= 0.50

   # Step 2: Data
   x_train= np.random.normal(loc=miu_train, scale= sigma, size= (size,))
   x_test= np.random.normal(loc=miu_test, scale= sigma, size= (size,))
   
   # Step 3: Train the Model
   model= ShewhartControlModel(k=3)
   model.fit(miu=miu_train, sigma=sigma)

   # Step 4: Analyze and Predict
   combined_data= np.append(x_train, x_test)
   ooc= model.predict(combined_data)
   model.plot(combined_data)
   ```

### Results
  
  #### Time Series Plot
  <p align="center">
  <img src="https://github.com/fernando-acosta/QualityTools/Main/Examples/TimeSeriesExample1.png" />
  </p>
  
  #### Out of Control Output
  <p align="center">
  <img src="https://github.com/fernando-acosta/QualityTools/blob/main/Examples/OutofControlExample1.PNG" />
  </p>


 
