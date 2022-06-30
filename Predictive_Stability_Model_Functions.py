import numpy as np
import matplotlib.pyplot as plt
# import math
# import sklearn.metrics.mean_squared_error
import seaborn as sns
from scipy.stats import linregress

def model1_calc(x, storage_temp_array, storage_time_array, storage_humidity_array, T0=0, deg_infinity=100):
    """
    Parameters:
    x (array):
        A (float): A constant that we will want to optimise later on (typical value (10:45))
        Ea (float): A constant that we will want to optimise later on (typical value (-200000:-60000))
        B (float): A constant that we will want to optimise later on (typical values (-0.1:0.1))
    
    storage_temp_array (array): array of storage temperatures
    storage_time_array (array): array of storage times
    storage_humidity_array (array): array of storage humidity
    T0 (value): Initial T value

    Returns:
    array: Returns an array of y values for the equation 
    Exp(A+ (Ea/Storage Temp + 273.15)+B*Storage relative humidity)*Storage Time/days +T0
    
    Model shape: Straight line
    """
    y_pred = np.exp((x[0]+x[1]/(storage_temp_array+273.15))+x[2]*storage_humidity_array)*storage_time_array + T0
    return y_pred

def model2_calc(x, storage_temp_array, storage_time_array, storage_humidity_array, T0=0, deg_infinity=100):
    """
    Parameters:
    x (array):
        A (float): A constant that we will want to optimise later on (typical value (10:45))
        Ea (float): A constant that we will want to optimise later on (typical value (-200000:-60000))
        B (float): A constant that we will want to optimise later on (typical values (-3:3))
    
    storage_temp_array (array): array of storage temperatures
    storage_time_array (array): array of storage times
    storage_humidity_array (array): array of storage humidity
    T0 (value): Initial T value

    Returns:
    array: Returns an array of y values for the equation 
    Exp(A+ (Ea/Storage Temp + 273.15)+B*log(Storage relative humidity))*Storage Time/days +T0
    
    Model shape: Straight line
    """
    y_pred = np.exp((x[0]+x[1]/(storage_temp_array+273.15))+x[2]*np.log(storage_humidity_array))*storage_time_array + T0
    return y_pred



def model3_calc(x, storage_temp_array, storage_time_array, storage_humidity_array, T0=0, deg_infinity=100):
    """
    Parameters:
    x (array):
        A (float): A constant that we will want to optimise later on (typical value (10:45))
        Ea (float): A constant that we will want to optimise later on (typical value (-30000:-5000))
        B (float): A constant that we will want to optimise later on (typical values (-0.1:0.1))
    
    storage_temp_array (array): array of storage temperatures
    storage_time_array (array): array of storage times
    storage_humidity_array (array): array of storage humidity
    T0 (value): Initial T value

    Returns:
    array: Returns an array of y values for the equation 
    (Exp(log(A) + (Ea/Storage Temp + 273.15)+B*Storage relative humidity)*Storage Time/days)**0.5 +T0
    
    Model shape: Curve over time
    """
    y_pred = (np.exp((np.log(x[0])+x[1]/(storage_temp_array+273.15))+x[2]*storage_humidity_array)*storage_time_array)**0.5 + T0
    return y_pred

def model4_calc(x, storage_temp_array, storage_time_array, storage_humidity_array, T0=0, deg_infinity=100):
    """
    Parameters:
    x (array):
        A (float): A constant that we will want to optimise later on (typical value (10:45))
        Ea (float): A constant that we will want to optimise later on (typical value (-30000:-5000))
        B (float): A constant that we will want to optimise later on (typical values (-2:2))
    
    storage_temp_array (array): array of storage temperatures
    storage_time_array (array): array of storage times
    storage_humidity_array (array): array of storage humidity
    T0 (value): Initial T value

    Returns:
    array: Returns an array of y values for the equation 
    (Exp(log(A) + (Ea/Storage Temp + 273.15)+B*log(Storage relative humidity))*Storage Time/days)**0.5 +T0
    
    Model shape: Curve over time
    """
    y_pred = (np.exp((np.log(x[0])+x[1]/(storage_temp_array+273.15))+x[2]*np.log(storage_humidity_array))*storage_time_array)**0.5 + T0
    return y_pred

def model5_calc(x, storage_temp_array, storage_time_array, storage_humidity_array, T0=0, deg_infinity=100):
    """
    Parameters:
    x (array):
        A (float): A constant that we will want to optimise later on (typical value (10:45))
        Ea (float): A constant that we will want to optimise later on (typical value (-30000:-5000))
        B (float): A constant that we will want to optimise later on (typical values (-0.1:0.1))
        n (float): A constant that we will want to optimise later on (typical values (0.4:0.8 and can also be >1))
        
    storage_temp_array (array): array of storage temperatures
    storage_time_array (array): array of storage times
    storage_humidity_array (array): array of storage humidity
    T0 (value): Initial T value
    
    Returns:
    array: Returns an array of y values for the equation 
    (Exp(log(A) + (Ea/Storage Temp + 273.15)+B*Storage relative humidity)*Storage Time/days)**n +T0
    
    Model shape: Curve over time. When n is < 1 shape is the same as model 3 and 4. When n>1, the line curves upwards over time.
    """
    y_pred = (np.exp((np.log(x[0])+x[1]/(storage_temp_array+273.15))+x[2]*storage_humidity_array)*storage_time_array)**x[3] + T0
    return y_pred

def model6_calc(x, storage_temp_array, storage_time_array, storage_humidity_array, T0=0, deg_infinity=100):
    """
    Parameters:
    x (array):
        A (float): A constant that we will want to optimise later on (typical value (10:45))
        Ea (float): A constant that we will want to optimise later on (typical value (-30000:-5000))
        B (float): A constant that we will want to optimise later on (typical values (-2:2))
        n (float): A constant that we will want to optimise later on (typical values (0.4:0.8 and can also be >1))
        
    storage_temp_array (array): array of storage temperatures
    storage_time_array (array): array of storage times
    storage_humidity_array (array): array of storage humidity
    T0 (value): Initial T value
    
    Returns:
    array: Returns an array of y values for the equation 
    (Exp(log(A) + (Ea/Storage Temp + 273.15)+B*log(Storage relative humidity))*Storage Time/days)**n +T0
    
    Model shape: Curve over time. When n is < 1 shape is the same as model 3 and 4. When n>1, the line curves upwards over time.
    """
    y_pred = (np.exp((np.log(x[0])+x[1]/(storage_temp_array+273.15))+x[2]*np.log(storage_humidity_array))*storage_time_array)**x[3] + T0
    return y_pred

def model7_calc(x, storage_temp_array, storage_time_array, storage_humidity_array, T0=0, deg_infinity=100):
    """
    Parameters:
    x (array):
        A (float): A constant that we will want to optimise later on (typical value (10:45))
        Ea (float): A constant that we will want to optimise later on (typical value (-30000:-5000))
        B (float): A constant that we will want to optimise later on (typical values (-2:2))
        n (float): A constant that we will want to optimise later on (typical values (0.4:0.8 and can also be >1))
        
    storage_temp_array (array): array of storage temperatures
    storage_time_array (array): array of storage times
    storage_humidity_array (array): array of storage humidity
    T0 (value): Initial T value
    deg_infinity (value): The y limit of the graph and default value is 100
    
    Returns:
    array: Returns an array of y values for the equation 
    deg_infinity*(1 - Exp(-(Exp(log(A)+Ea/((Storage Temp +273.15)*8.314) + B*Storage relative humidity)*Storage Time/days)))+T0
    
    Model shape: Curve over time. When n is < 1 shape is the same as model 3 and 4. When n>1, the line curves upwards over time.
    """
    y_pred = deg_infinity*(1-np.exp(-(np.exp(np.log(x[0]) + x[1]/((storage_temp_array+273.15)*8.314) + x[2]*storage_humidity_array)*storage_time_array)))+T0
    return y_pred

def model8_calc(x, storage_temp_array, storage_time_array, storage_humidity_array, T0=0, deg_infinity=100):
    """
    TBC
    """
    
    
def model9_calc(x, storage_temp_array, storage_time_array, storage_humidity_array, T0=0, deg_infinity=100):
    """
    Parameters:
    x (array):
        A (float): A constant that we will want to optimise later on (typical value (1:30))
        Ea (float): A constant that we will want to optimise later on (typical value (-50000:-5000))
        B (float): A constant that we will want to optimise later on (typical values (-0.1:0.1))
        n (float): A constant that we will want to optimise later on (typical values (0.4:0.8 and can also be >1))
        AP (float): A constant that we will want to optimise later on (typical values (5:40))
        EaRp (float): A constant that we will want to optimise later on (typical values (-200000:-40000))
        Bp (float): A constant that we will want to optimise later on (typical values (-0.1:0.1))
        
    storage_temp_array (array): array of storage temperatures
    storage_time_array (array): array of storage times
    storage_humidity_array (array): array of storage humidity
    T0 (value): Initial T value
    
    Returns:
    array: Returns an array of y values for the equation 
    ((Ap + EaRp/((Temperature + 273.15)*8.314) + Bp*Relative Humidity)-T0)*(1-Exp(-(Exp(A + Ea/((Temperature +273.15)*8.314) +B*Relative Humidity)*Time/days)))+T0
    
    Model shape: Curve over time to a limit. The limit is a function of storage temperature and storage relative humidity.    
    """
    
    y_pred = ((x[4] + x[5]/((storage_temp_array+273.15)*8.314) +x[6]*storage_humidity_array) - T0)*(1-np.exp(-(np.exp(x[0]+x[1]/((storage_temp_array+273.15)*8.314) + x[2]*storage_humidity_array)*storage_time_array)))+T0
    return y_pred


def model10_calc(x, storage_temp_array, storage_time_array, storage_humidity_array, T0=0, deg_infinity=100):
    """
    Parameters:
    x (array):
        A (float): A constant that we will want to optimise later on (typical value (1:30))
        Ea (float): A constant that we will want to optimise later on (typical value (-50000:-5000))
        B (float): A constant that we will want to optimise later on (typical values (-2:2))
        n (float): A constant that we will want to optimise later on (typical values (0.4:0.8 and can also be >1))
        AP (float): A constant that we will want to optimise later on (typical values (5:40))
        EaRp (float): A constant that we will want to optimise later on (typical values (-200000:-40000))
        Bp (float): A constant that we will want to optimise later on (typical values (-2:2))
        
    storage_temp_array (array): array of storage temperatures
    storage_time_array (array): array of storage times
    storage_humidity_array (array): array of storage humidity
    T0 (value): Initial T value
    
    Returns:
    array: Returns an array of y values for the equation 
    ((Ap + EaRp/((Temperature + 273.15)*8.314) + Bp*Log(Relative Humidity))-T0)*(1-Exp(-(Exp(A + Ea/((Temperature +273.15)*8.314) +B*Log(Relative Humidity))*Time/days)))+T0
    
    Model shape: Curve over time to a limit. The limit is a function of storage temperature and storage relative humidity.    
    """
    
    y_pred = ((x[4] + x[5]/((storage_temp_array+273.15)*8.314) +x[6]*np.log(storage_humidity_array)) - T0)*(1-np.exp(-(np.exp(x[0]+x[1]/((storage_temp_array+273.15)*8.314) + x[2]*np.log(storage_humidity_array))*storage_time_array)))+T0
    return y_pred


def model11_calc(x, storage_temp_array, storage_time_array, storage_ph_array, T0=0, deg_infinity=100):
    """
    Parameters:
    x (array):
        Abase (float): A constant that we will want to optimise later on (typical value (1:15))
        Eabase (float): A constant that we will want to optimise later on (typical value (-15000:-2000))
        Cbase (float): A constant that we will want to optimise later on (typical values (0:3) - could be negative in some cases)
        Aacid (float): A constant that we will want to optimise later on (typical values (1:15))
        Eaacid (float): A constant that we will want to optimise later on (typical values (-15000:-2000))
        Cacid (float): A constant that we will want to optimise later on (typical values (-3:0))
        
    storage_temp_array (array): array of storage temperatures
    storage_time_array (array): array of storage times (minutes)
    storage_ph_array (array): array of storage pH
    T0 (value): Initial T value
    
    Returns:
    array: Returns an array of y values for the equation 
    (Exp(Abase + Eabase/(Temperature + 273.15) + Cbase * pH) + Exp(Aacid + Eaacid/(Temperature +273.15) +Cacid * pH)) * Time/minutes
        
    Model shape: Linear growth of degradant over time. V shape degradation rate across the pH range.
    """
    
    y_pred = (np.exp(x[0] + x[1]/(storage_temp_array + 273.15) + x[2] * storage_ph_array) + np.exp(x[3] + x[4]/(storage_temp_array + 273.15) +x[5] * storage_ph_array)) * storage_time_array
    return y_pred


def model12_calc(x, storage_temp_array, storage_time_array, storage_ph_array, T0=0, deg_infinity=100):
    """
    Parameters:
    x (array):
        Abase (float): A constant that we will want to optimise later on (typical value (-643290.324029087))
        Eabase (float): A constant that we will want to optimise later on (typical value (-6947.43455603601))
        Cbase (float): A constant that we will want to optimise later on (typical values (86932.934007657))
        Aacid (float): A constant that we will want to optimise later on (typical values (14.6599078055257))
        Eaacid (float): A constant that we will want to optimise later on (typical values (-6911.13385513849))
        Cacid (float): A constant that we will want to optimise later on (typical values (-0.0853484028445312))
        
    storage_temp_array (array): array of storage temperatures
    storage_time_array (array): array of storage times (minutes)
    storage_ph_array (array): array of storage pH
    T0 (value): Initial T value
    
    Returns:
    array: Returns an array of y values for the equation 
    deg_infinity(1-Exp(-((Exp(Abase + Eabase/(Temperature + 273.15) + Cbase * pH) + Exp(Aacid + Eaacid/(Temperature +273.15) +Cacid * pH)) * Time/minutes))) + T0
        
    Model shape: Growth of degradant over time to a limit. V shape degradation rate across the pH range.  
    """
    
    y_pred = deg_infinity *(1 - np.exp(-((np.exp(x[0] + x[1]/(storage_temp_array + 273.15) + x[2] * storage_ph_array) + np.exp(x[3] + x[4]/(storage_temp_array + 273.15) +x[5] * storage_ph_array)) * storage_time_array))) + T0
    return y_pred



def model_error_squared(x, y_values_array, model_function, storage_temp_array, storage_time_array, storage_humidity_array, T0 = 0, deg_infinity=100):
    """
    Parameters:
    A (int): A constant that we will want to optimise later on
    Ea (int): A constant that we will want to optimise later on
    B (int): A constant that we will want to optimise later on
    
    Returns:
    array: Returns the error in model prediction 
    Exp(A+ (Ea/Storage Temp + 273.15)+B*Storage relative humidity)*Storage Time/days +T0
    """
    y_pred = model_function(x, storage_temp_array, storage_time_array, storage_humidity_array, T0, deg_infinity)
    
    y_values_array = y_values_array[~np.isnan(y_pred)]
    y_pred = y_pred[~np.isnan(y_pred)]
    
    y_pred = y_pred[~np.isnan(y_values_array)]
    y_values_array = y_values_array[~np.isnan(y_values_array)]
    
    error = 0.0
    for i in range(len(y_values_array)):
        error = error + ((y_pred[i]-y_values_array[i]))**2
    return error

def actual_vs_predicted_scatter(predicted_array, actual_array):
    """
    Parameters:
    predicted_array: The predicted values from the model that is can be taken from the model1_calc function when you put in the optimised parameters into the function
    actual_array: The actual values from the dataset
    
    Returns:
    Plot: Returns a plot of predicted values against the actual values of RRT1o140 column in the dataset
    In the plot there is the R-sqaured value too
    """
    
    
    #Getting rid of any nan values 
#     actual_array = actual_array[~np.isnan(predicted_array)]
#     predicted_array = predicted_array[~np.isnan(predicted_array)]
    
#     predicted_array = predicted_array[~np.isnan(actual_array)]
#     actual_array = actual_array[~np.isnan(actual_array)]
    
    
    slope, intercept, r_value, p_value, std_err = linregress(predicted_array, actual_array)
    # Finding slope, intercept, r_value, p_value, std_error for the regression line
    r_value_sqaured = r_value**2 # Calculating R^2
    # rmse = math.sqrt(sklearn.metrics.mean_squared_error(actual_array, predicted_array))# Finding RMSE
    
    rmse = np.sqrt(sum(np.square(actual_array - predicted_array))/len(actual_array))
    
    max_actual = np.amax(actual_array) # Finding max of actual array to be used in annotating the plot below
    
    plt.clf()
    fig_dims = (12,5)
    fig, ax = plt.subplots(figsize=fig_dims)
    ax = plt.scatter(actual_array, predicted_array)
    plt.title("Predicted vs Actual values")
    
    x = [0,max_actual]
    y = [0,max_actual]
    plt.plot(x,y)
    plt.ylabel("Predicted values")
    plt.xlabel("Actual values")
    plt.annotate("R-sqaured = {}".format(r_value_sqaured), (0.5,2.5))
    plt.annotate("RMSE = {}".format(rmse), (0.5,2.3))
    plt.annotate("Y = {} + {}X".format(round(intercept,3),round(slope,3)), (0.5,2.1))
    # Plotting the a scatter plot of predicted vs actual
    
    return ax

def actual_vs_predicted_metrics(predicted_array, actual_array):
    """
    Parameters:
    predicted_array: The predicted values from the model that is can be taken from the model1_calc function when you put in the optimised parameters into the function
    actual_array: The actual values from the dataset
    
    Returns:
    Array: Array of R^2, RMSE and p-value of the linear model
    """
    
    
    #Getting rid of any nan values 
    actual_array = actual_array[~np.isnan(predicted_array)]
    predicted_array = predicted_array[~np.isnan(predicted_array)]
    
    predicted_array = predicted_array[~np.isnan(actual_array)]
    actual_array = actual_array[~np.isnan(actual_array)]
    
    
    slope, intercept, r_value, p_value, std_err = linregress(predicted_array, actual_array)
    # Finding slope, intercept, r_value, p_value, std_error for the regression line
    r_value_sqaured = r_value**2 # Calculating R^2
    # rmse = np.sqrt(sklearn.metrics.mean_squared_error(actual_array, predicted_array)) # Finding RMSE
    
    rmse = np.sqrt(sum(np.square(actual_array - predicted_array))/len(actual_array))
    # max_actual = np.amax(actual_array) # Finding max of actual array to be used in annotating the plot below
    return np.array([r_value_sqaured, rmse, p_value])