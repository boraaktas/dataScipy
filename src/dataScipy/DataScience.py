import numpy as np
import matplotlib.pyplot as plt

##########################################################################################
###################################### ERROR METRICS #####################################
##########################################################################################

def make_length_equal_to_compare(realVals, forecastVals):
    """
    Make the length of the real and forecast values equal.
    :param realVals: the real values (list)
    :param forecastVals: the forecast values (list)
    :return: the real and forecast values with equal length (list, list)    
    """
    no_None = forecastVals.count(None)
    if no_None != 0:
        realVals = list(np.array(realVals)[no_None:])
        forecastVals = list((np.array(forecastVals)[no_None:])[:len(realVals)])

    return realVals, forecastVals



def MSE(realVals, forecastVals):
    """
    Compute the mean squared error between the real and forecast values.
    :param realVals: the real values (list)
    :param forecastVals: the forecast values (list)
    :return: the mean squared error (float)
    """
    realVals, forecastVals = make_length_equal_to_compare(realVals, forecastVals)

    return np.mean((np.array(realVals) - np.array(forecastVals))**2)



def RMSE(realVals, forecastVals):
    """
    Compute the root mean squared error between the real and forecast values.
    :param realVals: the real values (list)
    :param forecastVals: the forecast values (list)
    :return: the root mean squared error (float)
    """ 
    realVals, forecastVals = make_length_equal_to_compare(realVals, forecastVals)

    return np.sqrt(MSE(realVals, forecastVals))



def MAE(realVals, forecastVals):
    """
    Compute the mean absolute error between the real and forecast values.
    :param realVals: the real values (list)
    :param forecastVals: the forecast values (list)
    :return: the mean absolute error (float)
    """
    realVals, forecastVals = make_length_equal_to_compare(realVals, forecastVals)

    return np.mean(np.abs(np.array(realVals) - np.array(forecastVals)))



def MAPE(realVals, forecastVals):
    """
    Compute the mean absolute percentage error between the real and forecast values.
    :param realVals: the real values (list)
    :param forecastVals: the forecast values (list)
    :return: the mean absolute percentage error (float)
    """
    realVals, forecastVals = make_length_equal_to_compare(realVals, forecastVals)

    return np.mean(np.abs((np.array(realVals) - np.array(forecastVals))/np.array(realVals)))*100



def calculate_Error(error_method, realVals, forecastVals):
    """
    Compute the error between the real and forecast values.
    :param error_method: the error method to use (function)
    :param realVals: the real values (list)
    :param forecastVals: the forecast values (list)
    :return: the error (float)
    """
    realVals, forecastVals = make_length_equal_to_compare(realVals, forecastVals)
    
    return error_method(realVals, forecastVals)



def print_error_summary(data, forecasts, **error_method):
    """
    Print the error summary for the forecasts.
    :param data: the time series
    :param forecasts: the forecasts
    """
    print('Error Summary')
    print('-------------')
    for method, error in error_method.items():
        print(f'{method}: {error(data, forecasts):.3f}') 



##########################################################################################
##########################################################################################
##########################################################################################


##########################################################################################
##################################### FORECAST METHODS ###################################
##########################################################################################

def naive_forecast(data):
    """
    Forecast the next value in the time series using the naive method.
    :param data: time series (list)
    :return: naive forecast (float or integer or None)
    """
    if len(data) == 0:
        return None
    
    return data[-1]



def seasonal_forecast(data, k):
    """
    Forecast the next value in the time series using the single value seasonal method.
    :param data: time series (list)
    :param k: length of the season (integer)
    :return: single value seasonal forecast (float or integer or None)
    """
    if len(data) < k:
        return None
    
    return data[-k]



def MA_forecast(data, N):
    """
    Forecast the next value in a time series using a moving average.
    :param data: the time series (list)
    :param n: the number of previous values to use in the moving average (integer)
    :return: the forecast (float or integer or None)
    """
    if len(data) < N:
        return None
    
    return np.mean(data[-N:])



def ES_forecast(data, alpha):
    """
    Forecast the next value in a time series using an exponential smoothing.
    :param data: the time series (list)
    :param alpha: the smoothing parameter (float)
    :return: the forecast (float or integer or None)
    """

    if len(data) == 0:
        raise ValueError('data must not be empty')

    if len(data) == 1:
        return data[-1]
    
    return alpha * data[-1] + (1 - alpha) * ES_forecast(data[:-1], alpha)



def make_forecast(data, forecast_method, **kwargs):
    """
    Create a forecast for the next value in the time series.
    :param data: the time series (list)
    :param forecast_method: the forecasting method to use (function)
    :param kwargs: the keyword arguments for the forecasting method (parameters for the forecasting method)
    :return: the forecast (float or integer or None)
    """
    return forecast_method(data, **kwargs)



def make_forecast_for_all_data(data, period_ahead, forecast_method, **kwargs):
    """
    Create a forecast for each value in the time series.
    :param data: the time series (list)
    :param forecast_method: the forecasting method to use (function)
    :param kwargs: the keyword arguments for the forecasting method (parameters for the forecasting method)
    :return: the forecast list (list)
    """
    #forecasts = np.repeat(None, len(data) + period_ahead)
    forecasts = list(np.repeat(None, period_ahead))
    
    for i in range(len(data)):
        forecasts.append(forecast(data[:i+1], forecast_method, **kwargs))

    return list(forecasts)



##########################################################################################
##########################################################################################
##########################################################################################


##########################################################################################
#################################### FORECAST PLOTTING ###################################
##########################################################################################



def plot_forecasts(data, horizon, forecasts):
    """
    Plot the forecasts for the time series.
    :param data: the time series
    :param forecasts: the forecasts
    :param forecast_method: the forecasting method to use
    :param kwargs: the keyword arguments for the forecasting method
    """
    
    number_of_dots = len(horizon)

    plt.figure(figsize=(12, 3))
    plt.xticks(np.arange(min(horizon), max(horizon)+1, 2))
    plt.grid()
    plt.plot(horizon, data[:number_of_dots], label='data')
    plt.plot(horizon, forecasts[:number_of_dots], label='forecast')
    plt.xlabel('Month')
    plt.ylabel('Demand')
    plt.title('Australian Beer Production')
    plt.legend()
    plt.show()








