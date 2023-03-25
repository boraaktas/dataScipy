import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
import scipy.stats as stats
from pandas import DataFrame
import pandas as pd
from sklearn import preprocessing
import statsmodels.api as sm

##########################################################################################
####################################### SOME VALUES ######################################
##########################################################################################

def calculate_pvalue(residuals):
    """
    Calculate the p-value for the residuals.
    :param residuals: the residuals (list)
    :return: the p-value (float)

    :Example:
    >>> residuals = list of residuals
    >>> p_value = calculate_pvalue(residuals)
    """
    t_dist = abs(np.mean(residuals)) / (np.std(residuals, ddof=1) / np.sqrt(len(residuals)))
    p_value = (1 - t.cdf(t_dist, len(residuals) - 1, loc=0, scale=1)) * 2
    
    return p_value

##########################################################################################
##########################################################################################
##########################################################################################


##########################################################################################
###################################### ERROR METRICS #####################################
##########################################################################################

def make_length_equal_to_compare(realVals, forecastVals):
    """
    Make the length of the real and forecast values equal.
    :param realVals: the real values (list)
    :param forecastVals: the forecast values (list)
    :return: the real and forecast values with equal length (list, list)    

    :Example:
    >>> data = list of data values
    >>> forecasts = list of forecasts
    >>> realVals, forecastVals = make_length_equal_to_compare(data, forecasts)
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

    :Example:
    >>> data = list of data values
    >>> forecasts = list of forecasts
    >>> error = MSE(data, forecasts)
    """
    realVals, forecastVals = make_length_equal_to_compare(realVals, forecastVals)

    return np.mean((np.array(realVals) - np.array(forecastVals))**2)



def RMSE(realVals, forecastVals):
    """
    Compute the root mean squared error between the real and forecast values.
    :param realVals: the real values (list)
    :param forecastVals: the forecast values (list)
    :return: the root mean squared error (float)

    :Example:
    >>> data = list of data values
    >>> forecasts = list of forecasts
    >>> error = RMSE(data, forecasts)
    """ 
    realVals, forecastVals = make_length_equal_to_compare(realVals, forecastVals)

    return np.sqrt(MSE(realVals, forecastVals))



def MAE(realVals, forecastVals):
    """
    Compute the mean absolute error between the real and forecast values.
    :param realVals: the real values (list)
    :param forecastVals: the forecast values (list)
    :return: the mean absolute error (float)

    :Example:
    >>> data = list of data values
    >>> forecasts = list of forecasts
    >>> error = MAE(data, forecasts)
    """
    realVals, forecastVals = make_length_equal_to_compare(realVals, forecastVals)

    return np.mean(np.abs(np.array(realVals) - np.array(forecastVals)))



def MAPE(realVals, forecastVals):
    """
    Compute the mean absolute percentage error between the real and forecast values.
    :param realVals: the real values (list)
    :param forecastVals: the forecast values (list)
    :return: the mean absolute percentage error (float)

    :Example:
    >>> data = list of data values
    >>> forecasts = list of forecasts
    >>> error = MAPE(data, forecasts)
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

    :Example:
    >>> data = list of data values
    >>> forecasts = list of forecasts
    >>> error = calculate_Error(MSE, data, forecasts)
    """
    realVals, forecastVals = make_length_equal_to_compare(realVals, forecastVals)
    
    return error_method(realVals, forecastVals)



def get_all_residuals(realVals, forecastVals):
    """
    Get the errors for the forecasts.
    :param realVals: the real values (list)
    :param forecastVals: the forecast values (list)
    :return: the errors (list)

    :Example:
    >>> data = list of data values
    >>> forecasts = list of forecasts
    >>> residuals = get_all_residuals(data, forecasts)
    """
    realVals, forecastVals = make_length_equal_to_compare(realVals, forecastVals)

    return list(np.array(realVals) - np.array(forecastVals))



def print_error_summary(data, forecasts, **error_method):
    """
    Print the error summary for the forecasts.
    :param data: the time series
    :param forecasts: the forecasts
    :param error_method: the error methods to use (function)

    :Example:
    >>> data = list of data values
    >>> forecasts = list of forecasts
    >>> print_error_summary(data, forecasts, MSE=MSE, RMSE=RMSE, MAE=MAE, MAPE=MAPE)
    """

    print('Error Summary')
    print('-------------')
    for method, error in error_method.items():
        print(f'{method}: {error(data, forecasts):.4f}')



def print_resids_summary(residuals):
    """
    Print the summary of the residuals.
    :param residuals: the residuals (list)
    """
    residuals_mean = np.mean(residuals)
    residuals_sd = np.std(residuals, ddof=1)
    no_residuals = len(residuals)

    p_value = calculate_pvalue(residuals)
    width = (residuals_sd / np.sqrt(no_residuals)) * t.ppf(0.975, no_residuals - 1, loc=0, scale=1)
    
    print(f'Mean of Residual:   {residuals_mean:.4f}')
    print(f'S.D. of Residual:   {residuals_sd:.4f}')
    print(f'Half Width :        {width:.4f}   (degree of freedom = {no_residuals - 1}, Confidence Level = 0.95)')
    print(f'p-value :           {p_value:.4f}')



##########################################################################################
##########################################################################################
##########################################################################################


##########################################################################################
################################ PLOTTING RESIDUALS (ERRORS) #############################
##########################################################################################


def plot_resids(residuals):
    """
    Plot the residuals.
    :param residuals: the residuals (list)
    """

    residuals_mean = np.mean(residuals)
    no_residuals = len(residuals)
    mean_array = [residuals_mean] * no_residuals

    plt.figure(figsize=(8, 4))
    plt.plot(residuals, label='Residual', color='b')
    plt.plot(mean_array, label='Mean', linestyle='--', color='r')
    plt.title("Residuals for the Forecast", loc = 'center')
    plt.xlabel("Time")
    plt.ylabel("Residuals")
    plt.legend()
    plt.show()



def plot_normalized_resids(residuals):
    """
    Plot the normalized residuals.
    :param residuals: the residuals (list)
    """

    residual_array = np.array(residuals)

    mean = residual_array.mean()
    std = residual_array.std()

    normalized_resids = DataFrame((residual_array - np.array(mean)) / std)

    plt.figure(figsize=(6, 4))
    plt.plot(normalized_resids, label='Normalized Residual', color='b')
    plt.title("Normalized Residuals for the Forecast", loc = 'center')
    plt.xlabel("Time")
    plt.ylabel("Normalized Residuals")
    plt.legend()
    plt.show()



def plot_histogram_of_normalized_resids(residuals):
    """
    Plot the histogram of the normalized residuals.
    :param residuals: the residuals (list)
    """

    residual_array = np.array(residuals)
    
    mean = residual_array.mean()
    std = residual_array.std()

    normalized_resids = DataFrame((residual_array - np.array(mean)) / std)

    fig = plt.figure(figsize=(6, 4))
    normalized_resids.plot(kind='hist', density=True, color='b', ec='w', ax=fig.gca(), legend=False)
    normalized_resids.plot(kind='kde', color='r', ax=fig.gca(), legend=False)
    plt.title("Histogram Plus Estimated Density", loc = 'center')
    plt.ylabel("Density")
    plt.xlabel("Residuals")
    plt.show()



def plot_normal_of_normalized_resids(residuals):
    """
    Plot the normal Q-Q plot of the normalized residuals.
    :param residuals: the residuals (list)
    """

    residual_array = np.array(residuals)

    rng = (0, 1)
    scaler = preprocessing.MinMaxScaler(feature_range=(rng[0], rng[1]))
    normed = scaler.fit_transform(residual_array.reshape(-1, 1)) 
    residuals_norm = [round(i[0], 2) for i in normed]

    plt.figure(figsize=(6, 4))
    stats.probplot(residuals_norm, dist="norm", plot=plt)
    plt.ylabel("Residuals")
    plt.title("Normal Q-Q plot", loc='center')
    plt.show()



def show_all_normalized_resids_plots(residuals):
    """
    Show all the normalized residuals plots.
    :param residuals: the residuals (list)
    """

    # creating dataframe and then standardizing and normalizing
    residual_array = np.array(residuals)

    mean = residual_array.mean()
    std = residual_array.std()

    normalized_resids = DataFrame((residual_array - np.array(mean)) / std)

    rng = (0, 1)
    scaler = preprocessing.MinMaxScaler(feature_range=(rng[0], rng[1]))
    normed = scaler.fit_transform(residual_array.reshape(-1, 1))
    residuals_norm = [round(i[0], 2) for i in normed]

    
    # designing a 2 by 2 plot 
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.set_figheight(11)
    fig.set_figwidth(15)

    # top left subplot
    ax1.plot(normalized_resids, color='b')
    ax1.set_title("Normalized Residuals", loc = 'center')
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Residuals")

    # top right subplot
    ax2.set_xlim(-4, 4)
    ax2.set_title("Histogram Plus Estimated Density", loc='center')
    normalized_resids.plot(kind='hist', density=True, color='b', ec='w', ax=ax2)
    normalized_resids.plot(kind='kde', ax=ax2, color='r') 
    ax2.set_xlabel("Residuals")
    ax2.get_legend().remove()
    
    # bottom left subplot
    stats.probplot(residuals_norm, dist="norm", plot=ax3)
    ax3.set_title("Normal Q-Q plot", loc='center')
    ax3.set_ylabel("Residuals")
    
    
    # bottom right subplot
    max_lag = len(residuals)
    if max_lag > 30:
        max_lag = 30

    sm.graphics.tsa.plot_acf(normalized_resids, color='b', ax=ax4, lags=np.arange(1, max_lag + 1))
    ax4.set_title("Autocorrelation", loc='center')
    ax4.set_ylabel("Correlations")
    ax4.set_xlabel("Lags")

    plt.show()


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

    :Example:
    >>> naive_forecast([1, 2, 3, 4, 5])
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

    :Example:
    >>> seasonal_forecast([1, 2, 3, 4, 5], 3)
    >>> seasonal_forecast([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5)
    """
    if len(data) < k:
        return None
    
    return data[-k]



def MA_forecast(data, N, step_ahead=1):
    """
    Forecast the next value in a time series using a moving average.
    :param data: the time series (list)
    :param n: the number of previous values to use in the moving average (integer)
    :param step_ahead: the number of steps ahead to forecast (integer)
    :return: the forecast (float or integer or None)

    :Example:
    >>> MA_forecast([1, 2, 3, 4, 5], 3)
    >>> MA_forecast([1, 2, 3, 4, 5], 3, 2)
    """

    if step_ahead < 1:
        raise Exception('step_ahead must be greater than 1')
    
    elif (type(step_ahead) != int):
        raise Exception('step_ahead must be an integer')

    elif len(data) < N + (step_ahead - 1):
        return None

    elif step_ahead != 1: 
        return np.mean((data[:-(step_ahead - 1)])[-(N):])

    else:
        return np.mean(data[-(N):])



def ES_forecast(data, alpha, step_ahead=1):
    """
    Forecast the next value in a time series using an exponential smoothing.
    :param data: the time series (list)
    :param alpha: the smoothing parameter (float)
    :param step_ahead: the number of steps ahead to forecast (integer)
    :return: the forecast (float or integer or None)

    :Example:
    >>> ES_forecast([1, 2, 3, 4, 5], 0.5, 1)
    >>> ES_forecast([1, 2, 3, 4, 5], 0.5, 2)
    """

    if (step_ahead < 1):
        raise Exception('step_ahead must be greater than 1')
    
    elif (type(step_ahead) != int):
        raise Exception('step_ahead must be an integer')

    elif len(data) == 0:
        raise ValueError('data must not be empty')
    
    elif len(data) < step_ahead:
        return None

    elif len(data) == step_ahead:
        return data[-step_ahead]

    return alpha * data[-step_ahead] + (1 - alpha) * ES_forecast(data[:-1], alpha, step_ahead)



def make_forecast(data, forecast_method, **kwargs):
    """
    Create a forecast for the next value in the time series.
    :param data: the time series (list)
    :param forecast_method: the forecasting method to use (function)
    :param kwargs: the keyword arguments for the forecasting method (parameters for the forecasting method)
    :return: the forecast (float or integer or None)

    :Example:
    >>> data = list of data values
    >>> forecast = make_forecast(data, naive_forecast)
    >>> forecast = make_forecast(data, seasonal_forecast, k=4)
    >>> forecast = make_forecast(data, MA_forecast, N=4)
    >>> forecast = make_forecast(data, ES_forecast, alpha=0.5)
    """

    try :
        return forecast_method(data, **kwargs)
    except:
        raise Exception('Forecast method failed')
    


def make_forecast_for_all_data(data, forecast_method, **kwargs):
    """
    Create a forecast for each value in the time series.
    :param data: the time series (list)
    :param forecast_method: the forecasting method to use (function)
    :param kwargs: the keyword arguments for the forecasting method (parameters for the forecasting method)
    :return: the forecast list (list)

    :Example:
    >>> data = list of data values
    >>> forecasts = make_forecast_for_all_data(data, naive_forecast)
    >>> forecasts = make_forecast_for_all_data(data, seasonal_forecast, k=4)
    >>> forecasts = make_forecast_for_all_data(data, MA_forecast, N=4)
    """
    forecasts = [None]
    
    for i in range(len(data)):
        forecasts.append(make_forecast(data[:i+1], forecast_method, **kwargs))

    return list(forecasts)


def compare_forecasts(data, forecasts, **error_metrics):
    """
    This function takes in a list of data, a dictionary of forecasts, and
    a dictionary of error metrics. It returns a dataframe with the error
    metrics for each forecast.
    :param data: list of data values (list)
    :param forecasts: dictionary of forecasts (dictionary)
    :param error_metrics: dictionary of error metrics (dictionary)
    :return: dataframe of error metrics (pandas dataframe)

    :Example:
    >>> data = list of data values
    >>> forecasts = {
    ...     'naive': naive_forecast(data),
    ...     'seasonal': seasonal_forecast(data, k),
    ...     'MA': MA_forecast(data, N),
    ...     'ES': ES_forecast(data, alpha)
    ... }
    >>> compare_forecasts(data, forecasts, MAE=mean_absolute_error, RMSE=root_mean_squared_error)
    """

    if len(forecasts) == 0:
        raise ValueError("No forecast methods were passed to the function.")
        
    if len(error_metrics) == 0:
        raise ValueError("No error metrics were passed to the function.")
    

    results = {}

    for error_name, error_metric in error_metrics.items():
        results[error_name] = []
        for forecast in forecasts.values():
            error = error_metric(data, forecast)
            results[error_name].append(error)

    pd.options.display.float_format = '{:.2f}'.format

    df_results = pd.DataFrame(results, index=forecasts.keys())
    df_results
    
    return df_results



##########################################################################################
##########################################################################################
##########################################################################################


##########################################################################################
#################################### FORECAST PLOTTING ###################################
##########################################################################################



def plot_forecasts(data, horizon, forecasts, time_step=1):
    """
    Plot the forecasts for the time series.
    :param data: the time series (list)
    :param horizon: the time horizon (list)
    :param forecasts: the forecasts (list)
    :param time_step: the step size for the time axis (integer)

    :Example:
    >>> data = list of data values
    >>> horizon = list of time values
    >>> forecasts = list of forecast values
    >>> plot_forecasts(data, horizon, forecasts)
    >>> plot_forecasts(data, horizon, forecasts, time_step=2)
    """
    
    number_of_dots = len(horizon)

    plt.figure(figsize=(12, 3))
    plt.xticks(np.arange(min(horizon), max(horizon)+1, time_step))
    plt.grid()
    plt.plot(horizon, data[:number_of_dots], label='data')
    plt.plot(horizon, forecasts[:number_of_dots], label='forecast')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Forecast and Data')
    plt.legend()
    plt.show()



def plot_autocorrelation(values, no_lags=30):
    """
    Plot the autocorrelation of the values.
    :param residuals: the residuals (list)
    :param no_lags: the number of lags to plot (int)

    :Example:
    >>> data = list of data values
    >>> plot_autocorrelation(data , no_lags=10)
    >>> residuals = list of residuals
    >>> plot_autocorrelation(residuals)
    """
    
    residual_array = np.array(values)
    normalized_resids = residual_array - np.array(residual_array.mean()) / residual_array.std()

    max_lag = len(values)
    if max_lag > 30:
        max_lag = 31

    if no_lags < max_lag:
        max_lag = no_lags + 1

    plt.figure(figsize=(6, 4)) 
    sm.graphics.tsa.plot_acf(normalized_resids, color='b', ax=plt.gca(), lags=np.arange(1, max_lag))
    plt.title("Autocorrelation", loc='center')
    plt.ylabel("Correlations")
    plt.xlabel("Lags")
    plt.show()








