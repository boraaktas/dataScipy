import numpy as np


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