#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    ### your code goes here
    import numpy as np
    ## squeeze the array
    predictions,ages,net_worths=np.squeeze(predictions),np.squeeze(ages),np.squeeze(net_worths)
    ## check if the three arrays are in the same length
    if (len(predictions)-len(ages))*(len(ages)-len(net_worths)) != 0:
        print('Error: the data lengths are not consistent')
        return None
    else:
        ## get the number of points that we need to remove
        num_clean = int(np.round(0.1*len(predictions),0))
        ## get the index of the errors in a descending order
        errors = (net_worths - predictions)**2
        error_indx = np.argsort(errors)[::-1]
        ## get the first num_clean of indices to be removed
        clean_indx = error_indx[:num_clean]
        # print(f'sorted errors is {errors[clean_indx]}')
        ## clean the three arrays
        cl_ages, cl_net, cl_error = (np.delete(ages, clean_indx),
                                    np.delete(net_worths, clean_indx),
                                     np.delete(errors, clean_indx))

        cleaned_data = list(zip(cl_ages, cl_net, cl_error))
    
    return cleaned_data

