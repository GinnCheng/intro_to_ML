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
    ## check if the three arrays are in the same length
    if (len(predictions)-len(ages))*(len(ages)-len(net_worths)) != 0:
        print('Error: the data lengths are not consistent')
        return None
    else:
        ## get the number of points that we need to remove
        num_clean = int(np.round(len(predictions),0))
        ## get the index of the errors in a descending order
        error_indx = np.argsort((net_worths - predictions)**2, ascend=True)[::-1]
        ## get the first num_clean of indices to be removed
        clean_indx = error_indx[:num_clean]
        ## clean the three arrays
        cl_pred, cl_ages, cl_net = (np.delete(predictions, clean_indx),
                                    np.delete(ages, clean_indx),
                                    np.delete(net_worths, clean_indx))

        cleaned_data = zip(cl_pred, cl_ages, cl_net)
    
    return cleaned_data

