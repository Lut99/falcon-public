/* ACCURACY.h
 *   by Lut99
 *
 * Created:
 *   16 Jan 2023, 17:39:55
 * Last edited:
 *   16 Jan 2023, 17:40:45
 * Auto updated?
 *   Yes
 *
 * Description:
 *   Defines functions for working with accuracies.
**/

#ifndef ACCURACY_H
#define ACCURACY_H

#include <vector>

#include "globals.h"


/* Prints the confusion matrix and other metrics for the given ground truth and predicted values.
 * 
 * # Arguments
 * - `actual`: The actual labels of the samples.
 * - `predicted`: The predicted labels of the samples.
 * 
 * # Returns
 * Nothing directly, but does populate `stdout` with tables and other results.
 */
void printMetrics(const std::vector<smallType>& actual, const std::vector<smallType>& predicted);

#endif
