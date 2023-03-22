/* FUNCTIONALITIES.h
 *   by Lut99
 *
 * Created:
 *   22 Mar 2023, 17:57:43
 * Last edited:
 *   22 Mar 2023, 18:05:33
 * Auto updated?
 *   Yes
 *
 * Description:
 *   Custom implementation of functionalities that just does it easy-peasy
 *   local style, to see if everything works.
**/

#pragma once
#include <vector>
#include "globals.h"

/***** LIBRARY *****/
/* Performs matrix multiplication the simple(TM) way.
 *
 * Not very fast, but at least it'll work (hopefully). */
void funcMatMul(const std::vector<myType> &a, const std::vector<myType> &b, std::vector<myType> &c, 
				size_t rows, size_t common_dim, size_t columns,
			 	size_t transpose_a, size_t transpose_b, size_t truncation);
