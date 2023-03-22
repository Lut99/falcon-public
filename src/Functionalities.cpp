/* FUNCTIONALITIES.cpp
 *   by Lut99
 *
 * Created:
 *   22 Mar 2023, 17:59:27
 * Last edited:
 *   22 Mar 2023, 18:17:31
 * Auto updated?
 *   Yes
 *
 * Description:
 *   Custom implementation of functionalities that just does it easy-peasy
 *   local style, to see if everything works.
**/

#include <cstddef>

#include "tools.h"

#include "Functionalities.h"

using namespace std;


/***** LIBRARY *****/
void funcMatMul(const std::vector<myType> &a, const std::vector<myType> &b, std::vector<myType> &c, 
					size_t rows, size_t common_dim, size_t columns,
				 	size_t transpose_a, size_t transpose_b, size_t truncation)
{
	#ifdef MM_TRACE
	cout << "funcMatMul(): a.size() == rows*common_dim? (" << a.size() << " == " << rows << "x" << common_dim << " == " << (rows*common_dim) << ')' << endl;
	#endif
	assert(a.size() == rows*common_dim && "Matrix a incorrect for Mat-Mul");
	#ifdef MM_TRACE
	cout << "funcMatMul(): b.size() == common_dim*columns? (" << b.size() << " == " << common_dim << "x" << columns << " == " << (common_dim*columns) << ')' << endl;
	#endif
	assert(b.size() == common_dim*columns && "Matrix b incorrect for Mat-Mul");
	#ifdef MM_TRACE
	cout << "funcMatMul(): c.size() == rows*columns? (" << c.size() << " == " << rows << "x" << columns << " == " << (rows*columns) << ')' << endl;
	#endif
	assert(c.size() == rows*columns && "Matrix c incorrect for Mat-Mul");

#if (LOG_DEBUG)
	cout << "Rows, Common_dim, Columns: " << rows << "x" << common_dim << "x" << columns << endl;
#endif

	for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < columns; col++) {
            // This is the coordinate in C; now sum the product of row in A, column in B
            // NOTE: We use `.at()` to index the array, which is exactly the same as using the array index `[]`, except that it errors if out-of-bounds instead of displaying undefined (and unbuggable :P) behaviour
            c.at(row * columns + col) = 0;
            for (size_t k = 0; k < common_dim; k++) {
                // Decide how to index A & B based on whether they are transposed or not
                size_t a_i, b_i;
                if (transpose_a && transpose_b) {
                    a_i = k * rows + row;
                    b_i = col * common_dim + k;
                } else if (transpose_a) {
                    a_i = k * rows + row;
                    b_i = k * columns + col;
                } else if (transpose_b) {
                    a_i = row * common_dim + k;
                    b_i = col * common_dim + k;
                } else {
                    a_i = row * common_dim + k;
                    b_i = k * columns + col;
                };

                // Compute the multiplication
                c.at(row * columns + col) += a.at(a_i) * b.at(b_i);
            }
        }
    }

    // Aaaaaand C is now the multiplication...

	// RSSVectorMyType r(final_size), rPrime(final_size);
	// // Again, unless my eyes are deceiving me, the (1<<truncation) is unused by this function
	// PrecomputeObject.getDividedShares(r, rPrime, (1<<truncation), final_size);
	// for (int i = 0; i < final_size; ++i)
	// 	temp3[i] = temp3[i] - rPrime[i].first;
	
	// funcReconstruct3out3(temp3, diffReconst, final_size, "Mat-Mul diff reconst", false);
	// if (SECURITY_TYPE.compare("Malicious") == 0)
	// 	funcCheckMaliciousMatMul(a, b, c, temp3, rows, common_dim, columns, transpose_a, transpose_b);

	// // This effectively implements element-wise diffReconst / (2 ^ truncation) with some casting magic in between to make it a floating-point division
	// // It seems that increasing the FLOAT_PRECISION decreases the size of diffReconst
	// dividePlain(diffReconst, (1 << truncation));

	// // for (int i = 0; i < 128; ++i)
	// // 	print_linear(diffReconst[i], "FLOAT");
	// // cout << endl;

	// if (partyNum == PARTY_A)
	// {
	// 	for (int i = 0; i < final_size; ++i)
	// 	{
	// 		c[i].first = r[i].first + diffReconst[i];
	// 		c[i].second = r[i].second;
	// 	}
	// }

	// if (partyNum == PARTY_B)
	// {
	// 	for (int i = 0; i < final_size; ++i)
	// 	{
	// 		c[i].first = r[i].first;
	// 		c[i].second = r[i].second;
	// 	}
	// }

	// if (partyNum == PARTY_C)
	// {
	// 	for (int i = 0; i < final_size; ++i)
	// 	{
	// 		c[i].first = r[i].first;
	// 		c[i].second = r[i].second + diffReconst[i];
	// 	}
	// }	
}