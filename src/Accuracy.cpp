/* ACCURACY.cpp
 *   by Lut99
 *
 * Created:
 *   16 Jan 2023, 17:41:23
 * Last edited:
 *   23 Jan 2023, 15:19:38
 * Auto updated?
 *   Yes
 *
 * Description:
 *   <Todo>
**/

#include <cmath>
#include <iostream>

#include "globals.h"
#include "Accuracy.h"

using namespace std;


/***** GLOBALS *****/
/* The size of the last layer, i.e., the number of classes. */
extern size_t LAST_LAYER_SIZE;





/***** HELPER MACROS *****/
#define N_LENGTH(N) \
	((N) > 0 ? (size_t) floor(log10((double) (N)) + 1.0) : 1)





/***** HELPER FUNCTIONS *****/
/* Prints the label on top of the confusion matrix.
 * 
 * # Arguments
 * - `largest_cell_size`: The size (in number of digits) of the largest value in any of the cells.
 * 
 * # Returns
 * Nothing but does write it to stdout with proper formatting.
 */
void print_confusion_matrix_top_label(size_t largest_cell_size) {
	// Compute the total width
	size_t width = 1 + (1 + LAST_LAYER_SIZE) * (3 + largest_cell_size);

	// The width of the label is in the middle
	cout << "  " << string(width / 2 - 8, ' ') << "Predicted labels" << endl;
}

/* Prints a character of the vertical label or a space if not yet.
 *
 * # Arguments
 * - `y`: The current physical row number, zero-indexed.
 * 
 * # Returns
 * Either the character of the string or a space (`' '`).
 */
char confusion_matrix_side_label(size_t y) {
	// Compute the total height
	size_t height = 1 + (2 * (1 + LAST_LAYER_SIZE));

	// Return the character if it is in the range
	const char* text = "Ground truth";
	if (y >= (height / 2 - 6) && y < (height / 2 + 6)) {
		return text[y - (height / 2 - 6)];
	} else {
		return ' ';
	}
}

/* Prints the header of the confusion matrix.
 * 
 * # Arguments
 * - `y`: The current row number (zero-indexed).
 * - `largest_cell_size`: The size (in number of digits) of the largest value in any of the cells.
 * 
 * # Returns
 * Nothing, but does write it to stdout with proper formatting.
 */
void print_confusion_matrix_header(size_t y, size_t largest_cell_size) {
	// Print the empty cell
	size_t largest_class_size = N_LENGTH(LAST_LAYER_SIZE);
	cout << ' ' << confusion_matrix_side_label(y) << "| " << std::string(largest_class_size, ' ');

	// Next, print all of the classes
	for (int i = 0; i < LAST_LAYER_SIZE; i++) {
		cout << " | " << std::string(largest_cell_size - N_LENGTH(i), ' ') << i;
	}

	// End the line
	cout << " |" << endl;
}

/* Prints a single row of the confusion matrix.
 * 
 * We assume that the matrix is LAST_LAYER_SIZE * LAST_LAYER_SIZE, implying that the given array contains at least LAST_LAYER_SIZE elements (since it's a single row).
 * 
 * # Arguments
 * - `confusion_row`: A point to a row of at least LAST_LAYER_SIZE elements.
 * - `class_number`: The class that is represented at this row (vertically).
 * - `largest_cell_size`: The size (in number of digits) of the largest value in any of the cells.
 * 
 * # Returns
 * Nothing, but does write it to stdout with proper formatting.
 */
void print_confusion_matrix_row(size_t y, int* confusion_row, int class_number, size_t largest_cell_size) {
	// Print the class first
	size_t largest_class_size = N_LENGTH(LAST_LAYER_SIZE);
	cout << ' ' << confusion_matrix_side_label(y) << "| " << std::string(largest_class_size - N_LENGTH(class_number), ' ') << class_number;

	// Next, print all of the cells
	for (int i = 0; i < LAST_LAYER_SIZE; i++) {
		int clss = confusion_row[i];
		cout << " | " << std::string(largest_cell_size - N_LENGTH(clss), ' ') << clss;
	}

	// End the line
	cout << " |" << endl;
}

/* Prints a single row of the confusion matrix, but the "line" only.
 * 
 * # Arguments
 * - `largest_cell_size`: The size (in number of digits) of the largest value in any of the cells.
 * - `left`: The character to print at the leftmost part of the line.
 * - `middle`: The character to print at each intersection of a vertical and horizontal line.
 * - `right`: The character to print the rightmost part of the line.
 * 
 * # Returns
 * Nothing, but does write it to stdout with proper formatting.
 */
void print_confusion_matrix_row_lines(size_t y, size_t largest_cell_size, const char* left, const char* middle, const char* right) {
	// Print the class cell
	size_t largest_class_size = N_LENGTH(LAST_LAYER_SIZE);
	cout << ' ' << confusion_matrix_side_label(y) << left << '-' << std::string(largest_class_size, '-');

	// Next, print all of the cells
	for (int i = 0; i < LAST_LAYER_SIZE; i++) {
		cout << '-' << middle << '-' << std::string(largest_cell_size, '-');
	}

	// End the line
	cout << '-' << right << endl;
}





/***** LIBRARY *****/
void printMetrics(const vector<float>& actual, const vector<float>& predicted) {
	// Sanity check that the prediction is what we're looking for
	assert(actual.size() == predicted.size());

	// Now we can collect the metrics by comparing the results
	std::vector<int> confusion_matrix(LAST_LAYER_SIZE * LAST_LAYER_SIZE, 0);

	// Start computing the accuracy and stuff
	double accuracy = 0.0;
	double total    = 0.0;
	for (size_t i = 0; i < actual.size(); i += LAST_LAYER_SIZE) {
		// Compress the string of output layer outputs to a single number (the class)
		int act = -1;
		for (size_t j = 0; j < LAST_LAYER_SIZE; j++) {
			if ((act < 0 && actual[i + j] > 0) || actual[i + j] > actual[i + act]) {
				act = j;
			}
		}
		if (act < 0) { cerr << "FATAL: No class in golden truth sample " << (i / LAST_LAYER_SIZE) << endl; cout << " > Sample:"; for (size_t j = 0; j < LAST_LAYER_SIZE; j++) { cout << ' ' << actual[i + j]; }; cout << endl; exit(1); }
		int pred = -1;
		for (size_t j = 0; j < LAST_LAYER_SIZE; j++) {
			if ((pred < 0 && predicted[i + j] > 0) || predicted[i + j] > predicted[i + pred]) {
				pred = j;
			}
		}
		if (pred < 0) { cerr << "FATAL: No class in predicted sample " << (i / LAST_LAYER_SIZE) << endl; cout << " > Sample:"; for (size_t j = 0; j < LAST_LAYER_SIZE; j++) { cout << ' ' << predicted[i + j]; }; cout << endl; exit(1); }

		// Populate the confusion matrix accordingly
		confusion_matrix[act * LAST_LAYER_SIZE + pred] += 1;

		// Keep track for the global accuracy
		if (act == pred) { accuracy += 1; }
		total += 1;
	}
	accuracy /= total;

	// Print 'em
	size_t longest_len = 1;
	for (size_t i = 0; i < confusion_matrix.size(); i++) {
		size_t len = N_LENGTH(confusion_matrix[i]);
		if (len > longest_len) {
			longest_len = len;
		}
	}
	cout << "----------------------------------------------" << endl;
	cout << "Confusion matrix:" << endl;
	// Print the top lines
	print_confusion_matrix_top_label(longest_len);
	print_confusion_matrix_row_lines(0, longest_len, "┌", "┬", "┐");
	// Print the header
	print_confusion_matrix_header(1, longest_len);
	// Print the rows themselves
	for (size_t y = 0; y < LAST_LAYER_SIZE; y++) {
		print_confusion_matrix_row_lines(2 + 2 * y, longest_len, "├", "┼", "┤");
		print_confusion_matrix_row(2 + 2 * y + 1, confusion_matrix.data() + (y * LAST_LAYER_SIZE), y, longest_len);
	}
	// Print the bottom lines
	print_confusion_matrix_row_lines(2 + (2 * LAST_LAYER_SIZE), longest_len, "└", "┴", "┘");
	cout << "----------------------------------------------" << endl;

	// Print the global metrics
	cout << "Global" << endl;
	cout << " - Accuracy : " << (accuracy * 100.0) << '%' << endl;
	cout << "----------------------------------------------" << endl;

	// Then print the metrics per class
	for (size_t clss = 0; clss < LAST_LAYER_SIZE; clss++) {
		// Compute the TP, TN, FP and FN for this class
		int tp = 0, tn = 0, fp = 0, fn = 0;
		for (size_t y = 0; y < LAST_LAYER_SIZE; y++) {
			for (size_t x = 0; x < LAST_LAYER_SIZE; x++) {
				int n = confusion_matrix[y * LAST_LAYER_SIZE + x];

				// True positives is a single cell, namely all those predicted as this class that were this class (easy)
				if (x == clss && y == clss) { tp += n; }
				// True negatives are all the cells that do not have anything to do as this class (i.e., were not this class and not predicted as such)
				if (x != clss && y != clss) { tn += n; }
				// False positives are all the cells predicted as this class that were not in fact this class
				if (x == clss && y != clss) { fp += n; }
				// Finally, false negatives are all the cells predicted as not this class that in fact were
				if (x != clss && y == clss) { fn += n; }
			}
		}

		// Compute the metrics from this matrix
		double accuracy  = ((double) (tp + tn)) / ((double) (tp + tn + fp + fn));
		double precision = ((double) tp) / ((double) (tp + fp));
		double recall    = ((double) tp) / ((double) (tp + fn));
		double f1        = precision * recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

		// Print them
		cout << "Class " << clss << endl;
		cout << " - Accuracy  : " << (accuracy * 100.0) << '%' << endl;
		cout << " - Precision : " << precision << endl;
		cout << " - Recall    : " << recall << endl;
		cout << " - F1        : " << f1 << endl;
		cout << "----------------------------------------------" << endl;
	}

	// Uncomment to see what we're actually comparing!
	cout << "Predicted labels (raw):";
	for (size_t i = 0; i < (predicted.size() < (10 * LAST_LAYER_SIZE) ? predicted.size() : (10 * LAST_LAYER_SIZE)); i++) {
		cout << ' ' << ((int) predicted[i]);
		if (i % 10 == 9) { cout << "   "; }
	}
	cout << endl;
	cout << "Ground truth (raw):";
	for (size_t i = 0; i < (actual.size() < (10 * LAST_LAYER_SIZE) ? actual.size() : (10 * LAST_LAYER_SIZE)); i++) {
		cout << ' ' << ((int) actual[i]);
		if (i % 10 == 9) { cout << "   "; }
	}
	cout << endl;
	cout << "Predicted labels (class):";
	for (size_t i = 0; i < (predicted.size() < (10 * LAST_LAYER_SIZE) ? predicted.size() : (10 * LAST_LAYER_SIZE)); i += LAST_LAYER_SIZE) {
		// Compress the string of nodes to a single number (the class)
		int clss = -1;
		for (size_t j = 0; j < LAST_LAYER_SIZE; j++) {
			if (predicted[i + j] > 0.5) {
				clss = j;
			}
		}
		cout << ' ' << clss;
	}
	cout << endl;
	cout << "Ground truth (class):";
	for (size_t i = 0; i < (actual.size() < (10 * LAST_LAYER_SIZE) ? actual.size() : (10 * LAST_LAYER_SIZE)); i += LAST_LAYER_SIZE) {
		// Compress the string of nodes to a single number (the class)
		int clss = -1;
		for (size_t j = 0; j < LAST_LAYER_SIZE; j++) {
			if (actual[i + j] > 0.5) {
				clss = j;
			}
		}
		cout << ' ' << clss;
	}
	cout << endl;

	// Done
}
