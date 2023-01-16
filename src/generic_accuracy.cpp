/* GENERIC ACCURACY.cpp
 *   by Lut99
 *
 * Created:
 *   16 Jan 2023, 16:50:30
 * Last edited:
 *   16 Jan 2023, 18:15:49
 * Auto updated?
 *   Yes
 *
 * Description:
 *   Small tool entrypoint that runs the `printAccuracy()` function on some
 *   files provided by an external tool. This allows us to run (test) the
 *   implementation on other languages, e.g., Python.
**/

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <string.h>

#include "globals.h"
#include "Accuracy.h"

using namespace std;


/***** GLOBALS *****/
/* The size of the input layer (bogey). */
size_t INPUT_SIZE = 0;
/* The number of classes, effectively. */
size_t LAST_LAYER_SIZE;
/* Bogey number of classes (since we don't care about) the NN */
size_t NUM_LAYERS = 0;





/***** HELPER FUNCTIONS *****/
/* Prints the help menu for this tool.
 * 
 * # Arguments
 * - `exec_name`: The name of the executable, i.e., argument 0.
 * 
 * # Returns
 * Nothing directly, but does print the help menu to stdout.
 */
void print_help(const char* exec_name) {
    cout << "Usage: " << exec_name << " [<options>] <GROUND_TRUTH_FILE> <PREDICTION_FILE>" << endl;
    cout << endl;
    cout << "This executable will compute a confusion matrix and other accuracy-related metrics for a given set" << endl;
    cout << "of ground truths and a given set of predicted values." << endl;
    cout << "Both files should simply be a list of numerical values (in ASCII), delimited by spaces. The shortest" << endl;
    cout << "list will be used to determine the length of the input." << endl;
    cout << "The labels are expected to be given as a series of N values, where N is the number of classes (see" << endl;
    cout << "'--classes'). The predicted class is then the index of the only non-zero value in that set of values." << endl;
    cout << endl;
    cout << "Positional:" << endl;
    cout << "  <GROUND_TRUTH_FILE>    Path to the file that contains the ground truth labels." << endl;
    cout << "  <PREDICTION_FILE>      Path to the file that contains the predicted labels." << endl;
    cout << endl;
    cout << "Options:" << endl;
    cout << "  -c,--classes <N>       The number of classes. Default: 10." << endl;
    cout << endl;
}





/***** ENTRYPOINT *****/
int main(int argc, const char** argv) {
    /*** CLI ***/
    // Define the CLI options we are interested in
    string ground_truth_path;
    string predicted_path;
    size_t n_classes = 10;

    // Read the arguments to find them
    bool errored    = false;
    bool allow_opts = true;
    size_t pos_i    = 0;
    size_t state    = 0;
    for (int i = 1; i < argc; i++) {
        // Get the string value of the argument
        string arg = argv[i];

        // Skip empty arguments
        if (arg.size() == 0) { continue; }

        // Match on the state
        if (state == 0) {
            // Match on whether we see an option
            if (allow_opts && arg[0] == '-') {
                // It's an option
                if (arg == "-c" || arg == "--classes") {
                    // Parse the next argument as this value
                    state = 1;

                } else if (arg == "-h" || arg == "--help") {
                    // Print the help menu
                    print_help(argv[0]);
                    return 0;

                } else if (arg == "--") {
                    // Disallow options from now on
                    allow_opts = false;

                } else {
                    cerr << "Unknown option '" << arg << "'" << endl;
                    errored = true;
                }

            } else {
                // It's a positional argument
                if (pos_i == 0) {
                    ground_truth_path = arg;

                } else if (pos_i == 1) {
                    predicted_path = arg;

                } else {
                    cerr << "Unknown positional '" << arg << '\'' << endl;
                    errored = true;
                }

                // Increment the index
                pos_i++;

            }

        } else if (state <= 1) {
            // Make sure it isn't an option
            if (allow_opts && arg[0] == '-') {
                // Escape: we do allow '--'
                if (arg == "--") { allow_opts = false; continue; }
                cerr << "No value given for option" << endl;
                errored = true;
                state = 0;
                continue;
            }

            // Otherwise, set the value
            if (state == 1) {
                // Attempt to parse as an integer
                try {
                    n_classes = (size_t) stoul(arg);
                } catch (const std::invalid_argument& e) {
                    cerr << "Given value '" << arg << "' is not a valid unsigned integer: " << e.what() << endl;
                    errored = 1;
                } catch (const std::out_of_range& e) {
                    cerr << "Given value '" << arg << "' is out-of-range for an unsigned integer" << endl;
                    errored = 1;
                }
            }

            // Reset the state
            state = 0;

        } else {
            cerr << "Unknown state '" << state << '\'' << endl;
            return 1;
        }
    }

    // Assert the positional arguments are there
    if (ground_truth_path.empty() || predicted_path.empty()) {
        cerr << "Usage: " << argv[0] << " <GROUND_TRUTH_FILE> <PREDICTION_FILE>" << endl;
        cerr << endl;
        cerr << "Use '" << argv[0] << " --help' for more information." << endl;
        cerr << endl;
        errored = true;
    }

    // Quit if an error occurred
    if (errored) { return 1; }



    /*** FILE IO ***/
    // Read both input files
    cout << "Reading '" << ground_truth_path << "'..." << endl;
    vector<float> ground_truth;
    ifstream gt_handle(ground_truth_path);
    if (!gt_handle.is_open()) {
        cerr << "Failed to open ground truth file '" << ground_truth_path << "': " << strerror(errno) << endl;
        return errno;
    }
    float gt_num;
    while (gt_handle >> gt_num) {
        ground_truth.push_back(gt_num);
    }
    gt_handle.close();
    if (ground_truth.size() % n_classes != 0) {
        cerr << "Ground truth file '" << ground_truth_path << "' does not have a multiple of " << n_classes << " values, but " << ground_truth.size() << " instead (i.e., the number of classes)" << endl;
        return 1;
    }
    cout << " > Found " << (ground_truth.size() / n_classes) << " samples" << endl;

    cout << "Reading '" << predicted_path << "'..." << endl;
    vector<float> predicted;
    ifstream p_handle(predicted_path);
    if (!p_handle.is_open()) {
        cerr << "Failed to open prediction file '" << predicted_path << "': " << strerror(errno) << endl;
        return errno;
    }
    float p_num;
    while (p_handle >> p_num) {
        predicted.push_back(p_num);
    }
    p_handle.close();
    if (predicted.size() % n_classes != 0) {
        cerr << "Prediction file '" << predicted_path << "' does not have a multiple of " << n_classes << " values, but " << predicted.size() << " instead (i.e., the number of classes)" << endl;
        return 1;
    }
    cout << " > Found " << (predicted.size() / n_classes) << " samples" << endl;

    // Truncate the vectors to the smallest ones
    if (ground_truth.size() < predicted.size()) {
        predicted.resize(ground_truth.size());
    }
    if (ground_truth.size() > predicted.size()) {
        ground_truth.resize(predicted.size());
    }
    cout << "Computing for " << (ground_truth.size() / n_classes) << " samples" << endl;
    cout << endl;



    /*** PRINTING ***/
    // Convert the floating-point to fixed-point
    vector<smallType> ground_truth_fixed(ground_truth.size());
    for (size_t i = 0; i < ground_truth.size(); i++) {
        ground_truth_fixed[i] = (smallType) (((float) ground_truth[i]) / ((float) (1 << FLOAT_PRECISION)));
        if (i > 0) { cout << " "; }
        cout << ground_truth[i] << "/" << ((int) ground_truth_fixed[i]);
    }
    cout << endl;
    vector<smallType> predicted_fixed(predicted.size());
    for (size_t i = 0; i < predicted.size(); i++) {
        predicted_fixed[i] = (smallType) (((float) predicted[i]) / ((float) (1 << FLOAT_PRECISION)));
        // if (i > 0) { cout << " "; }
        // cout << predicted[i] << "/" << ((int) predicted_fixed[i]);
    }
    cout << endl;

    // Simply run the secondary file
    LAST_LAYER_SIZE = n_classes;
    printMetrics(ground_truth_fixed, predicted_fixed);



    // Done
    return 0;
}
