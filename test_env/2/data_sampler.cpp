/*
 * Data Sampler for Bayesian Networks
 *
 * This program samples data from a given Bayesian network ("gold" standard)
 * and generates a single dataset file.
 *
 * It includes the Bayesian network library functions from "startup_code.cpp".
 */

// Define BN_LIB to prevent the main() in startup_code.cpp from being compiled
#define BN_LIB
#include "startup_code.cpp" 

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <random>
#include <fstream>
#include <iomanip>
#include <chrono> // For seeding the random number generator

using namespace std;

/**
 * @brief Performs ancestral sampling to generate one complete data record.
 * * @param bn The Bayesian network (assumed to be topologically sorted).
 * @param name_to_index A map from node names to their integer indices.
 * @param rng A Mersenne Twister random number generator.
 * @return A vector<string> representing a single complete data record.
 */
vector<string> sample_record(const network& bn, 
                             const map<string, int>& name_to_index, 
                             std::mt19937& rng) {
    
    int n_nodes = bn.netSize();
    vector<string> record(n_nodes);
    map<int, int> sampled_value_indices; // Stores {node_index -> sampled_value_index}
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    // Iterate through nodes in topological order (assuming nodes in bn are sorted)
    for (int i = 0; i < n_nodes; ++i) {
        auto node_it = bn.getNodeConst(i);
        const auto& cpt = node_it->get_CPT();
        int nvalues = node_it->get_nvalues();

        // 1. Get the parent configuration index based on already sampled values
        //    We use the sampled_value_indices map which stores parent_index -> parent_value_index
        long long parent_config_index = get_parent_config_index(*node_it, bn, name_to_index, sampled_value_indices);

        // 2. Find the start of the CPT row for this parent configuration
        long long cpt_row_start_index = parent_config_index * nvalues;

        // 3. Sample from the discrete distribution defined by this CPT row
        float r = dist(rng);
        float cumulative_prob = 0.0f;
        int sampled_index = -1;

        for (int k = 0; k < nvalues; ++k) {
            float prob = 0.0f;
            if (cpt_row_start_index + k < cpt.size()) {
                prob = cpt[cpt_row_start_index + k];
            } else {
                // Fallback in case CPT is malformed (should not happen)
                prob = 1.0f / nvalues;
            }
            
            cumulative_prob += prob;
            if (r < cumulative_prob) {
                sampled_index = k;
                break;
            }
        }

        // Handle floating point inaccuracies (if r is ~1.0)
        if (sampled_index == -1) {
            sampled_index = nvalues - 1;
        }

        // 4. Store the sampled value (both its index and string representation)
        sampled_value_indices[i] = sampled_index;
        record[i] = node_it->get_values()[sampled_index];
    }

    return record;
}

/**
 * @brief Writes a single data record to an output file stream.
 * * @param out The output file stream.
 * @param record The data record (vector<string>) to write.
 */
void write_record(ofstream& out, const vector<string>& record) {
    for (size_t i = 0; i < record.size(); ++i) {
        out << "\"" << record[i] << "\"";
        if (i < record.size() - 1) {
            out << ",";
        }
    }
    out << endl;
}

/**
 * @brief Main function to drive the data sampling.
 * * Usage: ./data_sampler <gold_network.bif> <N_total_samples> <M_incomplete_samples> <output_prefix>
 * Example: ./data_sampler gold_hailfinder.bif 1000 800 prefix
 * This generates:
 * - prefix_records.dat (with 1000 total entries, 800 of which have one '?')
 */
int main(int argc, char* argv[]) {
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <gold_network.bif> <N_total_samples> <M_incomplete_samples> <output_prefix>" << endl;
        cerr << "Example: " << argv[0] << " gold_hailfinder.bif 1000 800 prefix" << endl;
        return 1;
    }

    // --- 1. Parse Arguments ---
    string gold_file = argv[1];
    int N_total_samples = 0;
    int M_incomplete_samples = 0;
    string output_prefix = argv[4];

    try {
        N_total_samples = stoi(argv[2]);
        M_incomplete_samples = stoi(argv[3]);
    } catch (const std::exception& e) {
        cerr << "Error: Invalid N_total_samples or M_incomplete_samples. Must be integers." << endl;
        return 1;
    }

    if (N_total_samples <= 0 || M_incomplete_samples < 0) {
        cerr << "Error: N_total_samples must be > 0, M_incomplete_samples must be >= 0." << endl;
        return 1;
    }
    if (M_incomplete_samples > N_total_samples) {
        cerr << "Error: M_incomplete_samples (" << M_incomplete_samples 
             << ") cannot be larger than N_total_samples (" << N_total_samples << ")." << endl;
        return 1;
    }

    cout << "=== Data Sampler ===" << endl;
    cout << "Loading gold network from: " << gold_file << endl;

    // --- 2. Load Network and Init RNG ---
    network bn = read_network(gold_file);
    if (bn.netSize() == 0) {
        cerr << "Error: Could not read network file or network is empty." << endl;
        return 1;
    }
    cout << "Network loaded successfully. Nodes: " << bn.netSize() << endl;

    // Build Name-to-Index Map
    map<string, int> name_to_index;
    for (int i = 0; i < bn.netSize(); ++i) {
        name_to_index[bn.getNode(i)->get_name()] = i;
    }

    // Seed the random number generator
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 rng(seed);

    // Distribution for picking which variable to hide
    std::uniform_int_distribution<int> var_dist(0, bn.netSize() - 1);

    cout << "Starting sampling..." << endl;
    
    // --- 3. Run Sampling Loop ---
    string out_filename = output_prefix + "_records.dat";
    ofstream out_file(out_filename);
    if (!out_file.is_open()) {
        cerr << "Error: Could not open output file " << out_filename << endl;
        return 1;
    }

    cout << "Generating " << N_total_samples << " total samples ("
         << M_incomplete_samples << " incomplete, "
         << (N_total_samples - M_incomplete_samples) << " complete) to "
         << out_filename << "..." << endl;

    for (int j = 0; j < N_total_samples; ++j) {
        // 1. Generate a complete record
        vector<string> record = sample_record(bn, name_to_index, rng);

        // 2. If j < M, pick one variable to hide
        if (j < M_incomplete_samples) {
            int missing_idx = var_dist(rng);
            record[missing_idx] = "?";
        }
        // else (j >= M_incomplete_samples), we leave the record complete.

        // 3. Write to file
        write_record(out_file, record);
    }
    
    out_file.close();
    cout << "  Successfully generated " << out_filename << endl;

    cout << "Sampling complete." << endl;
    return 0;
}

