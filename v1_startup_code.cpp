#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <map>
#include <iomanip>
#include <cmath>
#include <algorithm> 
#include <random>       
#include <numeric>      
#include <limits>       
#include <chrono>       
#include <utility>      

using namespace std;

// Forward Declarations
class network; 
class Graph_Node;
network read_network(const string& filename);
void write_network(const network& BayesNet, const string& filename);
int get_value_index(const Graph_Node& node, const string& value);
vector<vector<string>> read_data(const string& filename);

// --- NEW FUNCTION DECLARATIONS (k-NN + MLE) ---
long long get_parent_config_index(const Graph_Node& node, const network& bn, 
                                  const map<string, int>& name_to_index, 
                                  const map<int, int>& parent_val_indices);
long long get_cpt_index(const Graph_Node& node, const network& bn, 
                        const map<string, int>& name_to_index, 
                        int child_val_idx, const map<int, int>& parent_val_indices);
void learn_cpts_mle(network& bn, const vector<vector<string>>& dataset, 
                    const map<string, int>& name_to_index, float pseudo_count);
string find_knn_prediction(const vector<string>& incomplete_row, 
                           const vector<vector<string>>& complete_rows,
                           int missing_index, int k);
vector<vector<string>> create_complete_dataset(const vector<vector<string>>& raw_data, int k);


// ==========================================================================================
// BAYESIAN NETWORK CLASS DEFINITIONS
// ==========================================================================================

class Graph_Node {
private:
    string Node_Name;
    vector<int> Children;
    vector<string> Parents;
    int nvalues;
    vector<string> values;
    vector<float> CPT;
    
public:
    Graph_Node(string name, int n, vector<string> vals) {
        Node_Name = name;
        nvalues = n;
        values = vals;
    }

    string get_name() const {
        return Node_Name;
    }

    const vector<int>& get_children() const {
        return Children;
    }

    const vector<string>& get_Parents() const {
        return Parents;
    }

    const vector<float>& get_CPT() const {
        return CPT;
    }
    
    int get_nvalues() const {
        return nvalues;
    }

    const vector<string>& get_values() const {
        return values;
    }

    void set_CPT(const vector<float>& new_CPT) {
        CPT = new_CPT;
    }

    void set_Parents(const vector<string>& Parent_Nodes) {
        Parents = Parent_Nodes;
    }

    int add_child(int new_child_index) {
        for(int i = 0; i < Children.size(); i++) {
            if(Children[i] == new_child_index)
                return 0;
        }
        Children.push_back(new_child_index);
        return 1;
    }
    
    void add_parent_names(const vector<string>& pnames) {
        Parents.insert(Parents.end(), pnames.begin(), pnames.end());
    }
};

class network {
private: 
    list<Graph_Node> Pres_Graph;

public:
    int addNode(const Graph_Node& node) {
        Pres_Graph.push_back(node);
        return 0;
    }

    // --- Overload for std::move ---
    int addNode(Graph_Node&& node) {
        Pres_Graph.push_back(std::move(node));
        return 0;
    }

    list<Graph_Node>::iterator getNode(int i) {
        int count = 0;
        list<Graph_Node>::iterator listIt;
        for(listIt = Pres_Graph.begin(); listIt != Pres_Graph.end(); listIt++) {
            if(count++ == i)
                break;
        }
        return listIt;
    }

    list<Graph_Node>::const_iterator getNodeConst(int i) const {
        int count = 0;
        list<Graph_Node>::const_iterator listIt;
        for(listIt = Pres_Graph.begin(); listIt != Pres_Graph.end(); listIt++) {
            if(count++ == i)
                break;
        }
        return listIt;
    }

    list<Graph_Node>::iterator get_nth_node(int i) {
        return getNode(i);
    }

    list<Graph_Node>::const_iterator get_nth_node(int i) const {
        return getNodeConst(i);
    }

    int netSize() const { 
        return Pres_Graph.size(); 
    }

    int get_index(const string& val_name) const {
        int count = 0;
        for (const auto& node : Pres_Graph) {
            if (node.get_name() == val_name) {
                return count;
            }
            count++;
        }
        return -1;
    }

    list<Graph_Node>::iterator search_node(const string& node_name) {
        for (auto it = Pres_Graph.begin(); it != Pres_Graph.end(); ++it) {
            if (it->get_name() == node_name) {
                return it;
            }
        }
        return Pres_Graph.end();
    }
    
    list<Graph_Node>::const_iterator search_node_const(const string& node_name) const {
        for (auto it = Pres_Graph.begin(); it != Pres_Graph.end(); ++it) {
            if (it->get_name() == node_name) {
                return it;
            }
        }
        return Pres_Graph.end();
    }
    
    network() = default;
    network(const network& other) = default;
    network(network&& other) noexcept = default;
    network& operator=(const network& other) = default;
    network& operator=(network&& other) noexcept = default;
};

// ==========================================================================================
// UTILITY FUNCTIONS (Parsing, Writing)
// ==========================================================================================

string trim_whitespace(const string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (string::npos == first) {
        return str;
    }
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, (last - first + 1));
}

string clean_name(string s) {
    s = trim_whitespace(s);
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
        s = s.substr(1, s.size() - 2);
    }
    return s;
}


// ------------------------------------------------------------------------------------------
// ROBUST BIF PARSER (read_network)
// ------------------------------------------------------------------------------------------

network read_network(const string& filename) {
    network bn;
    ifstream file(filename);
    string line;
    
    map<string, vector<string>> variable_values;
    map<string, vector<string>> variable_parents;
    map<string, string> variable_cpt_blocks; // Store the raw CPT block
    vector<string> node_order; 

    if (!file.is_open()) {
        cerr << "Error: Could not open network file " << filename << endl;
        return bn;
    }

    // --- PASS 1: Read all variable definitions and probability headers ---
    while (getline(file, line)) {
        line = trim_whitespace(line);
        if (line.empty() || line.substr(0, 2) == "//") {
            continue;
        }

        // --- Parse 'variable' block ---
        if (line.substr(0, 8) == "variable") {
            string name = line.substr(8);
            name = trim_whitespace(name);
            if (!name.empty() && name.back() == '{') {
                name.pop_back();
            }
            name = clean_name(name);
            node_order.push_back(name);
            
            string block_line;
            while (getline(file, block_line)) {
                block_line = trim_whitespace(block_line);
                if (block_line.empty() || block_line.substr(0, 2) == "//") continue;
                if (block_line.substr(0, 1) == "}") { 
                    break;
                }
                if (block_line.substr(0, 13) == "type discrete") {
                    size_t start = block_line.find('{');
                    size_t end = block_line.find('}');
                    if (start != string::npos && end != string::npos) {
                        string values_str = block_line.substr(start + 1, end - start - 1);
                        stringstream vs(values_str);
                        string val;
                        while (getline(vs, val, ',')) {
                            variable_values[name].push_back(clean_name(val));
                        }
                    }
                }
            }
        } 
        // --- Parse 'probability' block ---
        else if (line.substr(0, 11) == "probability") {
            size_t child_start = line.find('(') + 1;
            size_t parents_end = line.find(')');
            if (child_start == string::npos || parents_end == string::npos) continue;

            string header = line.substr(child_start, parents_end - child_start);
            size_t child_end = header.find('|');
            
            string child_name;
            vector<string> parents;

            if (child_end != string::npos) {
                child_name = clean_name(header.substr(0, child_end));
                string parents_str = header.substr(child_end + 1);
                stringstream ps(parents_str);
                string parent_name;
                while (getline(ps, parent_name, ',')) {
                    parents.push_back(clean_name(parent_name));
                }
            } else {
                child_name = clean_name(header);
            }
            variable_parents[child_name] = parents;
            
            // Store the entire CPT block for Pass 2
            string cpt_block_content;
            string block_line;
            while (getline(file, block_line)) {
                cpt_block_content += block_line + "\n";
                if (trim_whitespace(block_line).substr(0, 1) == "}") {
                    break;
                }
            }
            variable_cpt_blocks[child_name] = cpt_block_content;
        }
    }
    file.close();

    // --- PASS 2: Construct network and parse CPT data ---
    map<string, int> node_name_to_index;
    for (int i = 0; i < node_order.size(); ++i) {
        const string& name = node_order[i];
        if (variable_values.find(name) == variable_values.end()) {
            continue;
        }
        
        const auto& vals = variable_values.at(name);
        Graph_Node node(name, vals.size(), vals);

        if (variable_parents.find(name) != variable_parents.end()) {
            node.set_Parents(variable_parents.at(name));
        }
        bn.addNode(std::move(node)); // Use move
        node_name_to_index[name] = i;
    }
    
    // --- PASS 3: Parse CPTs using the constructed network structure ---
    for (int i = 0; i < bn.netSize(); ++i) {
        auto node_it = bn.getNode(i);
        const string& child_name = node_it->get_name();
        const auto& parent_names = node_it->get_Parents();
        int num_child_values = node_it->get_nvalues();

        // Calculate total CPT size
        long long cpt_size = num_child_values;
        map<string, long long> parent_multipliers;
        map<string, map<string, int>> parent_val_indices;
        long long current_multiplier = 1;

        for (const string& pname : parent_names) {
            auto pnode_it = bn.search_node_const(pname);
            if (pnode_it->get_name() != pname) {
                 cerr << "Error: Parent node " << pname << " not found for " << child_name << endl;
                 continue;
            }
            parent_multipliers[pname] = current_multiplier;
            int n_p_values = pnode_it->get_nvalues();
            const auto& p_values = pnode_it->get_values();
            for(int p_val_idx = 0; p_val_idx < n_p_values; ++p_val_idx) {
                parent_val_indices[pname][p_values[p_val_idx]] = p_val_idx;
            }
            current_multiplier *= n_p_values;
        }
        cpt_size = current_multiplier * num_child_values;
        vector<float> final_cpt(cpt_size, -1.0f); // Initialize with -1

        // Parse the stored CPT block
        string cpt_block = variable_cpt_blocks[child_name];
        stringstream block_stream(cpt_block);
        string block_line;
        
        while (getline(block_stream, block_line)) {
            block_line = trim_whitespace(block_line);
            if (block_line.empty() || block_line.substr(0, 2) == "//" || block_line == "{" || block_line == "};") {
                continue;
            }

            vector<float> probs;
            long long base_index = 0;

            if (block_line.substr(0, 5) == "table") {
                // Root node
                base_index = 0;
                size_t probs_start = 5;
                size_t probs_end = block_line.find(';');
                if (probs_end == string::npos) probs_end = block_line.size();
                string probs_str = block_line.substr(probs_start, probs_end - probs_start);
                
                stringstream ps(probs_str);
                string prob_val;
                while (getline(ps, prob_val, ',')) {
                    string trimmed_val = trim_whitespace(prob_val);
                    if(trimmed_val == "-1") {
                        probs.push_back(-1.0f);
                    } else {
                        try { probs.push_back(stof(trimmed_val)); } catch(...) { probs.push_back(-1.0f); }
                    }
                }

            } else if (block_line.front() == '(') {
                // Conditional node
                size_t parent_vals_end = block_line.find(')');
                if (parent_vals_end == string::npos) continue;
                string parent_vals_str = block_line.substr(1, parent_vals_end - 1);
                
                vector<string> parent_values;
                stringstream pvs(parent_vals_str);
                string p_val;
                while(getline(pvs, p_val, ',')) {
                    parent_values.push_back(clean_name(p_val));
                }

                if(parent_values.size() != parent_names.size()) {
                    continue; // Mismatch, skip
                }

                // Calculate base index from parent values
                base_index = 0;
                bool parent_val_ok = true;
                for (int p_idx = 0; p_idx < parent_names.size(); ++p_idx) {
                    const string& pname = parent_names[p_idx];
                    const string& pval = parent_values[p_idx];
                    if (parent_val_indices.count(pname) == 0 || parent_val_indices[pname].count(pval) == 0) {
                        parent_val_ok = false;
                        break;
                    }
                    int p_val_idx = parent_val_indices.at(pname).at(pval);
                    base_index += (long long)p_val_idx * parent_multipliers.at(pname);
                }
                if (!parent_val_ok) continue; // Skip this row

                // Parse probabilities
                size_t probs_start = parent_vals_end + 1;
                size_t probs_end = block_line.find(';');
                if (probs_end == string::npos) probs_end = block_line.size();
                string probs_str = block_line.substr(probs_start, probs_end - probs_start);
                
                stringstream ps(probs_str);
                string prob_val;
                while (getline(ps, prob_val, ',')) {
                    string trimmed_val = trim_whitespace(prob_val);
                    if(trimmed_val == "-1") {
                        probs.push_back(-1.0f);
                    } else {
                        try { probs.push_back(stof(trimmed_val)); } catch(...) { probs.push_back(-1.0f); }
                    }
                }
            }

            // Insert probabilities into the final CPT vector at the correct index
            if (probs.size() == num_child_values) {
                long long final_cpt_base_idx = base_index * num_child_values;
                for (int k = 0; k < num_child_values; ++k) {
                    if (final_cpt_base_idx + k < final_cpt.size()) {
                        final_cpt[final_cpt_base_idx + k] = probs[k];
                    }
                }
            }
        }
        node_it->set_CPT(final_cpt);
    }
    
    // Set children (can only be done after all nodes are in the list)
    for (int i = 0; i < bn.netSize(); ++i) {
        auto node_it = bn.getNode(i);
        for (const auto& parent_name : node_it->get_Parents()) {
            if (node_name_to_index.count(parent_name)) {
                int parent_index = node_name_to_index.at(parent_name);
                auto parent_it = bn.getNode(parent_index);
                parent_it->add_child(i);
            }
        }
    }

    return bn;
}


// ------------------------------------------------------------------------------------------
// BIF NETWORK WRITER
// ------------------------------------------------------------------------------------------

void write_network(const network& BayesNet, const string& filename) {
    ofstream outfile(filename);
    if (!outfile.is_open()) {
        cerr << "Error: Could not open file for writing: " << filename << endl;
        return;
    }
    
    outfile << "// Bayesian Network (Learned via k-NN Imputation + MLE)" << endl << endl;

    for (int i = 0; i < BayesNet.netSize(); ++i) {
        auto node_it = BayesNet.getNodeConst(i); 
        outfile << "variable " << node_it->get_name() << " {" << endl;
        outfile << "  type discrete [ " << node_it->get_nvalues() << " ] = { ";
        const auto& values = node_it->get_values();
        for (int k = 0; k < node_it->get_nvalues(); ++k) {
            outfile << values[k]; 
            if (k < node_it->get_nvalues() - 1) outfile << ", ";
        }
        outfile << " };" << endl;
        outfile << "};" << endl;
    }

    for (int i = 0; i < BayesNet.netSize(); ++i) {
        auto node_it = BayesNet.getNodeConst(i);
        const auto& node_name = node_it->get_name();
        const auto& parents = node_it->get_Parents();
        const auto& CPT = node_it->get_CPT();
        const auto& values = node_it->get_values();
        
        outfile << "probability ( " << node_name;
        if (!parents.empty()) {
            outfile << " | ";
            for (int p = 0; p < parents.size(); ++p) {
                outfile << parents[p];
                if (p < parents.size() - 1) outfile << ", ";
            }
        }
        outfile << " ) {" << endl;

        if (parents.empty()) {
            outfile << "    table ";
            for (int k = 0; k < CPT.size(); ++k) {
                outfile << fixed << setprecision(6) << CPT[k];
                if (k < CPT.size() - 1) outfile << ", ";
            }
            outfile << ";" << endl;
        } else {
            vector<int> radices;
            long long table_size = 1;
            for(const auto& pname : parents) {
                auto pnode_it = BayesNet.search_node_const(pname);
                if (pnode_it->get_name() == pname) { 
                    radices.push_back(pnode_it->get_nvalues());
                    table_size *= pnode_it->get_nvalues();
                } else {
                    cerr << "Error (Write): Parent node " << pname << " not found." << endl;
                    outfile.close();
                    return;
                }
            }

            for (long long j = 0; j < table_size; ++j) {
                long long tmp = j;
                vector<int> idx(parents.size()); 

                for (int p = 0; p < (int)parents.size(); ++p) {
                    idx[p] = tmp % radices[p];
                    tmp /= radices[p];
                }

                outfile << "    ( ";
                for (int p = 0; p < (int)parents.size(); p++) {
                    auto pnode_it = BayesNet.search_node_const(parents[p]); 
                    const auto& pvals = pnode_it->get_values();
                    int vidx = idx[p];
                    outfile << pvals[vidx]; 
                    if (p < (int)parents.size() - 1) outfile << ", ";
                }
                outfile << " ) ";

                for (int k = 0; k < (int)values.size(); k++) {
                    long long current_cpt_index = j * values.size() + k;
                    if (current_cpt_index < (int)CPT.size()) {
                        outfile << fixed << setprecision(6) << CPT[current_cpt_index];
                    }
                    else outfile << "-1.000000";
                    if (k < (int)values.size() - 1) outfile << ", ";
                }
                outfile << ";" << endl;
            }
        }
        outfile << "};" << endl << endl;
    }
    outfile.close();
    cout << "Network written to file: " << filename << endl;
}

// ------------------------------------------------------------------------------------------
// 1. DATA PARSING FUNCTION
// ------------------------------------------------------------------------------------------

vector<vector<string>> read_data(const string& filename) {
    vector<vector<string>> data;
    ifstream file(filename);
    string line;
    
    if (!file.is_open()) {
        cerr << "Error: Could not open data file " << filename << endl;
        return data;
    }

    while (getline(file, line)) {
        line = trim_whitespace(line);
        if (line.empty() || line.front() == '[') continue;

        stringstream record_stream(line);
        string token;
        vector<string> record;

        while (getline(record_stream, token, ',')) {
            token = trim_whitespace(token); 
            
            if (token.size() >= 2 && token.front() == '"' && token.back() == '"') {
                token = token.substr(1, token.size() - 2);
            }
            record.push_back(token);
        }
        if (!record.empty()) {
            data.push_back(record);
        }
    }
    file.close();
    return data;
}

// ------------------------------------------------------------------------------------------
// 2. HELPER FUNCTIONS (MLE + k-NN)
// ------------------------------------------------------------------------------------------

/**
 * Helper to get the index for a specific value string of a node.
 */
int get_value_index(const Graph_Node& node, const string& value) {
    const auto& values = node.get_values();
    for (int i = 0; i < values.size(); ++i) {
        if (values[i] == value) {
            return i;
        }
    }
    return -1; // Value not found (e.g., "?")
}

/**
 * Helper to get the parent configuration index (the "row" of the CPT).
 * Assumes parent_val_indices is a map of {parent_index -> parent_value_index}.
 */
long long get_parent_config_index(const Graph_Node& node, const network& bn, 
                                  const map<string, int>& name_to_index, 
                                  const map<int, int>& parent_val_indices) {
    
    long long parent_config_index = 0;
    long long multiplier = 1;

    for(const string& pname : node.get_Parents()) {
        int p_idx = name_to_index.at(pname);
        int val_idx = 0;
        if(parent_val_indices.count(p_idx)) {
             val_idx = parent_val_indices.at(p_idx);
        } 
        
        parent_config_index += (long long)val_idx * multiplier;
        
        auto pnode_it = bn.getNodeConst(p_idx);
        multiplier *= pnode_it->get_nvalues();
    }
    return parent_config_index;
}

/**
 * Helper to get the full CPT index (row + column).
 */
long long get_cpt_index(const Graph_Node& node, const network& bn, 
                        const map<string, int>& name_to_index, 
                        int child_val_idx, const map<int, int>& parent_val_indices) {

    long long parent_config_index = get_parent_config_index(node, bn, name_to_index, parent_val_indices);
    return parent_config_index * node.get_nvalues() + child_val_idx;
}

// ------------------------------------------------------------------------------------------
// 3. k-NN IMPUTATION
// ------------------------------------------------------------------------------------------

/**
 * Calculates the Hamming distance between two records, ignoring a specific index.
 */
int hamming_distance(const vector<string>& row1, const vector<string>& row2, int ignore_index) {
    int distance = 0;
    for (int i = 0; i < row1.size(); ++i) {
        if (i == ignore_index) continue;
        if (row1[i] != row2[i]) {
            distance++;
        }
    }
    return distance;
}

/**
 * Finds the k-nearest neighbors and predicts the missing value by majority vote.
 */
string find_knn_prediction(const vector<string>& incomplete_row, 
                           const vector<vector<string>>& complete_rows,
                           int missing_index, int k) {

    // Store distances and the index of the neighbor
    vector<pair<int, int>> neighbors; // {distance, index_in_complete_rows}

    for (int i = 0; i < complete_rows.size(); ++i) {
        int dist = hamming_distance(incomplete_row, complete_rows[i], missing_index);
        neighbors.push_back({dist, i});
    }

    // Sort by distance (ascending)
    std::sort(neighbors.begin(), neighbors.end());

    // Get the top k neighbors and vote
    map<string, int> votes;
    for (int i = 0; i < k && i < neighbors.size(); ++i) {
        int neighbor_original_index = neighbors[i].second;
        const string& neighbor_value = complete_rows[neighbor_original_index][missing_index];
        votes[neighbor_value]++;
    }

    // Find the value with the most votes
    string best_value = "";
    int max_votes = -1;
    for (const auto& pair : votes) {
        if (pair.second > max_votes) {
            max_votes = pair.second;
            best_value = pair.first;
        }
    }
    
    // Fallback in case k=0 or no neighbors (shouldn't happen with reasonable k)
    if (best_value == "") { 
        return "True"; // Or any reasonable default
    }

    return best_value;
}

/**
 * Creates a 100% complete dataset by imputing missing values using k-NN.
 */
vector<vector<string>> create_complete_dataset(const vector<vector<string>>& raw_data, int k) {
    
    vector<vector<string>> complete_rows;
    vector<vector<string>> incomplete_rows;
    vector<int> incomplete_indices; // Stores the original index
    
    // --- Step 1: Partition the data ---
    for (int i = 0; i < raw_data.size(); ++i) {
        bool is_complete = true;
        for (const string& val : raw_data[i]) {
            if (val == "?") {
                is_complete = false;
                break;
            }
        }
        if (is_complete) {
            complete_rows.push_back(raw_data[i]);
        } else {
            incomplete_rows.push_back(raw_data[i]);
            incomplete_indices.push_back(i);
        }
    }
    
    cout << "  Partitioned data: " << complete_rows.size() 
         << " complete rows, " << incomplete_rows.size() 
         << " incomplete rows." << endl;

    if (complete_rows.empty()) {
        cerr << "Error: No complete rows found to train k-NN. Cannot impute." << endl;
        // As a fallback, just return the raw data (MLE will handle '?')
        return raw_data;
    }

    // --- Step 2: Impute missing values ---
    // We create a final dataset that respects the original order
    vector<vector<string>> final_imputed_set = raw_data; 
    
    for (int i = 0; i < incomplete_rows.size(); ++i) {
        const auto& row = incomplete_rows[i];
        int original_index = incomplete_indices[i];
        
        int missing_index = -1;
        for (int j = 0; j < row.size(); ++j) {
            if (row[j] == "?") {
                missing_index = j;
                break;
            }
        }
        
        if (missing_index == -1) continue; // Should not happen here

        // Find the best value using k-NN
        string predicted_value = find_knn_prediction(row, complete_rows, missing_index, k);
        
        // Fill in the '?' in our final dataset
        final_imputed_set[original_index][missing_index] = predicted_value;
    }
    
    cout << "  Imputation complete. Final dataset has " 
         << final_imputed_set.size() << " rows." << endl;
         
    return final_imputed_set;
}


// ------------------------------------------------------------------------------------------
// 4. MLE LEARNING (from a 100% complete dataset)
// ------------------------------------------------------------------------------------------

/**
 * Learns CPTs from a 100% complete dataset using simple MLE (counting).
 */
void learn_cpts_mle(network& bn, const vector<vector<string>>& dataset, 
                    const map<string, int>& name_to_index, float pseudo_count = 0.1f) {
    
    if (dataset.empty()) {
        cerr << "Error (MLE): Dataset is empty." << endl;
        return;
    }
    
    for (int i = 0; i < bn.netSize(); ++i) {
        auto node_it = bn.getNode(i); 
        const auto& node_name = node_it->get_name();
        const auto& parents = node_it->get_Parents();
        int num_child_values = node_it->get_nvalues();

        // Calculate CPT size
        long long cpt_size = num_child_values;
        for (const auto& pname : parents) {
            if (name_to_index.count(pname) == 0) {
                cerr << "Error (MLE): Parent " << pname << " for node " 
                     << node_name << " not in map." << endl;
                return;
            }
            auto pnode_it = bn.getNode(name_to_index.at(pname));
            cpt_size *= pnode_it->get_nvalues();
        }

        // Initialize counts with pseudo-count (smoothing)
        vector<float> numerators(cpt_size, pseudo_count);
        vector<float> denominators(cpt_size / num_child_values, (float)num_child_values * pseudo_count);
        
        // --- Count occurrences ---
        for (const auto& record : dataset) {
            if (record.size() != bn.netSize()) continue; // Skip malformed rows
            
            map<int, int> parent_val_indices;
            bool record_valid = true;
            
            // Get parent values
            for (const auto& pname : parents) {
                int p_idx = name_to_index.at(pname);
                auto pnode_it = bn.getNodeConst(p_idx);
                int val_idx = get_value_index(*pnode_it, record[p_idx]);
                if (val_idx == -1) { // Should not happen with imputed data
                    record_valid = false;
                    break;
                }
                parent_val_indices[p_idx] = val_idx;
            }
            if (!record_valid) continue;

            // Get child value
            int child_idx = name_to_index.at(node_name);
            int c_val_idx = get_value_index(*node_it, record[child_idx]);
            if (c_val_idx == -1) continue; // Should not happen
            
            // Get CPT indices
            long long parent_config_index = get_parent_config_index(*node_it, bn, name_to_index, parent_val_indices);
            long long cpt_index = parent_config_index * num_child_values + c_val_idx;
            
            if (cpt_index >= numerators.size() || parent_config_index >= denominators.size()) {
                 cerr << "FATAL (MLE): Index out of bounds." << endl;
                 return;
            }
            
            // Increment counts
            numerators[cpt_index]++;
            denominators[parent_config_index]++;
        }

        // --- Normalize counts to get probabilities ---
        vector<float> new_cpt(cpt_size);
        for (long long j = 0; j < denominators.size(); ++j) {
            float total_count = denominators[j];
            for (int k = 0; k < num_child_values; ++k) {
                long long cpt_index = j * num_child_values + k;
                
                if (total_count < 1e-9) { // Avoid division by zero
                    new_cpt[cpt_index] = 1.0f / num_child_values; 
                } else {
                    new_cpt[cpt_index] = numerators[cpt_index] / total_count;
                }
            }
        }
        
        node_it->set_CPT(new_cpt);
    }
}


// ==========================================================================================
// MAIN FUNCTION (k-NN + MLE Controller)
// ==========================================================================================

#ifndef BN_LIB
int main(int argc, char* argv[]) {
    
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <hailfinder_file> <data_file>" << endl;
    }

    string hailfinder_file = argv[1];
    string data_file = argv[2];

    // --- k-NN Hyperparameters ---
    const int K_NEIGHBORS = 30;      // Number of neighbors to use for voting
    const float PSEUDO_COUNT = 0.1f; // Smoothing for the final MLE
    // ----------------------------
    
    cout << "=== k-NN Imputation + MLE Learner ===" << endl;
    
    // --- Load Data and Network Structure ---
    network BayesNet = read_network(hailfinder_file);
    if (BayesNet.netSize() == 0) {
        cerr << "Fatal Error: Network structure load failed. Cannot proceed." << endl;
        return 1;
    }
    cout << "Network structure loaded successfully! Nodes: " << BayesNet.netSize() << endl;
    
    vector<vector<string>> raw_data = read_data(data_file);
    if (raw_data.empty()) {
        cerr << "Could not read data or data file is empty. Exiting." << endl;
        return 1;
    }
    cout << "Data loaded successfully! Records: " << raw_data.size() << endl;

    // Build Name-to-Index Map
    map<string, int> name_to_index;
    for (int i = 0; i < BayesNet.netSize(); ++i) {
        name_to_index[BayesNet.getNode(i)->get_name()] = i;
    }

    // --- Phase 1: Impute Data using k-NN ---
    cout << "\n--- Phase 1: Imputing missing data using k-NN (k=" 
         << K_NEIGHBORS << ") ---" << endl;
    vector<vector<string>> complete_dataset = create_complete_dataset(raw_data, K_NEIGHBORS);
    
    // --- Phase 2: Learn CPTs using MLE ---
    cout << "\n--- Phase 2: Learning CPTs from complete data (pseudo_count=" 
         << PSEUDO_COUNT << ") ---" << endl;
    learn_cpts_mle(BayesNet, complete_dataset, name_to_index, PSEUDO_COUNT);
    
    cout << "\n--- Learning complete ---" << endl;
    
    // --- Write the final network ---
    write_network(BayesNet, "solved.bif");

    return 0;
}
#endif

