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
#include <random>       // For random initialization and batch sampling
#include <numeric>      // For std::accumulate
#include <limits>       // For std::numeric_limits
#include <chrono>       // For time-keeping
#include <utility>      // For std::pair

using namespace std;

// Forward Declarations
class network; 
class Graph_Node;
network read_network(const string& filename);
void write_network(const network& BayesNet, const string& filename);
int get_value_index(const Graph_Node& node, const string& value);
vector<vector<string>> read_data(const string& filename);

// --- NEW/MODIFIED FUNCTION DECLARATIONS ---
long long get_parent_config_index(const Graph_Node& node, const network& bn, 
                                  const map<string, int>& name_to_index, 
                                  const map<int, int>& parent_val_indices);
long long get_cpt_index(const Graph_Node& node, const network& bn, 
                        const map<string, int>& name_to_index, 
                        int child_val_idx, const map<int, int>& parent_val_indices);
pair<vector<float>, double> calculate_posterior_and_likelihood(
                                  const network& bn, const vector<string>& record,
                                  int missing_index, const map<string, int>& name_to_index);
double calculate_observed_log_likelihood(
                                  const network& bn, const vector<vector<string>>& raw_data,
                                  const map<string, int>& name_to_index);
void initialize_cpts_randomly(network& bn, const map<string, int>& name_to_index,
                                std::mt19937& rng);
void learn_parameters_em(network& bn, const vector<vector<string>>& raw_data, 
                         const map<string, int>& name_to_index,
                         int N_iterations, int batch_size, std::mt19937& rng);


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
    
    // --- Needed for multi-start ---
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
    
    outfile << "// Bayesian Network (Learned via Stochastic EM)" << endl << endl;

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
// 2. EM ALGORITHM HELPER FUNCTIONS
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
 * NEW: Modular helper to get the parent configuration index (the "row" of the CPT).
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
        } else {
             // This can happen if a parent is missing but we are not that parent
             // In our 1-missing-val case, this shouldn't be an issue.
        }
        
        parent_config_index += (long long)val_idx * multiplier;
        
        auto pnode_it = bn.getNodeConst(p_idx);
        multiplier *= pnode_it->get_nvalues();
    }
    return parent_config_index;
}

/**
 * NEW: Modular helper to get the full CPT index (row + column).
 */
long long get_cpt_index(const Graph_Node& node, const network& bn, 
                        const map<string, int>& name_to_index, 
                        int child_val_idx, const map<int, int>& parent_val_indices) {

    long long parent_config_index = get_parent_config_index(node, bn, name_to_index, parent_val_indices);
    return parent_config_index * node.get_nvalues() + child_val_idx;
}


/**
 * NEW: Initializes all CPTs to random (but normalized) probabilities.
 * This is crucial for the multi-start strategy.
 */
void initialize_cpts_randomly(network& bn, const map<string, int>& name_to_index,
                                std::mt19937& rng) {
    
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    for (int i = 0; i < bn.netSize(); ++i) {
        auto node_it = bn.getNode(i);
        int nvalues = node_it->get_nvalues();

        long long cpt_size = nvalues;
        for (const auto& pname : node_it->get_Parents()) {
            cpt_size *= bn.getNode(name_to_index.at(pname))->get_nvalues();
        }

        vector<float> new_cpt(cpt_size);
        long long parent_configs = cpt_size / nvalues;

        for (long long j = 0; j < parent_configs; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < nvalues; ++k) {
                float rand_val = dist(rng) + 1e-6f; // Add epsilon to avoid 0
                new_cpt[j * nvalues + k] = rand_val;
                sum += rand_val;
            }
            // Normalize
            for (int k = 0; k < nvalues; ++k) {
                new_cpt[j * nvalues + k] /= sum;
            }
        }
        node_it->set_CPT(new_cpt);
    }
}


/**
 * MODIFIED: Calculates posterior P(Xm | d_obs) using LOG PROBABILITIES.
 * Returns a pair:
 * 1. The normalized posterior vector (for E-step)
 * 2. The log-likelihood of the observed data log(P(d_obs)) (for scoring)
 */
pair<vector<float>, double> calculate_posterior_and_likelihood(
                                  const network& bn, const vector<string>& record,
                                  int missing_index, const map<string, int>& name_to_index) {
    
    auto missing_node_it = bn.getNodeConst(missing_index);
    int nvalues = missing_node_it->get_nvalues();
    const float epsilon = 1e-9f; // Epsilon to prevent log(0)

    vector<float> log_posterior(nvalues, 0.0f); // Stores un-normalized log-probs

    // --- Term 1: log( P(Xm | Parents(Xm)) ) ---
    const auto& parents = missing_node_it->get_Parents();
    const auto& cpt = missing_node_it->get_CPT();
    
    long long xm_parent_config_index = 0;
    long long multiplier = 1;

    for (const string& pname : parents) {
        int p_idx = name_to_index.at(pname);
        auto pnode_it = bn.getNodeConst(p_idx);
        int val_idx = get_value_index(*pnode_it, record[p_idx]);
        xm_parent_config_index += (long long)val_idx * multiplier;
        multiplier *= pnode_it->get_nvalues();
    }

    for (int k = 0; k < nvalues; ++k) {
        long long cpt_index = xm_parent_config_index * nvalues + k;
        float prob = (cpt_index < cpt.size()) ? cpt[cpt_index] : (1.0f / nvalues);
        log_posterior[k] += log(prob + epsilon);
    }

    // --- Term 2: sum( log( P(Y | Parents(Y)) ) ) for Y in Children(Xm) ---
    const auto& children_indices = missing_node_it->get_children();
    
    for (int child_idx : children_indices) {
        auto child_node_it = bn.getNodeConst(child_idx);
        const auto& child_parents = child_node_it->get_Parents();
        const auto& child_cpt = child_node_it->get_CPT();
        int child_nvalues = child_node_it->get_nvalues();
        
        string y_val_str = record[child_idx];
        int y_val_idx = get_value_index(*child_node_it, y_val_str);
        if (y_val_idx == -1) continue; 

        for (int k = 0; k < nvalues; ++k) { // k is the hypothetical value_index of Xm
            
            long long y_parent_config_index = 0;
            long long y_multiplier = 1;

            for (const string& ypname : child_parents) {
                int yp_idx = name_to_index.at(ypname);
                int val_idx = -1;

                if (yp_idx == missing_index) {
                    val_idx = k; // Use the hypothetical value of Xm
                } else {
                    auto ypnode_it = bn.getNodeConst(yp_idx);
                    val_idx = get_value_index(*ypnode_it, record[yp_idx]);
                }
                
                y_parent_config_index += (long long)val_idx * y_multiplier;
                
                auto ypnode_it = bn.getNodeConst(yp_idx); 
                y_multiplier *= ypnode_it->get_nvalues();
            }

            long long child_cpt_index = y_parent_config_index * child_nvalues + y_val_idx;
            float prob = (child_cpt_index < child_cpt.size()) ? child_cpt[child_cpt_index] : (1.0f / child_nvalues);
            log_posterior[k] += log(prob + epsilon);
        }
    }

    // --- Normalize using Log-Sum-Exp trick ---
    vector<float> final_posterior(nvalues);
    float max_log_prob = *std::max_element(log_posterior.begin(), log_posterior.end());
    double sum_exp = 0.0;

    for (int k = 0; k < nvalues; ++k) {
        // Use double for intermediate sum to prevent precision loss
        double val = exp(log_posterior[k] - max_log_prob);
        final_posterior[k] = (float)val; // Store un-normalized exp
        sum_exp += val;
    }
    
    double log_likelihood_of_record = max_log_prob + log(sum_exp + epsilon);
    
    if (sum_exp < epsilon) {
        for (int k = 0; k < nvalues; ++k) {
            final_posterior[k] = 1.0f / nvalues;
        }
    } else {
        for (int k = 0; k < nvalues; ++k) {
            final_posterior[k] /= (float)sum_exp;
        }
    }

    return {final_posterior, log_likelihood_of_record};
}

/**
 * NEW: Scorer function to calculate the total log-likelihood of the
 * entire observed dataset. Used to compare networks from different restarts.
 */
double calculate_observed_log_likelihood(
                                  const network& bn, const vector<vector<string>>& raw_data,
                                  const map<string, int>& name_to_index) {
    
    double total_log_likelihood = 0.0;
    int n_nodes = bn.netSize();

    for (const auto& record : raw_data) {
        int missing_index = -1;
        for(int j=0; j < record.size(); ++j) {
            if(record[j] == "?") {
                missing_index = j;
                break;
            }
        }

        if (missing_index == -1) {
            // --- CASE 1: Complete Row ---
            double record_log_likelihood = 0.0;
            for (int i = 0; i < n_nodes; ++i) {
                auto node_it = bn.getNodeConst(i);
                int c_val_idx = get_value_index(*node_it, record[i]);
                if (c_val_idx == -1) continue;
                
                map<int, int> parent_val_indices;
                for(const string& pname : node_it->get_Parents()) {
                    int p_idx = name_to_index.at(pname);
                    auto pnode_it = bn.getNodeConst(p_idx);
                    parent_val_indices[p_idx] = get_value_index(*pnode_it, record[p_idx]);
                }
                
                long long cpt_index = get_cpt_index(*node_it, bn, name_to_index, c_val_idx, parent_val_indices);
                const auto& cpt = node_it->get_CPT();
                
                if(cpt_index < cpt.size()) {
                    record_log_likelihood += log(cpt[cpt_index] + 1e-9);
                }
            }
            total_log_likelihood += record_log_likelihood;
        } else {
            // --- CASE 2: Incomplete Row ---
            // We only need the second part of the pair: the record's log-likelihood
            total_log_likelihood += calculate_posterior_and_likelihood(
                                        bn, record, missing_index, name_to_index).second;
        }
    }
    return total_log_likelihood;
}


// ------------------------------------------------------------------------------------------
// 4. EM LEARNING FUNCTION (Mini-Batch)
// ------------------------------------------------------------------------------------------

/**
 * MODIFIED: Implements Stochastic Mini-Batch EM.
 * 1. E-Step: Initializes counts to 0, processes a random batch of data.
 * 2. M-Step: Applies pseudo-count smoothing *once* during CPT update.
 */
void learn_parameters_em(network& bn, const vector<vector<string>>& raw_data, 
                         const map<string, int>& name_to_index,
                         int N_iterations, int batch_size, std::mt19937& rng) {
    
    int n_nodes = bn.netSize();
    const float pseudo_count = 0.1f; // Smoothing applied in M-step
    
    // For sampling records
    std::uniform_int_distribution<int> data_sampler(0, raw_data.size() - 1);

    for (int iter = 0; iter < N_iterations; ++iter) {
        
        // --- E-STEP: Calculate Expected Counts from a Mini-Batch ---
        map<int, vector<float>> expected_numerators;
        map<int, vector<float>> expected_denominators;

        // Initialize all counts to 0.0f
        for (int i = 0; i < n_nodes; ++i) {
            auto node_it = bn.getNode(i);
            int nvalues = node_it->get_nvalues();
            long long cpt_size = nvalues;
            for(const auto& pname : node_it->get_Parents()) {
                cpt_size *= bn.getNode(name_to_index.at(pname))->get_nvalues();
            }
            
            long long denom_size = cpt_size / nvalues;
            expected_numerators[i] = vector<float>(cpt_size, 0.0f);
            expected_denominators[i] = vector<float>(denom_size, 0.0f);
        }

        // --- Process a random mini-batch ---
        for (int i_batch = 0; i_batch < batch_size; ++i_batch) {
            
            // Sample a random record
            const auto& record = raw_data[data_sampler(rng)];

            int missing_index = -1;
            for(int j=0; j < record.size(); ++j) {
                if(record[j] == "?") {
                    missing_index = j;
                    break;
                }
            }

            if (missing_index == -1) {
                // --- CASE 1: Complete Row ---
                for (int i = 0; i < n_nodes; ++i) {
                    auto node_it = bn.getNodeConst(i);
                    int c_val_idx = get_value_index(*node_it, record[i]);
                    if(c_val_idx == -1) continue;
                    
                    map<int, int> parent_val_indices;
                    for(const string& pname : node_it->get_Parents()) {
                        int p_idx = name_to_index.at(pname);
                        auto pnode_it = bn.getNodeConst(p_idx);
                        parent_val_indices[p_idx] = get_value_index(*pnode_it, record[p_idx]);
                    }

                    long long parent_config_index = get_parent_config_index(*node_it, bn, name_to_index, parent_val_indices);
                    long long num_index = get_cpt_index(*node_it, bn, name_to_index, c_val_idx, parent_val_indices);
                    
                    expected_numerators[i][num_index] += 1.0f;
                    expected_denominators[i][parent_config_index] += 1.0f;
                }

            } else {
                // --- CASE 2: Incomplete Row (Xm is missing) ---
                vector<float> posterior = calculate_posterior_and_likelihood(
                                              bn, record, missing_index, name_to_index).first;
                int xm_nvalues = bn.getNode(missing_index)->get_nvalues();

                for (int i = 0; i < n_nodes; ++i) {
                    auto node_it = bn.getNodeConst(i);
                    const auto& parents = node_it->get_Parents();
                    int nvalues = node_it->get_nvalues();
                    
                    bool is_missing_node = (i == missing_index);
                    bool is_child_of_missing = false;
                    for (const auto& pname : parents) {
                        if (name_to_index.at(pname) == missing_index) {
                            is_child_of_missing = true;
                            break;
                        }
                    }

                    if (!is_missing_node && !is_child_of_missing) {
                        // Node is unrelated. Treat as complete.
                        int c_val_idx = get_value_index(*node_it, record[i]);
                        if (c_val_idx == -1) continue;

                        map<int, int> parent_val_indices;
                        for(const string& pname : parents) {
                            int p_idx = name_to_index.at(pname);
                            auto pnode_it = bn.getNodeConst(p_idx);
                            parent_val_indices[p_idx] = get_value_index(*pnode_it, record[p_idx]);
                        }
                        
                        long long parent_config_index = get_parent_config_index(*node_it, bn, name_to_index, parent_val_indices);
                        long long num_index = get_cpt_index(*node_it, bn, name_to_index, c_val_idx, parent_val_indices);

                        expected_numerators[i][num_index] += 1.0f;
                        expected_denominators[i][parent_config_index] += 1.0f;

                    } else if (is_missing_node) {
                        // Node *is* the missing one (Xm)
                        long long parent_config_index = 0;
                        long long multiplier = 1;
                        for(const string& pname : parents) {
                            int p_idx = name_to_index.at(pname);
                            auto pnode_it = bn.getNodeConst(p_idx);
                            int val_idx = get_value_index(*pnode_it, record[p_idx]);
                            parent_config_index += (long long)val_idx * multiplier;
                            multiplier *= pnode_it->get_nvalues();
                        }
                        for (int k = 0; k < xm_nvalues; ++k) {
                            long long num_index = parent_config_index * nvalues + k;
                            expected_numerators[i][num_index] += posterior[k];
                            expected_denominators[i][parent_config_index] += posterior[k];
                        }
                    } else if (is_child_of_missing) {
                        // Node is a child (Y) of the missing one (Xm)
                        int y_val_idx = get_value_index(*node_it, record[i]);
                        if (y_val_idx == -1) continue;
                        
                        for (int k = 0; k < xm_nvalues; ++k) { // k is hypothetical value_index of Xm
                            long long parent_config_index = 0;
                            long long multiplier = 1;
                            for(const string& pname : parents) {
                                int p_idx = name_to_index.at(pname);
                                int val_idx = -1;
                                if (p_idx == missing_index) {
                                    val_idx = k; // Use hypothetical value
                                } else {
                                    auto pnode_it = bn.getNodeConst(p_idx);
                                    val_idx = get_value_index(*pnode_it, record[p_idx]);
                                }
                                parent_config_index += (long long)val_idx * multiplier;
                                auto pnode_it = bn.getNodeConst(p_idx);
                                multiplier *= pnode_it->get_nvalues();
                            }
                            
                            long long num_index = parent_config_index * nvalues + y_val_idx;
                            expected_numerators[i][num_index] += posterior[k];
                            expected_denominators[i][parent_config_index] += posterior[k];
                        }
                    }
                }
            }
        } // end for mini-batch

        // --- M-STEP: Recompute CPTs from Expected Counts ---
        // We update the CPTs by blending the new counts with the old CPTs.
        // This is a standard practice in stochastic/online EM.
        // We use a learning rate: eta = (iter + 2)^-0.7
        // (iter + 2) to avoid issues with iter=0. 0.7 is a common decay rate.
        float eta = std::pow(iter + 2.0f, -0.7f);

        for (int i = 0; i < n_nodes; ++i) {
            auto node_it = bn.getNode(i);
            int nvalues = node_it->get_nvalues();
            const auto& old_cpt = node_it->get_CPT();
            vector<float> new_cpt(old_cpt.size());
            
            const auto& numerators = expected_numerators.at(i);
            const auto& denominators = expected_denominators.at(i);
            
            for (int j = 0; j < denominators.size(); ++j) { // For each parent config
                
                // Get the counts for this batch
                float total_count = denominators[j];
                
                // Apply smoothing
                float smoothed_den = total_count + (pseudo_count * nvalues);

                for (int k = 0; k < nvalues; ++k) { // For each child value
                    long long cpt_index = (long long)j * nvalues + k;
                    float smoothed_num = numerators[cpt_index] + pseudo_count;
                    
                    float batch_prob; // The probability from this batch
                    if (smoothed_den < 1e-9) {
                        batch_prob = 1.0f / nvalues;
                    } else {
                        batch_prob = smoothed_num / smoothed_den;
                    }

                    // Blend old and new probabilities
                    // new = (1 - eta) * old + eta * batch
                    new_cpt[cpt_index] = (1.0f - eta) * old_cpt[cpt_index] + eta * batch_prob;
                }
            }
            // Normalize the new CPT row just in case of float drift
            for (int j = 0; j < denominators.size(); ++j) {
                float row_sum = 0.0f;
                for(int k=0; k < nvalues; ++k) row_sum += new_cpt[j*nvalues + k];
                if (row_sum > 1e-9) {
                    for(int k=0; k < nvalues; ++k) new_cpt[j*nvalues + k] /= row_sum;
                }
            }

            node_it->set_CPT(new_cpt);
        }

        if ((iter + 1) % 50 == 0) {
            cout << "  ... Iteration " << iter + 1 << " complete." << endl;
        }

    } // end for iterations
}


// ==========================================================================================
// MAIN FUNCTION (Multi-Start EM Controller)
// ==========================================================================================

#ifndef BN_LIB
int main(int argc, char* argv[]) {
    
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <hailfinder_file> <data_file>" << endl;
    }

    string hailfinder_file = argv[1];
    string data_file = argv[2];

    // --- Algorithm Hyperparameters ---
    const int NUM_RESTARTS = 10;        // Run 10 different random initializations
    const int N_ITERATIONS = 100;       // Iterations *per restart*
    const float BATCH_SIZE_PERCENT = 0.20f; // Use 20% of data per iteration
    const long TIME_LIMIT_MS = 110000;  // 110 seconds (to be safe for 2min limit)
    // ---------------------------------
    
    auto main_start_time = std::chrono::steady_clock::now();
    
    // --- Load Data and Initial Network Structure ---
    network base_network_structure = read_network(hailfinder_file);
    if (base_network_structure.netSize() == 0) {
        cerr << "Fatal Error: Network structure load failed. Cannot proceed." << endl;
        return 1;
    }
    cout << "Network structure loaded successfully! Nodes: " << base_network_structure.netSize() << endl;
    
    vector<vector<string>> raw_data = read_data(data_file);
    if (raw_data.empty()) {
        cerr << "Could not read data or data file is empty. Exiting." << endl;
        return 1;
    }
    cout << "Data loaded successfully! Records: " << raw_data.size() << endl;

    int batch_size = static_cast<int>(raw_data.size() * BATCH_SIZE_PERCENT);
    cout << "Using " << NUM_RESTARTS << " restarts, " 
         << N_ITERATIONS << " iterations/restart, "
         << "and mini-batch size of " << batch_size << "." << endl;

    // --- Build Name-to-Index Map (Do this ONCE) ---
    map<string, int> name_to_index;
    for (int i = 0; i < base_network_structure.netSize(); ++i) {
        name_to_index[base_network_structure.getNode(i)->get_name()] = i;
    }

    // --- Initialize RNG and Best-Network Tracking ---
    std::random_device rd;
    std::mt19937 rng(rd());
    network best_network;
    double best_likelihood = -std::numeric_limits<double>::infinity();

    // --- MAIN RESTART LOOP ---
    for (int restart = 0; restart < NUM_RESTARTS; ++restart) {
        cout << "\n--- RESTART " << restart + 1 << "/" << NUM_RESTARTS << " ---" << endl;
        
        // 1. Create a fresh network copy
        network current_network = read_network(hailfinder_file); // Re-read to reset CPTs
        
        // 2. Initialize it randomly
        cout << "  Initializing CPTs randomly..." << endl;
        initialize_cpts_randomly(current_network, name_to_index, rng);
        
        // 3. Train it using Mini-Batch EM
        cout << "  Starting Stochastic EM..." << endl;
        learn_parameters_em(current_network, raw_data, name_to_index, 
                            N_ITERATIONS, batch_size, rng);
        
        // 4. Score it using the *full* dataset
        cout << "  Training complete. Scoring network..." << endl;
        double current_likelihood = calculate_observed_log_likelihood(
                                        current_network, raw_data, name_to_index);
        
        cout << "  Restart " << restart + 1 << " Log-Likelihood: " << current_likelihood << endl;

        // 5. Compare and save if it's the best one
        if (current_likelihood > best_likelihood) {
            cout << "  *** New best network found! ***" << endl;
            best_likelihood = current_likelihood;
            best_network = std::move(current_network); // Efficiently move
        }

        // 6. Check time limit
        auto now = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - main_start_time).count();
        if (elapsed_ms > TIME_LIMIT_MS && restart < NUM_RESTARTS - 1) {
            cout << "\n--- Time limit approaching (" << elapsed_ms / 1000.0
                 << "s). Halting restarts. ---" << endl;
            break;
        }
    }
    
    cout << "\n--- All restarts complete ---" << endl;
    cout << "Best Log-Likelihood found: " << best_likelihood << endl;
    
    // Write the best network found across all restarts
    write_network(best_network, "solved.bif");

    return 0;
}
#endif

