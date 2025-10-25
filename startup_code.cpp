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

using namespace std;

// Forward Declarations
class network; 
network read_network(const string& filename);
void write_network(const network& BayesNet, const string& filename);
int get_value_index(const class Graph_Node& node, const string& value);
void learn_cpts_mle(network& bn, const vector<vector<string>>& dataset, float pseudo_count);
string sample_missing_value_approx(const network& bn, const vector<string>& incomplete_record,
                                  int missing_index, const map<string, int>& name_to_index);
void learn_parameters_em(network& bn, const vector<vector<string>>& raw_data, 
                         int N_iterations, int N_samples_per_missing, const vector<string>& node_names);
vector<vector<string>> read_data(const string& filename);

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

    // FIX: Added 'const' to all getter methods
    string get_name() const {
        return Node_Name;
    }

    vector<int> get_children() const {
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

    void resize_cpt(long long size) {
        CPT.resize(size, -1.0f);
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

    // FIX: Added get_nth_node alias
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
};

// ==========================================================================================
// UTILITY FUNCTIONS 
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
        bn.addNode(node);
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
                    probs.push_back(stof(trim_whitespace(prob_val)));
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
                    cerr << "Error: Parent value count mismatch for " << child_name << endl;
                    continue;
                }

                // Calculate base index from parent values
                base_index = 0;
                for (int p_idx = 0; p_idx < parent_names.size(); ++p_idx) {
                    const string& pname = parent_names[p_idx];
                    const string& pval = parent_values[p_idx];
                    if (parent_val_indices.count(pname) == 0 || parent_val_indices[pname].count(pval) == 0) {
                        cerr << "Error: Unknown parent value " << pval << " for " << pname << endl;
                        base_index = -1; // Flag error
                        break;
                    }
                    int p_val_idx = parent_val_indices.at(pname).at(pval);
                    base_index += (long long)p_val_idx * parent_multipliers.at(pname);
                }
                if (base_index == -1) continue; // Skip this row

                // Parse probabilities
                size_t probs_start = parent_vals_end + 1;
                size_t probs_end = block_line.find(';');
                if (probs_end == string::npos) probs_end = block_line.size();
                string probs_str = block_line.substr(probs_start, probs_end - probs_start);
                
                stringstream ps(probs_str);
                string prob_val;
                while (getline(ps, prob_val, ',')) {
                    probs.push_back(stof(trim_whitespace(prob_val)));
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
// CORRECTED NETWORK WRITER (NO QUOTES)
// ------------------------------------------------------------------------------------------

void write_network(const network& BayesNet, const string& filename) {
    ofstream outfile(filename);
    if (!outfile.is_open()) {
        cerr << "Error: Could not open file for writing: " << filename << endl;
        return;
    }
    
    outfile << "// Bayesian Network (Learned via EM)" << endl << endl;

    for (int i = 0; i < BayesNet.netSize(); ++i) {
        auto node_it = BayesNet.getNodeConst(i); 
        outfile << "variable " << node_it->get_name() << " {" << endl;
        outfile << "  type discrete [ " << node_it->get_nvalues() << " ] = { ";
        const auto& values = node_it->get_values();
        for (int k = 0; k < node_it->get_nvalues(); ++k) {
            // FIX: Removed quotes to match hailfinder.bif format
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
                    cerr << "Error: Parent node " << pname << " not found." << endl;
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
                    // FIX: Removed quotes to match hailfinder.bif format
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
// HELPER FOR VALUE INDEXING
// ------------------------------------------------------------------------------------------

int get_value_index(const Graph_Node& node, const string& value) {
    const auto& values = node.get_values();
    for (int i = 0; i < values.size(); ++i) {
        if (values[i] == value) {
            return i;
        }
    }
    
    // FIX: Removed the incorrect hack. The loop above handles "StronUp" correctly.
    
    if (value != "?") {
        cerr << "Warning: Value '" << value << "' not found for node " << node.get_name() << endl;
    }
    return -1; 
}


// ------------------------------------------------------------------------------------------
// 2. MLE CALCULATION (M-STEP)
// ------------------------------------------------------------------------------------------

void learn_cpts_mle(network& bn, const vector<vector<string>>& dataset, float pseudo_count = 1.0) {
    
    map<string, int> name_to_index;
    for (int i = 0; i < bn.netSize(); ++i) {
        name_to_index[bn.getNode(i)->get_name()] = i; 
    }
    
    if (name_to_index.empty() && bn.netSize() > 0) {
        cerr << "FATAL (MLE): name_to_index map is empty. Network read failed." << endl;
        return;
    }

    for (int i = 0; i < bn.netSize(); ++i) {
        auto node_it = bn.getNode(i); 
        const auto& node_name = node_it->get_name();
        const auto& parents = node_it->get_Parents();
        int num_child_values = node_it->get_nvalues();

        long long cpt_size = num_child_values;
        for (const auto& pname : parents) {
            if (name_to_index.count(pname) == 0) {
                cerr << "Error (MLE): Parent " << pname << " for node " << node_name << " not in map." << endl;
                return;
            }
            auto pnode_it = bn.getNode(name_to_index.at(pname));
            cpt_size *= pnode_it->get_nvalues();
        }

        vector<float> numerator(cpt_size, pseudo_count);
        vector<float> denominator(cpt_size / num_child_values, (float)num_child_values * pseudo_count);
        
        for (const auto& record : dataset) {
            map<string, int> parent_value_indices;
            bool record_valid = true;
            
            for (const auto& pname : parents) {
                int p_idx = name_to_index.at(pname);
                if (p_idx >= record.size() || record[p_idx] == "?") {
                    record_valid = false;
                    break;
                }
                auto pnode_it = bn.getNode(p_idx);
                int val_idx = get_value_index(*pnode_it, record[p_idx]);
                if (val_idx == -1) {
                    record_valid = false;
                    break;
                }
                parent_value_indices[pname] = val_idx;
            }
            if (!record_valid) continue;

            int child_idx = name_to_index.at(node_name);
            if (child_idx >= record.size() || record[child_idx] == "?") {
                record_valid = false; 
            }
            if (!record_valid) continue;

            int c_val_idx = get_value_index(*node_it, record[child_idx]);
            if (c_val_idx == -1) continue; 
            
            int parent_config_index = 0;
            long long multiplier = 1;
            
            for (int p = 0; p < (int)parents.size(); ++p) {
                const string& parent_name = parents[p];
                int p_val_idx = parent_value_indices.at(parent_name);
                parent_config_index += p_val_idx * multiplier;
                
                auto pnode_it = bn.getNode(name_to_index.at(parent_name));
                multiplier *= pnode_it->get_nvalues();
            }

            int cpt_index = parent_config_index * num_child_values + c_val_idx;
            
            if (cpt_index >= numerator.size() || parent_config_index >= denominator.size()) {
                 cerr << "FATAL (MLE): Index out of bounds." << endl;
                 return;
            }
            
            numerator[cpt_index]++;
            denominator[parent_config_index]++;
        }

        vector<float> new_cpt(cpt_size);
        for (long long j = 0; j < cpt_size / num_child_values; ++j) {
            float total_count = denominator[j];
            for (int k = 0; k < num_child_values; ++k) {
                int cpt_index = j * num_child_values + k;
                if (total_count > 0) {
                    new_cpt[cpt_index] = numerator[cpt_index] / total_count;
                } else {
                    new_cpt[cpt_index] = 1.0f / num_child_values; 
                }
            }
        }
        
        node_it->set_CPT(new_cpt);
    }
}

// ------------------------------------------------------------------------------------------
// 3. E-STEP (SAMPLING/IMPUTATION) HELPER
// ------------------------------------------------------------------------------------------

string sample_missing_value_approx(const network& bn, const vector<string>& incomplete_record,
                                  int missing_index, const map<string, int>& name_to_index) {
    
    auto missing_node_it = bn.getNodeConst(missing_index);
    const auto& parents = missing_node_it->get_Parents();
    const auto& possible_values = missing_node_it->get_values();
    int num_child_values = missing_node_it->get_nvalues();

    int parent_config_index = 0;
    long long multiplier = 1;
    
    for (const auto& pname : parents) {
        if (name_to_index.count(pname) == 0) {
             cerr << "Error (Sample): Parent " << pname << " not in map." << endl;
             return "?";
        }
        int p_idx = name_to_index.at(pname); 
        
        if (incomplete_record[p_idx] == "?") {
             return "?"; 
        }
        
        auto pnode_it = bn.getNodeConst(p_idx); 
        int val_idx = get_value_index(*pnode_it, incomplete_record[p_idx]);

        if (val_idx == -1) {
            return "?"; 
        }

        parent_config_index += val_idx * multiplier;
        multiplier *= pnode_it->get_nvalues();
    }

    const auto& cpt = missing_node_it->get_CPT();
    int cpt_start_index = parent_config_index * num_child_values;
    
    vector<float> conditional_probs(num_child_values);
    float sum_probs = 0.0f;
    for (int k = 0; k < num_child_values; ++k) {
        if (cpt_start_index + k >= cpt.size()) {
             cerr << "Error (Sample): CPT index out of bounds for node " << missing_node_it->get_name() << endl;
             return "?";
        }
        conditional_probs[k] = cpt[cpt_start_index + k];
        sum_probs += conditional_probs[k];
    }

    if (sum_probs < 1e-6) {
        static random_device rd;
        static mt19937 gen(rd());
        uniform_int_distribution<> dis(0, possible_values.size() - 1);
        return possible_values[dis(gen)];
    }
    
    if (abs(sum_probs - 1.0) > 1e-5) {
        for (int k = 0; k < num_child_values; ++k) {
            conditional_probs[k] /= sum_probs;
        }
    }

    static random_device rd;
    static mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);
    float rand_val = dis(gen);
    
    float cumulative_prob = 0.0;
    for (int k = 0; k < num_child_values; ++k) {
        cumulative_prob += conditional_probs[k];
        if (rand_val < cumulative_prob) {
            return possible_values[k];
        }
    }
    
    return possible_values.back();
}


// ------------------------------------------------------------------------------------------
// 4. EM LEARNING FUNCTION (Main Loop)
// ------------------------------------------------------------------------------------------

void learn_parameters_em(network& bn, const vector<vector<string>>& raw_data, 
                         int N_iterations, int N_samples_per_missing, const vector<string>& node_names) {
    
    map<string, int> name_to_index;
    for (int i = 0; i < bn.netSize(); ++i) {
        name_to_index[bn.getNode(i)->get_name()] = i;
    }
    
    vector<vector<string>> complete_data;
    vector<pair<vector<string>, int>> incomplete_data_templates; 
    
    for (const auto& record : raw_data) {
        int missing_count = 0;
        int missing_index = -1;
        for (int j = 0; j < record.size(); ++j) {
            if (record[j] == "?") {
                missing_count++;
                missing_index = j;
            }
        }
        
        if (missing_count == 0) {
            complete_data.push_back(record);
        } else if (missing_count == 1) {
            incomplete_data_templates.push_back({record, missing_index});
        }
    }

    cout << "\nData separated." << endl;
    cout << "  Complete records (for initial CPT): " << complete_data.size() << endl;
    cout << "  Incomplete records (to be imputed): " << incomplete_data_templates.size() << endl;

    cout << "\n--- Step 1: Initial CPT (MLE on complete data) ---" << endl;
    if (complete_data.empty()) {
        cerr << "Error: No complete data found. Cannot initialize CPTs via MLE. Exiting EM." << endl;
        return;
    }
    learn_cpts_mle(bn, complete_data, 1.0); 
    cout << "Initial CPTs computed using MLE on complete data." << endl;
    

    cout << "\n--- Step 2 & 3: Starting EM Iterations (N=" << N_iterations << ") ---" << endl;
    for (int iter = 0; iter < N_iterations; ++iter) {
        cout << "\n--- Iteration " << iter + 1 << "/" << N_iterations << " ---" << endl;
        
        vector<vector<string>> imputed_dataset = complete_data;
        cout << "  E-Step: Generating " << N_samples_per_missing << " samples for each of " << incomplete_data_templates.size() << " incomplete records." << endl;
        
        int imputed_count = 0;
        for (const auto& item : incomplete_data_templates) {
            const vector<string>& incomplete_record = item.first;
            int missing_index = item.second;
            
            for (int k = 0; k < N_samples_per_missing; ++k) {
                vector<string> new_record = incomplete_record;
                string sampled_value = sample_missing_value_approx(bn, incomplete_record, missing_index, name_to_index);
                
                if (sampled_value != "?") {
                    new_record[missing_index] = sampled_value;
                    imputed_dataset.push_back(new_record);
                    imputed_count++;
                }
            }
        }

        cout << "  Imputed " << imputed_count << " records. Total dataset size for M-step: " << imputed_dataset.size() << endl;

        cout << "  M-Step: Recomputing CPTs..." << endl;
        learn_cpts_mle(bn, imputed_dataset, 1.0); 
        
        cout << "Iteration " << iter + 1 << " complete." << endl;
    }
    
    cout << "\n--- EM Algorithm Complete ---" << endl;
}


// ==========================================================================================
// MAIN FUNCTION
// ==========================================================================================

#ifndef BN_LIB
int main() {
    // This is the main function for your EM algorithm
    network BayesNet = read_network("hailfinder.bif");
    
    vector<string> node_names;
    for(int i = 0; i < BayesNet.netSize(); ++i) {
        node_names.push_back(BayesNet.getNode(i)->get_name());
    }

    cout << "Network structure loaded successfully! Number of nodes: " << BayesNet.netSize() << endl;
    
    vector<vector<string>> raw_data = read_data("records.dat");
    if (raw_data.empty()) {
        cerr << "Could not read data or data file is empty. Exiting." << endl;
        return 1;
    }
    
    if (raw_data.size() > 0 && raw_data[0].size() != BayesNet.netSize()) {
        cerr << "Error: Number of data columns (" << raw_data[0].size() << ") does not match number of network nodes (" << BayesNet.netSize() << ")." << endl;
        if (BayesNet.netSize() == 0) {
             cerr << "Fatal Error: Network structure load failed. Cannot proceed with EM." << endl;
             return 1;
        }
         cerr << "Warning: Mismatch between data and network structure. Results may be incorrect." << endl;
    }

    cout << "Data loaded successfully! Total raw records: " << raw_data.size() << endl;

    const int N_ITERATIONS = 30; // N
    const int N_SAMPLES_PER_MISSING = 5; // variable number
    
    learn_parameters_em(BayesNet, raw_data, N_ITERATIONS, N_SAMPLES_PER_MISSING, node_names);
    
    write_network(BayesNet, "solved.bif");

    return 0;
}
#endif