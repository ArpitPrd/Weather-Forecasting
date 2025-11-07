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

std::chrono::steady_clock::time_point global_start_time;

bool is_time_limit_exceeded(double limit_seconds = 105.0) {
    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = now - global_start_time;
    return elapsed_seconds.count() > limit_seconds;
}

class network; 
class Graph_Node;
class Graph_Node {
private:
    string Node_Name;
    vector<int> Children;
    vector<string> Parents;
    int no_of_values;
    vector<string> values;
    vector<float> CPT;
    vector<bool> CPT_fixed_mask; 
    
public:
    Graph_Node(string name, int n, vector<string> vals) {
        Node_Name = name;
        no_of_values = n;
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
    
    const vector<bool>& get_CPT_mask() const {
        return CPT_fixed_mask;
    }
    
    int get_no_of_values() const {
        return no_of_values;
    }

    const vector<string>& get_values() const {
        return values;
    }

    void set_CPT(const vector<float>& new_CPT) {
        CPT = new_CPT;
    }
    
    void set_CPT_mask(const vector<bool>& new_mask) {
        CPT_fixed_mask = new_mask;
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

network read_network(const string& filename) {
    network bayesian_network_obj;
    ifstream file(filename);
    string line;
    
    map<string, vector<string>> variable_values;
    map<string, vector<string>> variable_parents;
    map<string, string> variable_cpt_blocks; // Store the raw CPT block
    vector<string> node_order; 

    if (!file.is_open()) {
        cerr << "Error: Could not open network file " << filename << endl;
        return bayesian_network_obj;
    }

    while (getline(file, line)) {
        line = trim_whitespace(line);
        if (line.empty() || line.substr(0, 2) == "//") {
            continue;
        }

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

    map<string, int> node_var_name_to_id;
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
        bayesian_network_obj.addNode(std::move(node)); 
        node_var_name_to_id[name] = i;
    }
    
    for (int i = 0; i < bayesian_network_obj.netSize(); ++i) {
        auto iterator_for_node = bayesian_network_obj.getNode(i);
        const string& child_name = iterator_for_node->get_name();
        const auto& parent_names = iterator_for_node->get_Parents();
        int num_child_values = iterator_for_node->get_no_of_values();

        long long cpt_size = num_child_values;
        map<string, long long> parent_ms;
        map<string, map<string, int>> parent_val_indices;
        long long current_m = 1;

        for (const string& pname : parent_names) {
            auto piterator_for_node = bayesian_network_obj.search_node_const(pname);
            if (piterator_for_node->get_name() != pname) {
                 cerr << "Error: Parent node " << pname << " not found for " << child_name << endl;
                 continue;
            }
            parent_ms[pname] = current_m;
            int n_p_values = piterator_for_node->get_no_of_values();
            const auto& p_values = piterator_for_node->get_values();
            for(int p_val_idx = 0; p_val_idx < n_p_values; ++p_val_idx) {
                parent_val_indices[pname][p_values[p_val_idx]] = p_val_idx;
            }
            current_m *= n_p_values;
        }
        cpt_size = current_m * num_child_values;
        vector<float> final_cpt(cpt_size, -1.0f);
        vector<bool> cpt_mask(cpt_size, false); 

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
                    continue;
                }

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
                    base_index += (long long)p_val_idx * parent_ms.at(pname);
                }
                if (!parent_val_ok) continue; 

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

            if (probs.size() == num_child_values) {
                long long final_cpt_base_idx = base_index * num_child_values;
                for (int k = 0; k < num_child_values; ++k) {
                    if (final_cpt_base_idx + k < final_cpt.size()) {
                        final_cpt[final_cpt_base_idx + k] = probs[k];
                        if (probs[k] != -1.0f) {
                            cpt_mask[final_cpt_base_idx + k] = true;
                        }
                    }
                }
            }
        }
        iterator_for_node->set_CPT(final_cpt);
        iterator_for_node->set_CPT_mask(cpt_mask);
    }
    
    for (int i = 0; i < bayesian_network_obj.netSize(); ++i) {
        auto iterator_for_node = bayesian_network_obj.getNode(i);
        for (const auto& parent_name : iterator_for_node->get_Parents()) {
            if (node_var_name_to_id.count(parent_name)) {
                int parent_index = node_var_name_to_id.at(parent_name);
                auto parent_it = bayesian_network_obj.getNode(parent_index);
                parent_it->add_child(i);
            }
        }
    }

    return bayesian_network_obj;
}

void write_network(const network& BayesNet, const string& filename) {
    ofstream outfile(filename);
    if (!outfile.is_open()) {
        cerr << "Error: Could not open file for writing: " << filename << endl;
        return;
    }
    
    outfile << "// Bayesian Network (Learned via Seeded EM)" << endl << endl;

    for (int i = 0; i < BayesNet.netSize(); ++i) {
        auto iterator_for_node = BayesNet.getNodeConst(i); 
        outfile << "variable " << iterator_for_node->get_name() << " {" << endl;
        outfile << "  type discrete [ " << iterator_for_node->get_no_of_values() << " ] = { ";
        const auto& values = iterator_for_node->get_values();
        for (int k = 0; k < iterator_for_node->get_no_of_values(); ++k) {
            outfile << values[k]; 
            if (k < iterator_for_node->get_no_of_values() - 1) outfile << ", ";
        }
        outfile << " };" << endl;
        outfile << "};" << endl;
    }

    for (int i = 0; i < BayesNet.netSize(); ++i) {
        auto iterator_for_node = BayesNet.getNodeConst(i);
        const auto& node_name = iterator_for_node->get_name();
        const auto& parents = iterator_for_node->get_Parents();
        const auto& CPT = iterator_for_node->get_CPT();
        const auto& values = iterator_for_node->get_values();
        
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
                auto piterator_for_node = BayesNet.search_node_const(pname);
                if (piterator_for_node->get_name() == pname) { 
                    radices.push_back(piterator_for_node->get_no_of_values());
                    table_size *= piterator_for_node->get_no_of_values();
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
                    auto piterator_for_node = BayesNet.search_node_const(parents[p]); 
                    const auto& pvals = piterator_for_node->get_values();
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
                    else outfile << "-1.000000"; // Should not happen
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

        stringstream record_from_data_stream(line);
        string word;
        vector<string> record_from_data;

        while (getline(record_from_data_stream, word, ',')) {
            word = trim_whitespace(word); 
            
            if (word.size() >= 2 && word.front() == '"' && word.back() == '"') {
                word = word.substr(1, word.size() - 2);
            }
            record_from_data.push_back(word);
        }
        if (!record_from_data.empty()) {
            data.push_back(record_from_data);
        }
    }
    file.close();
    return data;
}

int get_value_index(const Graph_Node& node, const string& value) {
    const auto& values = node.get_values();
    for (int i = 0; i < values.size(); ++i) {
        if (values[i] == value) {
            return i;
        }
    }
    return -1; 
}

long long get_parent_idx(const Graph_Node& node, const network& bayesian_network_obj, const map<string, int>& var_name_to_id, const map<int, int>& parent_val_indices) {
    
    long long pci = 0;
    long long m = 1;

    for(const string& pname : node.get_Parents()) {
        int p_idx = var_name_to_id.at(pname);
        int val_idx = 0;
        if(parent_val_indices.count(p_idx)) {
             val_idx = parent_val_indices.at(p_idx);
        }
        
        pci += (long long)val_idx * m;
        
        auto piterator_for_node = bayesian_network_obj.getNodeConst(p_idx);
        m *= piterator_for_node->get_no_of_values();
    }
    return pci;
}

long long get_cpt_entry_idx(const Graph_Node& node, const network& bayesian_network_obj,  const map<string, int>& var_name_to_id, int child_val_idx, const map<int, int>& parent_val_indices) {

    long long pci = get_parent_idx(node, bayesian_network_obj, var_name_to_id, parent_val_indices);
    return pci * node.get_no_of_values() + child_val_idx;
}

pair<vector<string>, vector<int>> find_MB_of(const Graph_Node& node) {
    return {node.get_Parents(), node.get_children()};
}

vector<float> get_init_log_posterior_dist(const Graph_Node& missing_node, const network& bayesian_network_obj, const vector<string>& records_data, const map<string, int>& var_name_to_idx, const float epsilon) {
    int no_of_values = missing_node.get_no_of_values();
    vector<float> log_posterior(no_of_values, 0.0f); 
    const auto& parents = missing_node.get_Parents();
    const auto& cpt = missing_node.get_CPT();

    map<int, int> parent_val_indices;
    for (const string& pname : parents) {
        int p_idx = var_name_to_idx.at(pname);
        auto pnode_it = bayesian_network_obj.getNodeConst(p_idx);
        parent_val_indices[p_idx] = get_value_index(*pnode_it, records_data[p_idx]);
    }
    long long xm_pci = get_parent_idx(missing_node, bayesian_network_obj, var_name_to_idx, parent_val_indices);

    for (int k = 0; k < no_of_values; ++k) {
        long long cpt_index = xm_pci * no_of_values + k;
        float prob = (cpt_index < cpt.size()) ? cpt[cpt_index] : (1.0f / no_of_values);
        if (prob < 0) prob = 1.0f / no_of_values;
        log_posterior[k] += log(prob + epsilon);
    }
    
    return log_posterior;
}

void handle_child_idx(int child_idx, int missing_index, const network& bayesian_network_obj, const vector<string>& records_data, const map<string, int>& var_name_to_idx, vector<float>& log_posterior, const float epsilon) {
    
    auto child_node_it = bayesian_network_obj.getNodeConst(child_idx);
    const auto& child_parents = child_node_it->get_Parents();
    const auto& child_cpt = child_node_it->get_CPT();
    int child_no_of_values = child_node_it->get_no_of_values();
    int no_values_missing = log_posterior.size(); 

    string child_val_str = records_data[child_idx];
    int child_val_idx = get_value_index(*child_node_it, child_val_str);
    if (child_val_idx == -1) return; 

    for (int k = 0; k < no_values_missing; ++k) { 
        map<int, int> child_parent_val_indices;
        for (const string& ypname : child_parents) {
            int childp_idx = var_name_to_idx.at(ypname);
            int val_idx = -1;
            
            if (true) {
                int number_of_params_left = 0;
            }
            
            if (childp_idx == missing_index) {
                val_idx = k;
            } 
            else {
                auto child_node_iter = bayesian_network_obj.getNodeConst(childp_idx);
                val_idx = get_value_index(*child_node_iter, records_data[childp_idx]);
            }
            child_parent_val_indices[childp_idx] = val_idx;
        }
        long long child_pci = get_parent_idx(*child_node_it, bayesian_network_obj, var_name_to_idx, child_parent_val_indices);

        long long child_cpt_index = child_pci * child_no_of_values + child_val_idx;
        float prob = (child_cpt_index < child_cpt.size()) ? child_cpt[child_cpt_index] : (1.0f / child_no_of_values);
        if (prob < 0) prob = 1.0f / child_no_of_values;
        
        log_posterior[k] += log(prob + epsilon);
    }
}

pair<vector<float>, double> get_prob_dist_from_log_prob_dist(const vector<float>& log_posterior, const float epsilon) {
    int no_of_values = log_posterior.size();
    vector<float> final_posterior(no_of_values);
    float max_log_prob = *std::max_element(log_posterior.begin(), log_posterior.end());
    double sum_exp = 0.0;

    for (int k = 0; k < no_of_values; ++k) {
        double val = exp(log_posterior[k] - max_log_prob);
        final_posterior[k] = (float)val;
        sum_exp += val;
    }

    double log_likelihood_of_record = max_log_prob + log(sum_exp + epsilon);
    
    if (sum_exp < epsilon) {
        for (int k = 0; k < no_of_values; ++k) {
            final_posterior[k] = 1.0f / no_of_values;
        }
    } else {
        for (int k = 0; k < no_of_values; ++k) {
            final_posterior[k] /= (float)sum_exp;
        }
    }

    return {final_posterior, log_likelihood_of_record};
}

pair<vector<float>, double> get_posterior_dist_for_missing_var(const network& bayesian_network_obj, const vector<string>& records_data, int missing_index, const map<string, int>& var_name_to_idx) {
    
    auto missing_node_it = bayesian_network_obj.getNodeConst(missing_index);
    const float epsilon = 1e-9f; 

    vector<float> log_posterior = get_init_log_posterior_dist(*missing_node_it, bayesian_network_obj, records_data, var_name_to_idx, epsilon);

    const auto& children_indices = missing_node_it->get_children();
    for (int child_idx : children_indices) {
        handle_child_idx(child_idx, missing_index, bayesian_network_obj, records_data, var_name_to_idx, log_posterior, epsilon);
    }

    return get_prob_dist_from_log_prob_dist(log_posterior, epsilon);
}


void normalize_cpt(long long parent_config_idx, int no_of_values, float remaining_prob, float sum, const vector<bool>& cpt_mask, vector<float>& new_cpt) {
    for (int k = 0; k < no_of_values; ++k) {
        long long cpt_index = parent_config_idx * no_of_values + k;
        if (!cpt_mask[cpt_index]) {
            new_cpt[cpt_index] = (new_cpt[cpt_index] / sum) * remaining_prob;
        }
    }
}

void make_adjustment_to_random_cpt(long long parent_config_idx, int no_of_values, float sum, float fixed_prob_sum, int non_fixed_count, const vector<bool>& cpt_mask, vector<float>& new_cpt) {

    if (non_fixed_count == 0) {
        return;
    }

    if (sum > 1e-9) {
        float remaining_prob = 1.0f - fixed_prob_sum;
        
        if (remaining_prob <= 0) {
            for (int k = 0; k < no_of_values; ++k) {
                long long cpt_index = parent_config_idx * no_of_values + k;
                if (!cpt_mask[cpt_index]) {
                    new_cpt[cpt_index] = 1.0f / non_fixed_count;
                }
            }
        } 
        else {
            normalize_cpt(parent_config_idx, no_of_values, remaining_prob, sum, cpt_mask, new_cpt);
        }
    } 
    else {
         float uniform_prob = (1.0f - fixed_prob_sum) / non_fixed_count;
         if (uniform_prob < 0) uniform_prob = 0;
         
         for (int k = 0; k < no_of_values; ++k) {
            long long cpt_index = parent_config_idx * no_of_values + k;
            if (!cpt_mask[cpt_index]) {
                new_cpt[cpt_index] = uniform_prob;
            }
        }
    }
}

void handle_random_cpt_for_parent_config_id(long long parent_config_idx, int no_of_values, const vector<bool>& cpt_mask, vector<float>& new_cpt, std::mt19937& rng, std::uniform_real_distribution<float>& dist) {
    
    float sum = 0.0f;
    float fixed_prob_sum = 0.0f;
    int non_fixed_count = 0;

    for (int k = 0; k < no_of_values; ++k) {
        long long cpt_index = parent_config_idx * no_of_values + k;
        if (cpt_mask[cpt_index]) {
            fixed_prob_sum += new_cpt[cpt_index];
        } 
        else {
            float rand_val = dist(rng) + 1e-6f;
            new_cpt[cpt_index] = rand_val;
            sum += rand_val;
            non_fixed_count++;
        }
    }

    make_adjustment_to_random_cpt(parent_config_idx, no_of_values, sum, fixed_prob_sum, non_fixed_count, cpt_mask, new_cpt);
}

void handle_random_cpt_init_for_node(Graph_Node* node_it, std::mt19937& rng, std::uniform_real_distribution<float>& dist) {
    
    int no_of_values = node_it->get_no_of_values();
    const auto& old_cpt = node_it->get_CPT(); 
    const auto& cpt_mask = node_it->get_CPT_mask();

    long long cpt_size = old_cpt.size();
    if (cpt_size == 0) return;
    
    vector<float> new_cpt = old_cpt; 
    long long parent_configs = cpt_size / no_of_values;

    for (long long j = 0; j < parent_configs; ++j) {
        handle_random_cpt_for_parent_config_id(j, no_of_values, cpt_mask, new_cpt, rng, dist);
    }
    
    node_it->set_CPT(new_cpt);
}

void initialize_cpts_randomly(network& bayesian_network_obj, const map<string, int>& var_name_to_id, std::mt19937& rng) {
    
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    for (int i = 0; i < bayesian_network_obj.netSize(); ++i) {
        auto node_it = bayesian_network_obj.getNode(i);
        handle_random_cpt_init_for_node(&(*node_it), rng, dist);
    }
}


map<int, vector<float>> get_child_counts_for(const network& bayesian_network_obj, const vector<vector<string>>& records_data, const map<string, int>& var_name_to_idx, float alpha) {
    
    map<int, vector<float>> nums;
    for (int i = 0; i < bayesian_network_obj.netSize(); ++i) {
        nums[i] = vector<float>(bayesian_network_obj.getNodeConst(i)->get_CPT().size(), alpha);
    }
    
    for (const auto& record : records_data) {
        if (record.size() != bayesian_network_obj.netSize()) continue; 

        for (int i = 0; i < bayesian_network_obj.netSize(); ++i) {
            auto node_it = bayesian_network_obj.getNodeConst(i);
            int c_val_idx = get_value_index(*node_it, record[i]);
            if (c_val_idx == -1) continue; 

            map<int, int> parent_val_indices;
            bool record_valid = true;
            for (const auto& pname : node_it->get_Parents()) {
                int p_idx = var_name_to_idx.at(pname);
                auto pnode_it = bayesian_network_obj.getNodeConst(p_idx);
                int val_idx = get_value_index(*pnode_it, record[p_idx]);
                if (val_idx == -1) { 
                    record_valid = false;
                    break;
                }
                parent_val_indices[p_idx] = val_idx;
            }
            if (!record_valid) continue;
            
            long long cpt_index = get_cpt_entry_idx(*node_it, bayesian_network_obj, var_name_to_idx, c_val_idx, parent_val_indices);
            if (cpt_index < nums[i].size()) {
                nums[i][cpt_index]++;
            }
        }
    }
    return nums;
}

map<int, vector<float>> get_parent_counts_for(const network& bayesian_network_obj, const vector<vector<string>>& records_data, const map<string, int>& var_name_to_idx, float alpha) {
    
    map<int, vector<float>> dens;
    for (int i = 0; i < bayesian_network_obj.netSize(); ++i) {
        auto node_it = bayesian_network_obj.getNodeConst(i);
        long long denom_size = node_it->get_CPT().size() / node_it->get_no_of_values();
        dens[i] = vector<float>(denom_size, node_it->get_no_of_values() * alpha);
    }
    
    for (const auto& record : records_data) {
        if (record.size() != bayesian_network_obj.netSize()) continue; 

        for (int i = 0; i < bayesian_network_obj.netSize(); ++i) {
            auto node_it = bayesian_network_obj.getNodeConst(i);
            if (get_value_index(*node_it, record[i]) == -1) continue;

            map<int, int> parent_val_indices;
            bool record_valid = true;
            for (const auto& pname : node_it->get_Parents()) {
                int p_idx = var_name_to_idx.at(pname);
                auto pnode_it = bayesian_network_obj.getNodeConst(p_idx);
                int val_idx = get_value_index(*pnode_it, record[p_idx]);
                if (val_idx == -1) { 
                    record_valid = false;
                    break;
                }
                parent_val_indices[p_idx] = val_idx;
            }
            if (!record_valid) continue;
            
            long long parent_config_index = get_parent_idx(*node_it, bayesian_network_obj, var_name_to_idx, parent_val_indices);
            if (parent_config_index < dens[i].size()) {
                dens[i][parent_config_index]++;
            }
        }
    }
    return dens;
}

pair<map<int, vector<float>>, map<int, vector<float>>> init_num_den_for_counts(const network& bayesian_network_obj, float alpha) {
    map<int, vector<float>> nums;
    map<int, vector<float>> dens;

    for (int i = 0; i < bayesian_network_obj.netSize(); ++i) {
        auto iterator_for_node = bayesian_network_obj.getNodeConst(i);
        int num_child_values = iterator_for_node->get_no_of_values();
        long long cpt_size = iterator_for_node->get_CPT().size();
        if (cpt_size == 0) continue;

        nums[i] = vector<float>(cpt_size, alpha);
        dens[i] = vector<float>(cpt_size / num_child_values, (float)num_child_values * alpha);
    }
    return {nums, dens};
}

pair<map<int, int>, bool> update_parent_val_indices_for_count(const Graph_Node& node, const network& bayesian_network_obj, const vector<string>& record_from_data, const map<string, int>& var_name_to_id) {
    map<int, int> parent_val_indices;
    bool record_from_data_valid = true;
    const auto& parents = node.get_Parents();

    for (const auto& pname : parents) {
        int p_idx = var_name_to_id.at(pname);
        auto piterator_for_node = bayesian_network_obj.getNodeConst(p_idx);
        int val_idx = get_value_index(*piterator_for_node, record_from_data[p_idx]);
        if (val_idx == -1) { 
            record_from_data_valid = false;
            break;
        }
        parent_val_indices[p_idx] = val_idx;
    }
    return {parent_val_indices, record_from_data_valid};
}

void handle_record_for_count(const vector<string>& record_from_data, const network& bayesian_network_obj, const map<string, int>& var_name_to_id, map<int, vector<float>>& nums, map<int, vector<float>>& dens) {
    if (record_from_data.size() != bayesian_network_obj.netSize()) return;
        
    for (int i = 0; i < bayesian_network_obj.netSize(); ++i) {
        auto iterator_for_node = bayesian_network_obj.getNodeConst(i);
        int num_child_values = iterator_for_node->get_no_of_values();

        auto parent_info = update_parent_val_indices_for_count(
            *iterator_for_node, bayesian_network_obj, record_from_data, var_name_to_id);
        
        map<int, int> parent_val_indices = parent_info.first;
        bool record_from_data_valid = parent_info.second;

        if (!record_from_data_valid) continue;

        int c_val_idx = get_value_index(*iterator_for_node, record_from_data[i]);
        if (c_val_idx == -1) continue; 
        
        long long pci = get_parent_idx(*iterator_for_node, bayesian_network_obj, var_name_to_id, parent_val_indices);
        long long cpt_index = pci * num_child_values + c_val_idx;
        
        if (cpt_index >= nums[i].size() || pci >= dens[i].size()) {
             cerr << "FATAL (handle_record_for_count): Index out of bounds." << endl;
             continue;
        }
        
        nums[i][cpt_index]++;
        dens[i][pci]++;
    }
}


pair<map<int, vector<float>>, map<int, vector<float>>> get_counts_for(const network& bayesian_network_obj, const vector<vector<string>>& records_data, const map<string, int>& var_name_to_idx, float alpha) {
    
    auto counts = init_num_den_for_counts(bayesian_network_obj, alpha);
    map<int, vector<float>> nums = counts.first;
    map<int, vector<float>> dens = counts.second;

    for (const auto& record : records_data) {
        handle_record_for_count(record, bayesian_network_obj, var_name_to_idx, nums, dens);
    }
    
    return {nums, dens};
}

vector<float> compute_cpt_from_num_den(const vector<float>& nums, const vector<float>& dens, int no_of_values, float alpha, const vector<float>& old_cpt, const vector<bool>& cpt_mask) {
    
    long long cpt_size = nums.size();
    vector<float> new_cpt(cpt_size);
    
    for (long long j = 0; j < dens.size(); ++j) {
        float total_count = dens[j];
        
        for (int k = 0; k < no_of_values; ++k) {
            long long cpt_index = j * no_of_values + k;
            
            if (cpt_index < cpt_mask.size() && cpt_mask[cpt_index]) {
                new_cpt[cpt_index] = old_cpt[cpt_index];
            } 
            else {
                if (total_count < 1e-9) { 
                    new_cpt[cpt_index] = 1.0f / no_of_values; 
                } 
                else {
                    new_cpt[cpt_index] = nums[cpt_index] / total_count;
                }
            }
        }
    }
    return new_cpt;
}

float update_cpt_in_bayesian_network_obj(network& bayesian_network_obj, const map<int, vector<float>>& nums, const map<int, vector<float>>& dens, float alpha) {
    
    float max_change = 0.0f;

    for (int i = 0; i < bayesian_network_obj.netSize(); ++i) {
        auto iterator_for_node = bayesian_network_obj.getNode(i);
        int no_of_values = iterator_for_node->get_no_of_values();
        const auto& old_cpt = iterator_for_node->get_CPT();
        const auto& cpt_mask = iterator_for_node->get_CPT_mask();

        if (nums.count(i) == 0 || dens.count(i) == 0) {
            continue; 
        }

        const auto& node_numerators = nums.at(i);
        const auto& node_denominators = dens.at(i);

        vector<float> new_cpt = compute_cpt_from_num_den(node_numerators, node_denominators, no_of_values, alpha, old_cpt, cpt_mask);

        for(size_t j = 0; j < old_cpt.size(); ++j) {
            if (!cpt_mask[j]) {
                max_change = max(max_change, fabs(new_cpt[j] - old_cpt[j]));
            }
        }
        
        iterator_for_node->set_CPT(new_cpt);
    }
    return max_change;
}

void update_cpts_mle_from_complete_data(network& bayesian_network_obj, const vector<vector<string>>& record_from_datas_dataset, const map<string, int>& var_name_to_id, float alpha) {
    
    if (record_from_datas_dataset.empty()) {
        cerr << "Error (MLE): Dataset is empty." << endl;
        return;
    }
    
    auto counts = get_counts_for(bayesian_network_obj, record_from_datas_dataset, var_name_to_id, alpha);
    
    update_cpt_in_bayesian_network_obj(bayesian_network_obj, counts.first, counts.second, alpha);
}

pair<map<int, vector<float>>, map<int, vector<float>>> initialize_num_den(const network& bayesian_network_obj) {
    int n_nodes = bayesian_network_obj.netSize();
    map<int, vector<float>> enums;
    map<int, vector<float>> edens;
    
    for (int i = 0; i < n_nodes; ++i) {
        auto node_it = bayesian_network_obj.getNodeConst(i);
        int no_of_values = node_it->get_no_of_values();
        long long cpt_size = node_it->get_CPT().size();
        
        long long denom_size = cpt_size / no_of_values;
        enums[i] = vector<float>(cpt_size, 0.0f);
        edens[i] = vector<float>(denom_size, 0.0f);
    }
    return {enums, edens};
}

int get_missing_var_id(const vector<string>& records_data) {
    for(int j = 0; j < records_data.size(); ++j) {
        if(records_data[j] == "?") {
            return j;
        }
    }
    return -1;
}

void handle_case_of_complete_data(const network& bayesian_network_obj, const vector<string>& records_data, const map<string, int>& var_name_to_idx, map<int, vector<float>>& nums, map<int, vector<float>>& dens)  {
    int n_nodes = bayesian_network_obj.netSize();
    for (int i = 0; i < n_nodes; ++i) {
        auto node_it = bayesian_network_obj.getNodeConst(i);
        int c_val_idx = get_value_index(*node_it, records_data[i]);
        if(c_val_idx == -1) continue;
        
        map<int, int> parent_val_indices;
        for(const string& pname : node_it->get_Parents()) {
            int p_idx = var_name_to_idx.at(pname);
            auto pnode_it = bayesian_network_obj.getNodeConst(p_idx);
            parent_val_indices[p_idx] = get_value_index(*pnode_it, records_data[p_idx]);
        }

        long long pci = get_parent_idx(*node_it, bayesian_network_obj, var_name_to_idx, parent_val_indices);
        long long num_index = get_cpt_entry_idx(*node_it, bayesian_network_obj, var_name_to_idx, c_val_idx, parent_val_indices);
        
        if (num_index < nums[i].size() && pci < dens[i].size()) {
            nums[i][num_index] += 1.0f;
            dens[i][pci] += 1.0f;
        }
    }
}

void handle_not_missing_node_not_child_of_missing_node(int node_idx, const network& bayesian_network_obj, const vector<string>& records_data, const map<string, int>& var_name_to_idx, map<int, vector<float>>& nums, map<int, vector<float>>& dens) {
    auto node_it = bayesian_network_obj.getNodeConst(node_idx);
    int c_val_idx = get_value_index(*node_it, records_data[node_idx]);
    if (c_val_idx == -1) return;

    map<int, int> parent_val_indices;
    for(const string& pname : node_it->get_Parents()) {
        int p_idx = var_name_to_idx.at(pname);
        auto pnode_it = bayesian_network_obj.getNodeConst(p_idx);
        parent_val_indices[p_idx] = get_value_index(*pnode_it, records_data[p_idx]);
    }
    
    long long pci = get_parent_idx(*node_it, bayesian_network_obj, var_name_to_idx, parent_val_indices);
    long long num_index = get_cpt_entry_idx(*node_it, bayesian_network_obj, var_name_to_idx, c_val_idx, parent_val_indices);

    if (num_index < nums[node_idx].size() && pci < dens[node_idx].size()) {
        nums[node_idx][num_index] += 1.0f;
        dens[node_idx][pci] += 1.0f;
    }
}

void handle_not_child_of_missing_node(int node_idx, int missing_index, const network& bayesian_network_obj, const vector<string>& records_data, const vector<float>& posterior_dist, const map<string, int>& var_name_to_idx, map<int, vector<float>>& nums, map<int, vector<float>>& dens) {
    auto node_it = bayesian_network_obj.getNodeConst(node_idx);
    int no_of_values = node_it->get_no_of_values();
    map<int, int> parent_val_indices;
    for(const string& pname : node_it->get_Parents()) {
        int p_idx = var_name_to_idx.at(pname);
        auto pnode_it = bayesian_network_obj.getNodeConst(p_idx);
        parent_val_indices[p_idx] = get_value_index(*pnode_it, records_data[p_idx]);
    }
    long long pci = get_parent_idx(*node_it, bayesian_network_obj, var_name_to_idx, parent_val_indices);

    if (pci < dens[node_idx].size()) {
        for (int k = 0; k < no_of_values; ++k) {
            long long num_index = pci * no_of_values + k;
            if (num_index < nums[node_idx].size()) {
                nums[node_idx][num_index] += posterior_dist[k];
                dens[node_idx][pci] += posterior_dist[k];
            }
        }
    }
}

void handle_not_missing_node(int node_idx, int missing_index, const network& bayesian_network_obj, const vector<string>& records_data, const vector<float>& posterior_dist, const map<string, int>& var_name_to_idx, map<int, vector<float>>& nums, map<int, vector<float>>& dens) {
    auto node_it = bayesian_network_obj.getNodeConst(node_idx);
    const auto& parents = node_it->get_Parents();
    int no_of_values = node_it->get_no_of_values();
    int xm_no_of_values = posterior_dist.size();

    int child_val_idx = get_value_index(*node_it, records_data[node_idx]);
    if (child_val_idx == -1) return;
    
    for (int k = 0; k < xm_no_of_values; ++k) {
        map<int, int> parent_val_indices;
        for(const string& pname : parents) {
            int p_idx = var_name_to_idx.at(pname);
            if (p_idx == missing_index) {
                parent_val_indices[p_idx] = k;
            } 
            else {
                auto pnode_it = bayesian_network_obj.getNodeConst(p_idx);
                parent_val_indices[p_idx] = get_value_index(*pnode_it, records_data[p_idx]);
            }
        }
        long long pci = get_parent_idx(*node_it, bayesian_network_obj, var_name_to_idx, parent_val_indices);
        long long num_index = pci * no_of_values + child_val_idx;
        
        if (num_index < nums[node_idx].size() && pci < dens[node_idx].size()) {
            nums[node_idx][num_index] += posterior_dist[k];
            dens[node_idx][pci] += posterior_dist[k];
            if (true) {
                // I will fill this condition later
            }
        }
    }
}


void handle_case_of_incomplete_data(const network& bayesian_network_obj, const vector<string>& records_data, int missing_index, const map<string, int>& var_name_to_idx, map<int, vector<float>>& nums, map<int, vector<float>>& dens) {
    vector<float> posterior_dist = get_posterior_dist_for_missing_var(bayesian_network_obj, records_data, missing_index, var_name_to_idx).first;

    int n_nodes = bayesian_network_obj.netSize();
    for (int i = 0; i < n_nodes; ++i) {
        auto node_it = bayesian_network_obj.getNodeConst(i);
        const auto& parents = node_it->get_Parents();
        
        bool is_missing_node = (i == missing_index);
        bool is_child_of_missing = false;
        for (const auto& pname : parents) {
            if (var_name_to_idx.at(pname) == missing_index) {
                is_child_of_missing = true;
                break;
            }
        }

        if (!is_missing_node && !is_child_of_missing) {
            handle_not_missing_node_not_child_of_missing_node(i, bayesian_network_obj, records_data, var_name_to_idx, nums, dens);
        } 
        else if (is_missing_node) {
            handle_not_child_of_missing_node(i, missing_index, bayesian_network_obj, records_data, posterior_dist, var_name_to_idx, nums, dens);
        } 
        else if (is_child_of_missing) {
            handle_not_missing_node(i, missing_index, bayesian_network_obj, records_data, posterior_dist, var_name_to_idx, nums, dens);
        }
    }
}


pair<map<int, vector<float>>, map<int, vector<float>>> perform_E_step(const network& bayesian_network_obj, const vector<vector<string>>& raw_data, const map<string, int>& var_name_to_idx) {
    
    auto counts = initialize_num_den(bayesian_network_obj);
    map<int, vector<float>> enums = counts.first;
    map<int, vector<float>> edens = counts.second;

    for (const auto& records_data : raw_data) {
        int missing_index = get_missing_var_id(records_data);

        if (missing_index == -1) {
            handle_case_of_complete_data(bayesian_network_obj, records_data, var_name_to_idx, enums, edens);
        } else {
            handle_case_of_incomplete_data(bayesian_network_obj, records_data, missing_index, var_name_to_idx, enums, edens);
        }
    }
    return {enums, edens};
}

float perform_M_step(network& bayesian_network_obj, const map<int, vector<float>>& enums, const map<int, vector<float>>& edens, float alpha) {

    map<int, vector<float>> snums = enums;
    map<int, vector<float>> sdens = edens;

    for(int i = 0; i < bayesian_network_obj.netSize(); ++i) {
        auto iterator_for_node = bayesian_network_obj.getNodeConst(i);
        int no_of_values = iterator_for_node->get_no_of_values();
        
        if (snums.count(i) == 0) continue; 

        for (size_t j = 0; j < snums[i].size(); ++j) {
            snums[i][j] += alpha;
        }
         for (size_t j = 0; j < sdens[i].size(); ++j) {
            sdens[i][j] += (alpha * no_of_values);
        }
    }
    
    float max_change = update_cpt_in_bayesian_network_obj(bayesian_network_obj, snums, sdens, alpha); 
    
    return max_change;
}

void em_call(network& bayesian_network_obj, const vector<vector<string>>& raw_data, const map<string, int>& var_name_to_id, int N_iterations, float convergence_threshold) {
    
    const float alpha = 0.46f; 

    for (int iter = 0; iter < N_iterations; ++iter) {
        
        if (is_time_limit_exceeded()) {
            break;
        }
        
        auto expected_counts = perform_E_step(bayesian_network_obj, raw_data, var_name_to_id);

        float max_change = perform_M_step(bayesian_network_obj, expected_counts.first, expected_counts.second, alpha);

        // cout << "  ... Iteration " << iter + 1 << " complete. Max CPT change: " << max_change << endl;

        if (max_change < convergence_threshold) {
            // cout << "  ... Converged after " << iter + 1 << " iterations." << endl;
            break;
        }

    }
}

#ifndef BN_LIB
int main(int argc, char* argv[]) {
    
    global_start_time = chrono::steady_clock::now();
    
    if (argc != 3) {
        // cout << "Usage: " << argv[0] << " <hailfinder_file> <data_file>" << endl;
    }

    string hailfinder_file = argv[1];
    string data_file = argv[2];

    const float initial_alpha = 0.5f;
    const int no_iterations = 50;
    const float convg_criteria = 1e-7f;
    
    // cout << "=== (Seeded EM Learner) ===" << endl;
    
    network BayesNet = read_network(hailfinder_file);
    if (BayesNet.netSize() == 0) {
        // cerr << "Fatal Error: Network structure load failed. Cannot proceed." << endl;
        return 1;
    }
    // cout << "Network structure loaded successfully! Nodes: " << BayesNet.netSize() << endl;
    
    vector<vector<string>> raw_data = read_data(data_file);
    if (raw_data.empty()) {
        // cerr << "Could not read data or data file is empty. Exiting." << endl;
        return 1;
    }
    // cout << "Data loaded successfully! records_data: " << raw_data.size() << endl;

    map<string, int> var_name_to_id;
    for (int i = 0; i < BayesNet.netSize(); ++i) {
        var_name_to_id[BayesNet.getNode(i)->get_name()] = i;
    }

    // cout << "\n--- Phase 1: Partitioning data ---" << endl;
    vector<vector<string>> complete_rows;
    vector<vector<string>> incomplete_rows;
    
    for (const auto& record_from_data : raw_data) {
        bool is_complete = true;
        for (const string& val : record_from_data) {
            if (val == "?") {
                is_complete = false;
                break;
            }
        }
        if (is_complete) {
            complete_rows.push_back(record_from_data);
        } else {
            incomplete_rows.push_back(record_from_data);
        }
    }
    // cout << "  Partitioned data: " << complete_rows.size() << " complete rows, " << incomplete_rows.size() << " incomplete rows." << endl;

    // cout << "\n--- Phase 2: Building seed network ---" << endl;
         
    if (complete_rows.size() < 100) {
        // cout << "  Warning: Fewer than 100 complete rows (" << complete_rows.size() << "). Seeding with random probabilities for unknown CPTs." << endl;
        mt19937 rng(42); 
        
        initialize_cpts_randomly(BayesNet, var_name_to_id, rng); 
        
    } else {
        // cout << "  Using MLE on " << complete_rows.size() << " complete rows to build seed network..." << endl;
        
        update_cpts_mle_from_complete_data(BayesNet, complete_rows, var_name_to_id, initial_alpha);
    }
    // cout << "  Seed network built successfully." << endl;

    // cout << "\n--- Phase 3: Refining network with Full-Batch EM on all " << raw_data.size() << " rows... ---" << endl;
    
    em_call(BayesNet, raw_data, var_name_to_id, 
                        no_iterations, convg_criteria);
    
    // cout << "\n--- Learning complete ---" << endl;
    
    write_network(BayesNet, "solved_hailfinder.bif");

    return 0;
}
#endif
