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
#include <random> // For sampling

using namespace std;

// =========================================================================================
// FORWARD DECLARATIONS
// =========================================================================================

class network; // Forward declaration for Graph_Node
network read_network(const string& filename);
int get_value_index(const class Graph_Node& node, const string& value);
vector<string> generate_sample(network& bn, mt19937& gen, map<string, int>& name_to_index);

// =========================================================================================
// BAYESIAN NETWORK CLASS DEFINITIONS
// =========================================================================================

/**
 * @brief Represents a single node in the Bayesian Network.
 */
class Graph_Node {
private:
    string Node_Name;
    vector<int> Children;
    vector<string> Parents;
    int nvalues;
    vector<string> values;
    vector<float> CPT;

public:
    // Constructor
    Graph_Node(string name, int n, vector<string> vals) {
        Node_Name = name;
        nvalues = n;
        values = vals;
    }

    // Accessors
    string get_name() const { return Node_Name; }
    int get_nvalues() const { return nvalues; }
    const vector<string>& get_values() const { return values; }
    vector<float>& get_CPT() { return CPT; }
    const vector<float>& get_CPT() const { return CPT; }
    const vector<string>& get_Parents() const { return Parents; }
    const vector<int>& get_Children() const { return Children; }

    // Modifiers
    void set_CPT(vector<float> new_CPT) { CPT = new_CPT; }
    void add_parent(string name) { Parents.push_back(name); }
    void add_child(int id) { Children.push_back(id); }
};

/**
 * @brief Represents the entire Bayesian Network.
 */
class network {
private:
    list<Graph_Node> Pres_Nodes;
    int nNodes;

public:
    // Constructor
    network() { nNodes = 0; }

    // Accessors
    int netSize() const { return nNodes; }
    
    Graph_Node* get_nth_node(int n) {
        if (n < 0 || n >= nNodes) {
            cerr << "Error: Invalid node index " << n << endl;
            return nullptr;
        }
        auto it = Pres_Nodes.begin();
        advance(it, n);
        return &(*it);
    }

    const Graph_Node* get_nth_node(int n) const {
         if (n < 0 || n >= nNodes) {
            cerr << "Error: Invalid node index " << n << endl;
            return nullptr;
        }
        auto it = Pres_Nodes.begin();
        advance(it, n);
        return &(*it);
    }

    // Find node by name
    Graph_Node* getNode(const string& name) {
        for (auto it = Pres_Nodes.begin(); it != Pres_Nodes.end(); ++it) {
            if (it->get_name() == name) {
                return &(*it);
            }
        }
        return nullptr;
    }

    const Graph_Node* getNode(const string& name) const {
         for (auto it = Pres_Nodes.begin(); it != Pres_Nodes.end(); ++it) {
            if (it->get_name() == name) {
                return &(*it);
            }
        }
        return nullptr;
    }

    // Modifier
    int addNode(Graph_Node node) {
        Pres_Nodes.push_back(node);
        nNodes++;
        return nNodes - 1; // Return the index of the added node
    }
};

// =========================================================================================
// HELPER FUNCTIONS
// =========================================================================================

/**
 * @brief Gets the index of a specific value string for a given node.
 * @param node The node to check.
 * @param value The value string (e.g., "True", "False").
 * @return The integer index of that value, or -1 if not found.
 */
int get_value_index(const Graph_Node& node, const string& value) {
    const vector<string>& values = node.get_values();
    for (size_t i = 0; i < values.size(); ++i) {
        if (values[i] == value) {
            return i;
        }
    }
    // cerr << "Error: Value '" << value << "' not found in node '" << node.get_name() << "'" << endl;
    return -1; // Not found
}


/**
 * @brief Reads a Bayesian Network from a .bif file.
 * @param filename The path to the .bif file.
 * @return A network object.
 */
network read_network(const string& filename) {
    network BayesNet;
    ifstream fin(filename);
    if (!fin) {
        cerr << "Error: Could not open file: " << filename << endl;
        return BayesNet;
    }

    string line;
    map<string, int> name_to_index;
    int node_counter = 0;

    // First pass: Read all 'variable' blocks to create nodes
    while (getline(fin, line)) {
        // Find variable declarations
        if (line.find("variable") == 0) {
            string name;
            stringstream ss(line);
            string temp;
            ss >> temp >> name; // temp="variable", name="NodeName"
            
            // Read the 'type discrete' line
            getline(fin, line);
            stringstream ss_type(line);
            string type_str, discrete_str, bracket_str, size_str;
            ss_type >> type_str >> discrete_str >> bracket_str >> size_str; // "type", "discrete", "[", "N", "]"
            int nvalues = stoi(size_str);

            string values_line;
            size_t start = line.find("{") + 1;
            size_t end = line.find("}");
            values_line = line.substr(start, end - start);
            
            stringstream ss_vals(values_line);
            string val;
            vector<string> values;
            while (ss_vals >> val) {
                if (val.back() == ',') val.pop_back(); // Remove trailing comma
                values.push_back(val);
            }
            
            // Create and add the node
            Graph_Node new_node(name, nvalues, values);
            int index = BayesNet.addNode(new_node);
            name_to_index[name] = index;
            node_counter++;
        }
        // Stop first pass when we hit probability definitions
        if (line.find("probability") == 0) {
            break;
        }
    }

    // Reset file stream to beginning to read probabilities
    fin.clear();
    fin.seekg(0, ios::beg);

    // Second pass: Read all 'probability' blocks to set parents and CPTs
    while (getline(fin, line)) {
        if (line.find("probability") == 0) {
            stringstream ss(line);
            string temp, node_name;
            
            ss >> temp; // Read "probability"
            ss >> temp; // Read "("
            
            if (temp != "(") {
                if (temp.front() == '(') {
                     temp.erase(0, 1); // Remove '('
                     node_name = temp;
                } else {
                    cerr << "Error: Expected '(' after 'probability' in line: " << line << endl;
                    continue;
                }
            } else {
                ss >> node_name; // Read "NodeName"
            }

            // Check for parents
            vector<string> parent_names;
            
            while (ss >> temp && temp != ")") { // Stop reading at the closing parenthesis
                if (temp == "|") continue;
                if (temp.back() == ',') temp.pop_back(); // remove trailing comma
                if (!temp.empty()) {
                    parent_names.push_back(temp);
                }
            }

            Graph_Node* node = BayesNet.getNode(node_name);
            if (!node) {
                cerr << "Error: Node '" << node_name << "' defined in probability but not in variable block." << endl;
                continue;
            }

            // Add parents to the node
            for (const string& parent_name : parent_names) {
                node->add_parent(parent_name);
                Graph_Node* parent_node = BayesNet.getNode(parent_name);
                if(parent_node) {
                    parent_node->add_child(name_to_index[node_name]);
                }
            }
            
            // Read CPT values
            vector<float> cpt_values;
            while (getline(fin, line) && line.find("}") != 0) {
                string values_str;
                if (line.find("table") != string::npos) {
                    // Read "table" line: table 0.1, 0.9;
                    size_t start = line.find("table") + 5;
                    values_str = line.substr(start);
                } else if (line.find("(") != string::npos) {
                    // Read "()" line: (Val1, Val2) 0.1, 0.9;
                    size_t start = line.find(")") + 1;
                    values_str = line.substr(start);
                } else {
                    continue;
                }
                
                stringstream ss_vals(values_str);
                float val;
                char comma_or_semicolon;

                while (ss_vals >> val) {
                    cpt_values.push_back(val);
                    if (ss_vals.peek() == ',') {
                        ss_vals >> comma_or_semicolon;
                    }
                }
            }
            node->set_CPT(cpt_values);
        }
    }

    fin.close();
    return BayesNet;
}


// =========================================================================================
// NEW SAMPLING FUNCTION
// =========================================================================================

/**
 * @brief Generates a single complete data sample from the network.
 * Assumes network nodes are in topological order (parents before children).
 * * @param bn The Bayesian network to sample from.
 * @param gen The random number generator.
 * @param name_to_index A map from node name to its index (0 to N-1).
 * @return A vector<string> representing one complete data record.
 */
vector<string> generate_sample(network& bn, mt19937& gen, map<string, int>& name_to_index) {
    int n = bn.netSize();
    vector<string> sample(n);

    // Iterate through nodes in topological order (assumed)
    for (int i = 0; i < n; ++i) {
        Graph_Node* node = bn.get_nth_node(i);
        const vector<string>& parents = node->get_Parents();
        const vector<float>& cpt = node->get_CPT();
        int n_values = node->get_nvalues();

        // 1. Find the CPT offset based on parent values from the sample
        int offset = 0;
        int multiplier = 1;

        const vector<string>& parent_names = node->get_Parents();
        for (auto it = parent_names.rbegin(); it != parent_names.rend(); ++it) {
            const string& parent_name = *it;
            Graph_Node* parent_node = bn.getNode(parent_name);
            int parent_idx_in_record = name_to_index[parent_name];
            const string& parent_val_str = sample[parent_idx_in_record];

            if (parent_val_str.empty()) {
                cerr << "Fatal Error: Sampling order incorrect. Parent " 
                     << parent_name << " not sampled before child " << node->get_name() << endl;
                exit(1);
            }
            
            int parent_val_idx = get_value_index(*parent_node, parent_val_str);
            
            offset += parent_val_idx * multiplier;
            multiplier *= parent_node->get_nvalues();
        }

        // 2. Get the probability distribution for the current node
        int cpt_start_index = offset * n_values;
        vector<float> probabilities;
        for (int v = 0; v < n_values; ++v) {
            probabilities.push_back(cpt[cpt_start_index + v]);
        }

        // 3. Sample from this distribution
        discrete_distribution<int> dist(probabilities.begin(), probabilities.end());
        int sampled_value_index = dist(gen);

        // 4. Store the sampled value string in our record
        sample[i] = node->get_values()[sampled_value_index];
    }

    return sample;
}


// =========================================================================================
// MAIN DRIVER PROGRAM
// =========================================================================================

int main() {
    // --- Configuration ---
    const int N_DATASETS_TO_GENERATE = 10000; // Number of data points to generate
    const string GOLD_NETWORK_FILE = "gold_hailfinder.bif";
    const string OUTPUT_DATA_FILE = "records.dat"; 
    // ---------------------

    cout << "========================================" << endl;
    cout << "         DATASET GENERATOR              " << endl;
    cout << "========================================" << endl;

    // Initialize random number generator
    random_device rd;
    mt19937 gen(rd());

    // 1. Load the gold standard network
    cout << "Loading gold standard network from: " << GOLD_NETWORK_FILE << "..." << endl;
    network gold_network = read_network(GOLD_NETWORK_FILE);
    if (gold_network.netSize() == 0) {
        cerr << "Fatal Error: Could not load gold network." << endl;
        return 1;
    }
    cout << "Loaded " << gold_network.netSize() << " nodes." << endl;

    // 2. Create the name-to-index map
    map<string, int> name_to_index;
    for (int i = 0; i < gold_network.netSize(); ++i) {
        name_to_index[gold_network.get_nth_node(i)->get_name()] = i;
    }

    // 3. Open output file
    ofstream fout(OUTPUT_DATA_FILE);
    if (!fout) {
        cerr << "Fatal Error: Could not open output file: " << OUTPUT_DATA_FILE << endl;
        return 1;
    }

    // 4. Generate and write samples
    cout << "Generating " << N_DATASETS_TO_GENERATE << " samples and writing to " << OUTPUT_DATA_FILE << "..." << endl;
    
    for (int i = 0; i < N_DATASETS_TO_GENERATE; ++i) {
        vector<string> sample = generate_sample(gold_network, gen, name_to_index);
        
        // Write the sample in the "records.dat" format
        for (size_t j = 0; j < sample.size(); ++j) {
            fout << "\"" << sample[j] << "\"";
            if (j < sample.size() - 1) {
                fout << ",";
            }
        }
        fout << "\n";

        // Print progress
        if ((i + 1) % 5000 == 0) {
            cout << "  ... " << (i + 1) << " samples generated." << endl;
        }
    }
    
    fout.close();

    cout << "\n========================================" << endl;
    cout << "    Data generation complete." << endl;
    cout << "    Total samples generated: " << N_DATASETS_TO_GENERATE << endl;
    cout << "    Output file: " << OUTPUT_DATA_FILE << endl;
    cout << "========================================" << endl;

    return 0;
}
