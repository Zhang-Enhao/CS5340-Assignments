""" CS5340 Lab 1: Belief Propagation and Maximal Probability
See accompanying PDF for instructions.

Name: ZHANG ENHAO
Email: e1132290@u.nus.edu
Student ID: A0276557M
"""

import copy
from typing import List

import numpy as np

from factor import Factor, index_to_assignment, assignment_to_index, generate_graph_from_factors, \
    visualize_graph


"""For sum product message passing"""
def factor_product(A, B):
    """Compute product of two factors.

    Suppose A = phi(X_1, X_2), B = phi(X_2, X_3), the function should return
    phi(X_1, X_2, X_3)
    """
    if A.is_empty():
        return B
    if B.is_empty():
        return A

    # Create output factor. Variables should be the union between of the
    # variables contained in the two input factors
    out = Factor()
    out.var = np.union1d(A.var, B.var)

    # Compute mapping between the variable ordering between the two factors
    # and the output to set the cardinality
    out.card = np.zeros(len(out.var), np.int64)
    mapA = np.argmax(out.var[None, :] == A.var[:, None], axis=-1)
    mapB = np.argmax(out.var[None, :] == B.var[:, None], axis=-1)
    out.card[mapA] = A.card
    out.card[mapB] = B.card

    # For each assignment in the output, compute which row of the input factors
    # it comes from
    out.val = np.zeros(np.prod(out.card))
    assignments = out.get_all_assignments()
    idxA = assignment_to_index(assignments[:, mapA], A.card)
    idxB = assignment_to_index(assignments[:, mapB], B.card)

    """ YOUR CODE HERE
    You should populate the .val field with the factor product
    Hint: The code for this function should be very short (~1 line). Try to
      understand what the above lines are doing, in order to implement
      subsequent parts.
    """
    out.val = A.val[idxA] * B.val[idxB]
    return out


def factor_marginalize(factor, var):
    """Sums over a list of variables.

    Args:
        factor (Factor): Input factor
        var (List): Variables to marginalize out

    Returns:
        out: Factor with variables in 'var' marginalized out.
    """
    out = Factor()

    """ YOUR CODE HERE
    Marginalize out the variables given in var
    """
    # Set var and card
    out.var = np.setdiff1d(factor.var, var)
    out.card = factor.card[np.isin(factor.var, out.var)]

    # compute val
    out.val = np.zeros(np.prod(out.card))
    all_assignments = factor.get_all_assignments()
    for i, assignment in enumerate(all_assignments):
        idx = assignment_to_index(assignment[np.isin(factor.var, out.var)], out.card)
        out.val[idx] += factor.val[i]

    return out


def observe_evidence(factors, evidence=None):
    """Modify a set of factors given some evidence

    Args:
        factors (List[Factor]): List of input factors
        evidence (Dict): Dictionary, where the keys are the observed variables
          and the values are the observed values.

    Returns:
        List of factors after observing evidence
    """
    if evidence is None:
        return factors
    out = copy.deepcopy(factors)

    """ YOUR CODE HERE
    Set the probabilities of assignments which are inconsistent with the 
    evidence to zero.
    """
    for factor in out:
        all_assignments = factor.get_all_assignments()
        for i, assignment in enumerate(all_assignments):
            for key, value in evidence.items():
                # Variables not in observation set to 0
                if assignment[np.isin(factor.var, key)] != value:
                    factor.val[i] = 0.0

    return out


"""For max sum meessage passing (for MAP)"""
def factor_sum(A, B):
    """Same as factor_product, but sums instead of multiplies
    """
    if A.is_empty():
        return B
    if B.is_empty():
        return A

    # Create output factor. Variables should be the union between of the
    # variables contained in the two input factors
    out = Factor()
    out.var = np.union1d(A.var, B.var)

    # Compute mapping between the variable ordering between the two factors
    # and the output to set the cardinality
    out.card = np.zeros(len(out.var), np.int64)
    mapA = np.argmax(out.var[None, :] == A.var[:, None], axis=-1)
    mapB = np.argmax(out.var[None, :] == B.var[:, None], axis=-1)
    out.card[mapA] = A.card
    out.card[mapB] = B.card

    # For each assignment in the output, compute which row of the input factors
    # it comes from
    out.val = np.zeros(np.prod(out.card))
    assignments = out.get_all_assignments()
    idxA = assignment_to_index(assignments[:, mapA], A.card)
    idxB = assignment_to_index(assignments[:, mapB], B.card)

    """ YOUR CODE HERE
    You should populate the .val field with the factor sum. The code for this
    should be very similar to the factor_product().
    """
    out.val = A.val[idxA] + B.val[idxB]
    return out


def factor_max_marginalize(factor, var):
    """Marginalize over a list of variables by taking the max.

    Args:
        factor (Factor): Input factor
        var (List): Variable to marginalize out.

    Returns:
        out: Factor with variables in 'var' marginalized out. The factor's
          .val_argmax field should be a list of dictionary that keep track
          of the maximizing values of the marginalized variables.
          e.g. when out.val_argmax[i][j] = k, this means that
            when assignments of out is index_to_assignment[i],
            variable j has a maximizing value of k.
          See test_lab1.py::test_factor_max_marginalize() for an example.
    """
    out = Factor()

    """ YOUR CODE HERE
    Marginalize out the variables given in var. 
    You should make use of val_argmax to keep track of the location with the
    maximum probability.
    """
    # Set variables
    out.var = np.setxor1d(factor.var, var)

    for o_var in out.var:
        out.card = np.append(out.card, factor.card[np.isin(factor.var, o_var)])

    assignment = factor.get_all_assignments()
    out.val_argmax = []

    #Delete the assignment corresponding to variables that are not in var
    mask = np.ones(assignment.shape[1], dtype=bool)
    for s_var in var:
        mask &= (factor.var != s_var)
    delete_assignment = assignment[:, mask]

    out.val = np.zeros(np.prod(out.card))
    var_indices = {var_name: np.where(factor.var == var_name)[0][0] for var_name in var}

    for unique_row in np.unique(delete_assignment, axis=0):
        # Find the indices where unique_row matches delete_assignment
        index_set = np.where(np.all(unique_row == delete_assignment, axis=1))[0]
        if index_set.size > 0:
            # Find the index with maximum value in factor.val
            max_index = index_set[np.argmax(factor.val[index_set])]
            single_assignment = assignment_to_index(unique_row, out.card)
            out.val[single_assignment] = factor.val[max_index]

            temp_dict = {var_name: assignment[max_index][var_indices[var_name]] for var_name in var}
            out.val_argmax.append(temp_dict)
    return out


def compute_joint_distribution(factors):
    """Computes the joint distribution defined by a list of given factors

    Args:
        factors (List[Factor]): List of factors

    Returns:
        Factor containing the joint distribution of the input factor list
    """
    joint = Factor()

    """ YOUR CODE HERE
    Compute the joint distribution from the list of factors. You may assume
    that the input factors are valid so no input checking is required.
    """
    for factor in factors:
        joint = factor_product(joint, factor)

    return joint


def compute_marginals_naive(V, factors, evidence):
    """Computes the marginal over a set of given variables

    Args:
        V (int): Single Variable to perform inference on
        factors (List[Factor]): List of factors representing the graphical model
        evidence (Dict): Observed evidence. evidence[k] = v indicates that
          variable k has the value v.

    Returns:
        Factor representing the marginals
    """

    output = Factor()

    """ YOUR CODE HERE
    Compute the marginal. Output should be a factor.
    Remember to normalize the probabilities!
    """
    # Compute joint distribution in factors
    joint_distribution = compute_joint_distribution(factors)
    variables = factors[0].var
    for factor_index in range(1, len(factors)):
        variables = np.union1d(variables, factors[factor_index].var)

    evidence_keys = list(evidence.keys())
    after_marginalize = factor_marginalize(joint_distribution, np.setxor1d(variables, np.append(evidence_keys, V)))

    new_output = observe_evidence([after_marginalize], evidence)[0]
    # Normalize
    normalize_sum = np.sum(new_output.val)
    new_output.val /= normalize_sum

    # Remove the evidence variable from new_output's variables, cardinalities, and values
    assignment = new_output.get_all_assignments()
    for evid_key, evid_value in evidence.items():
        delete_index = np.where(new_output.var == evid_key)[0][0]
        match_index = np.where(assignment[:, delete_index] != evid_value)[0]
        new_output.var = np.delete(new_output.var, delete_index)
        new_output.card = np.delete(new_output.card, delete_index)
        new_output.val = np.delete(new_output.val, match_index, axis=0)
        assignment = np.delete(assignment, match_index, axis=0)
        assignment = np.delete(assignment, delete_index, axis=1)

    output = new_output
    return output

def send_message(graph, node, parent, messages):
    node_nb = graph[node].keys()
    out = graph[parent][node]['factor']
    #Compute the product of incoming messages from all neighbors of j
    for next_node in node_nb:
        if next_node == parent:
            continue
        out = factor_product(out, messages[next_node][node])
    if graph.nodes[node]:
        out = factor_product(out, graph.nodes[node]['factor'])
    messages[node][parent] = factor_marginalize(out,[node])

    return messages

def collect(graph, parent, node,  messages):
    node_nb = graph[node].keys()
    #Recursively collect messages from all neighbors of j
    for next_node in node_nb:
        if next_node == parent:
            continue
        messages = collect(graph,node,next_node, messages)
    messages = send_message(graph,node,parent,messages)

    return messages

def distribute(graph,parent,node,messages):
    messages = send_message(graph, parent, node, messages)
    node_nb = graph[node].keys()
    # Recursively distribute messages from all neighbors of node
    for next_node in node_nb:
        if next_node == parent:
            continue
        messages = distribute(graph, node, next_node, messages)
    return messages

def compute_margin(graph, marginal_nodes, message):
    output = []
    for node in marginal_nodes:
        node_nb = graph[node]
        msg_product = Factor()
        for next_node in node_nb:
            msg_product = factor_product(msg_product, message[next_node][node])
        if graph.nodes[node]:
            msg_product = factor_product(msg_product, graph.nodes[node]['factor'])
        # Normalize the message
        msg_product.val /= np.sum(msg_product.val)
        output.append(msg_product)
    return output


def compute_marginals_bp(V, factors, evidence):
    """Compute single node marginals for multiple variables
    using sum-product belief propagation algorithm

    Args:
        V (List): Variables to infer single node marginals for
        factors (List[Factor]): List of factors representing the grpahical model
        evidence (Dict): Observed evidence. evidence[k]=v denotes that the
          variable k is assigned to value v.

    Returns:
        marginals: List of factors. The ordering of the factors should follow
          that of V, i.e. marginals[i] should be the factor for variable V[i].
    """
    # Dummy outputs, you should overwrite this with the correct factors
    marginals = []

    # Setting up messages which will be passed
    factors = observe_evidence(factors, evidence)
    graph = generate_graph_from_factors(factors)

    # Uncomment the following line to visualize the graph. Note that we create
    # an undirected graph regardless of the input graph since 1) this
    # facilitates graph traversal, and 2) the algorithm for undirected and
    # directed graphs is essentially the same for tree-like graphs.
    # visualize_graph(graph)

    # You can use any node as the root since the graph is a tree. For simplicity
    # we always use node 0 for this assignment.
    root = 0

    # Create structure to hold messages
    num_nodes = graph.number_of_nodes()
    messages = [[None] * num_nodes for _ in range(num_nodes)]


    """ YOUR CODE HERE
    Use the algorithm from lecture 4 and perform message passing over the entire
    graph. Recall the message passing protocol, that a node can only send a
    message to a neighboring node only when it has received messages from all
    its other neighbors.
    Since the provided graphical model is a tree, we can use a two-phase 
    approach. First we send messages inward from leaves towards the root.
    After this is done, we can send messages from the root node outward.

    Hint: You might find it useful to add auxilliary functions. You may add 
      them as either inner (nested) or external functions.
    """
    #collect messages from all neighboring nodes of the root
    root_nb = graph[root].keys()
    for node in root_nb:
        messages = collect(graph, root, node, messages)
    for node in root_nb:
        messages = distribute(graph, root, node, messages)
    marginals = compute_margin(graph, V, messages)
    return marginals


def map_sendmessage(graph, node, parent, prob, conf):
    node_nb = graph[node].keys()
    out = graph[parent][node]['factor']
    # Compute the sum of incoming messages from all neighbors of node
    for next_node in node_nb:
        if next_node == parent:
            continue
        out = factor_sum(out, prob[next_node][node])
    if graph.nodes[node]:
        out = factor_sum(out, graph.nodes[node]['factor'])
    prob[node][parent] = factor_max_marginalize(out, [node])

    conf[node][parent] = {}
    assignments = out.get_all_assignments()
    parent_index = np.where(out.var == parent)[0][0]
    node_index = np.where(out.var == node)[0][0]
    for index in range(prob[node][parent].card[0]):
        mask = (assignments[:, parent_index] == index)
        # Get the exact value of variable node corresponding to the max probability
        max_prob_index = np.argmax(out.val[mask])
        j_exact_value = assignments[mask][max_prob_index][node_index]
        conf[node][parent][index] = j_exact_value

    return prob, conf


def map_collect(graph, parent, node, prob, conf):
    node_nb = graph[node]
    # Recursively collect messages from all neighbors of node
    for next_node in node_nb:
        if next_node == parent:
            continue
        prob, conf = map_collect(graph, node, next_node, prob, conf)
    # Send the message from node node to node parent and update prob and conf
    prob, conf = map_sendmessage(graph, node, parent, prob, conf)

    return prob, conf


def map_distribute(graph, parent, node, conf, max_decoding):
    # Update max_decoding for nodes along the path from parent to node
    max_decoding[node] = conf[node][parent][max_decoding[parent]]
    node_nb = graph[node]
    # Recursively update max_decoding for neighbors of node
    for next_node in node_nb:
        if next_node == parent:
            continue
        max_decoding = map_distribute(graph, node, next_node, conf, max_decoding)
    return max_decoding




def map_eliminate(factors, evidence):
    """Obtains the maximum a posteriori configuration for a tree graph
    given optional evidence

    Args:
        factors (List[Factor]): List of factors representing the graphical model
        evidence (Dict): Observed evidence. evidence[k]=v denotes that the
          variable k is assigned to value v.

    Returns:
        max_decoding (Dict): MAP configuration
        log_prob_max: Log probability of MAP configuration. Note that this is
          log p(MAP, e) instead of p(MAP|e), i.e. it is the unnormalized
          representation of the conditional probability.
    """

    max_decoding = {}
    log_prob_max = 0.0

    """ YOUR CODE HERE
    Use the algorithm from lecture 5 and perform message passing over the entire
    graph to obtain the MAP configuration. Again, recall the message passing 
    protocol.
    Your code should be similar to compute_marginals_bp().
    To avoid underflow, first transform the factors in the probabilities
    to **log scale** and perform all operations on log scale instead.
    You may ignore the warning for taking log of zero, that is the desired
    behavior.
    """
    factors = observe_evidence(factors, evidence)
    for factor in factors:
        #Avoiding zero logarithmic variables, an eps was added
        factor.val = np.log(factor.val + np.finfo(float).eps)
    graph = generate_graph_from_factors(factors)

    num_nodes = graph.number_of_nodes()
    prob = [[None] * num_nodes for _ in range(num_nodes)]
    conf = [[None] * num_nodes for _ in range(num_nodes)]
    root = 0

    # collect messages from neighboring nodes
    root_nb = graph[root]
    for node in root_nb:
        prob, conf = map_collect(graph, root, node, prob, conf)
    # Compute the sum of collected probabilities
    prob_sum = Factor()
    for node in root_nb:
        prob_sum = factor_sum(prob_sum, prob[node][root])
    if graph.nodes[root]:
        prob_sum = factor_sum(graph.nodes[root]['factor'], prob_sum)
    # Find the maximum log probability and its corresponding assignment
    log_prob_max = np.max(prob_sum.val)
    max_decoding[root] = np.argmax(prob_sum.val)
    # distribute updated information to neighboring nodes
    for node in root_nb:
        max_decoding = map_distribute(graph, root, node, conf, max_decoding)
    # Remove evidence variables from the max_decoding dictionary
    for key in evidence.keys():
        max_decoding.pop(key, None)
    return max_decoding, log_prob_max