""" CS5340 Lab 2 Part 1: Junction Tree Algorithm
See accompanying PDF for instructions.

Name: ZHANG ENHAO
Email: e1132290@u.nus.edu
Student ID: A0276557M
"""

import os
import numpy as np
import json
import networkx as nx
from argparse import ArgumentParser

from factor import Factor
from jt_construction import construct_junction_tree
from factor_utils import factor_product, factor_evidence, factor_marginalize

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'inputs')  # we will store the input data files here!
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')  # we will store the prediction files here!


""" ADD HELPER FUNCTIONS HERE """
def get_neighbors(jt_edges, node):
    """
    Retrieve the neighbors of a given node in a junction tree.

    Args:
    - jt_edges: List of edges in the junction tree.
    - node: The node for which neighbors are to be retrieved.

    Returns:
    - List of neighbors for the given node.
    """
    return [edge[1] if edge[0] == node else edge[0] for edge in jt_edges if node in edge]

def send_message(jt_cliques, jt_edges, jt_clique_factors, node, parent, messages, neighbors=None):
    """
    Send a message from a node to its parent in a junction tree.

    Args:
    - jt_cliques: List of nodes in the junction tree.
    - jt_edges: List of edges in the junction tree.
    - jt_clique_factors: List of factors associated with each clique in the junction tree.
    - node: The node sending the message.
    - parent: The node receiving the message.
    - messages: Current messages in the junction tree.
    - neighbors: Neighbors of the node (optional).

    Returns:
    - Updated messages after sending.
    """
    if neighbors is None:
        neighbors = get_neighbors(jt_edges, node)
        if parent in neighbors:
            neighbors.remove(parent)

    combined_factor = jt_clique_factors[node]
    for neighbor in neighbors:
        combined_factor = factor_product(combined_factor, messages[neighbor][node])

    sep_set = set(jt_cliques[parent]).intersection(jt_cliques[node])
    marginal_vars = list(set(jt_cliques[node]) - sep_set)
    messages[node][parent] = factor_marginalize(combined_factor, marginal_vars)

    return messages

def collect(jt_cliques, jt_edges, jt_clique_factors, parent, node, messages):
    """
    Recursive function for the collect phase of the sum-product algorithm in a junction tree.

    Args:
    - jt_cliques, jt_edges, jt_clique_factors: Junction tree components.
    - parent: Parent node in the current context.
    - node: Current node being processed.
    - messages: Current messages in the junction tree.

    Returns:
    - Updated messages after the collect phase.
    """
    neighbors = get_neighbors(jt_edges, node)
    if parent in neighbors:
        neighbors.remove(parent)

    for neighbor in neighbors:
        messages = collect(jt_cliques, jt_edges, jt_clique_factors, node, neighbor, messages)
    messages = send_message(jt_cliques, jt_edges, jt_clique_factors, node, parent, messages, neighbors)

    return messages

def distribute(jt_cliques, jt_edges, jt_clique_factors, parent, node, msg):
    msg = send_message(jt_cliques, jt_edges, jt_clique_factors, parent, node, msg)
    neighbors = get_neighbors(jt_edges, node)
    if parent in neighbors:
        neighbors.remove(parent)
    for neighbor in neighbors:
        msg = distribute(jt_cliques, jt_edges, jt_clique_factors, node, neighbor, msg)
    return msg

def compute_clique_marginals(jt_cliques, jt_edges, jt_clique_factors, messages):
    """
    Compute the marginal probabilities for each clique in the junction tree.

    Args:
    - jt_cliques, jt_edges, jt_clique_factors: Junction tree components.
    - messages: Messages in the junction tree.

    Returns:
    - List of computed marginals for each clique.
    """
    marginals = []
    for idx, clique in enumerate(jt_cliques):
        msg_product = Factor()

        for neighbor in get_neighbors(jt_edges, idx):
            msg_product = factor_product(msg_product, messages[neighbor][idx])

        combined = factor_product(jt_clique_factors[idx], msg_product)
        combined.val /= np.sum(combined.val)
        marginals.append(combined)
    return marginals

""" END HELPER FUNCTIONS HERE """


def _update_mrf_w_evidence(all_nodes, evidence, edges, factors):
    """
    Update the MRF graph structure from observing the evidence

    Args:
        all_nodes: numpy array of nodes in the MRF
        evidence: dictionary of node:observation pairs where evidence[x1] returns the observed value of x1
        edges: numpy array of edges in the MRF
        factors: list of Factors in teh MRF

    Returns:
        numpy array of query nodes
        numpy array of updated edges (after observing evidence)
        list of Factors (after observing evidence; empty factors should be removed)
    """

    query_nodes = all_nodes
    updated_edges = edges
    updated_factors = factors

    """ YOUR CODE HERE """
    # Update factors based on evidence
    updated_factors = [factor_evidence(factor, evidence) for factor in factors]

    # Remove evidence nodes from query nodes and edges
    for evi, evi_val in evidence.items():
        query_nodes = np.setdiff1d(query_nodes, evi)
        updated_edges = [edge for edge in updated_edges if evi not in edge]

    """ END YOUR CODE HERE """

    return query_nodes, updated_edges, updated_factors


def _get_clique_potentials(jt_cliques, jt_edges, jt_clique_factors):
    """
    Returns the list of clique potentials after performing the sum-product algorithm on the junction tree

    Args:
        jt_cliques: list of junction tree nodes e.g. [[x1, x2], ...]
        jt_edges: numpy array of junction tree edges e.g. [i,j] implies that jt_cliques[i] and jt_cliques[j] are
                neighbors
        jt_clique_factors: list of clique factors where jt_clique_factors[i] is the factor for cliques[i]

    Returns:
        list of clique potentials computed from the sum-product algorithm
    """
    clique_potentials = jt_clique_factors

    """ YOUR CODE HERE """
    # Initialization
    root = 0
    num_cliques = len(jt_cliques)
    msg = [[None] * num_cliques for _ in range(num_cliques)]

    # Perform Collect phase
    root_neighbors = get_neighbors(jt_edges, root)
    for neighbor in root_neighbors:
        msg = collect(jt_cliques, jt_edges, jt_clique_factors, root, neighbor, msg)

    # Perform Distribute phase
    for neighbor in root_neighbors:
        msg = distribute(jt_cliques, jt_edges, jt_clique_factors, root, neighbor, msg)

    # Compute clique potentials
    clique_potentials = compute_clique_marginals(jt_cliques, jt_edges, jt_clique_factors, msg)

    """ END YOUR CODE HERE """

    assert len(clique_potentials) == len(jt_cliques)
    return clique_potentials


def _get_node_marginal_probabilities(query_nodes, cliques, clique_potentials):
    """
    Returns the marginal probability for each query node from the clique potentials.

    Args:
        query_nodes: numpy array of query nodes e.g. [x1, x2, ..., xN]
        cliques: list of cliques e.g. [[x1, x2], ... [x2, x3, .., xN]]
        clique_potentials: list of clique potentials (Factor class)

    Returns:
        list of node marginal probabilities (Factor class)

    """
    query_marginal_probabilities = []

    """ YOUR CODE HERE """
    # Compute node marginal probabilities
    for node in query_nodes:
        smallest_clique_size = float('inf')
        smallest_clique_idx = None
        for idx, clique in enumerate(cliques):
            if node in clique and len(clique) < smallest_clique_size:
                smallest_clique_size = len(clique)
                smallest_clique_idx = idx

        to_marginalize = [var for var in cliques[smallest_clique_idx] if var != node]
        query_marginal_probabilities.append(factor_marginalize(clique_potentials[smallest_clique_idx], to_marginalize))

    """ END YOUR CODE HERE """

    return query_marginal_probabilities


def get_conditional_probabilities(all_nodes, evidence, edges, factors):
    """
    Returns query nodes and query Factors representing the conditional probability of each query node
    given the evidence e.g. p(xf|Xe) where xf is a single query node and Xe is the set of evidence nodes.

    Args:
        all_nodes: numpy array of all nodes (random variables) in the graph
        evidence: dictionary of node:evidence pairs e.g. evidence[x1] returns the observed value for x1
        edges: numpy array of all edges in the graph e.g. [[x1, x2],...] implies that x1 is a neighbor of x2
        factors: list of factors in the MRF.

    Returns:
        numpy array of query nodes
        list of Factor
    """
    query_nodes, updated_edges, updated_node_factors = _update_mrf_w_evidence(all_nodes=all_nodes, evidence=evidence,
                                                                              edges=edges, factors=factors)

    jt_cliques, jt_edges, jt_factors = construct_junction_tree(nodes=query_nodes, edges=updated_edges,
                                                               factors=updated_node_factors)

    clique_potentials = _get_clique_potentials(jt_cliques=jt_cliques, jt_edges=jt_edges, jt_clique_factors=jt_factors)

    query_node_marginals = _get_node_marginal_probabilities(query_nodes=query_nodes, cliques=jt_cliques,
                                                            clique_potentials=clique_potentials)

    return query_nodes, query_node_marginals


def parse_input_file(input_file: str):
    """ Reads the input file and parses it. DO NOT EDIT THIS FUNCTION. """
    with open(input_file, 'r') as f:
        input_config = json.load(f)

    nodes = np.array(input_config['nodes'])
    edges = np.array(input_config['edges'])

    # parse evidence
    raw_evidence = input_config['evidence']
    evidence = {}
    for k, v in raw_evidence.items():
        evidence[int(k)] = v

    # parse factors
    raw_factors = input_config['factors']
    factors = []
    for raw_factor in raw_factors:
        factor = Factor(var=np.array(raw_factor['var']), card=np.array(raw_factor['card']),
                        val=np.array(raw_factor['val']))
        factors.append(factor)
    return nodes, edges, evidence, factors


def main():
    """ Entry function to handle loading inputs and saving outputs. DO NOT EDIT THIS FUNCTION. """
    argparser = ArgumentParser()
    argparser.add_argument('--case', type=int, required=True,
                           help='case number to create observations e.g. 1 if 1.json')
    args = argparser.parse_args()

    case = args.case
    input_file = os.path.join(INPUT_DIR, '{}.json'.format(case))
    nodes, edges, evidence, factors = parse_input_file(input_file=input_file)

    # solution part:
    query_nodes, query_conditional_probabilities = get_conditional_probabilities(all_nodes=nodes, edges=edges,
                                                                                 factors=factors, evidence=evidence)

    predictions = {}
    for i, node in enumerate(query_nodes):
        probability = query_conditional_probabilities[i].val
        predictions[int(node)] = list(np.array(probability, dtype=float))

    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)
    prediction_file = os.path.join(PREDICTION_DIR, '{}.json'.format(case))
    with open(prediction_file, 'w') as f:
        json.dump(predictions, f, indent=1)
    print('INFO: Results for test case {} are stored in {}'.format(case, prediction_file))


if __name__ == '__main__':
    main()
