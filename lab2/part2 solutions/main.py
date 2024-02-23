""" CS5340 Lab 2 Part 2: Parameter Learning
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

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')  # we will store the input data files here!
OBSERVATION_DIR = os.path.join(DATA_DIR, 'observations')
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')


""" ADD HELPER FUNCTIONS HERE """

""" D ADD HELPER FUNCTIONS HERE """


def _learn_node_parameter_w(outputs, inputs=None):
    """
    Returns the weight parameters of the linear Gaussian [w0, w1, ..., wI], where I is the number of inputs. Students
    are encouraged to use numpy.linalg.solve() to get the weights. Learns weights for one node only.
    Call once for each node.

    Args:
        outputs: numpy array of N output observations of the node
        inputs: N x I numpy array of input observations to the linear Gaussian model

    Returns:
        numpy array of (I + 1) weights [w0, w1, ..., wI]
    """
    num_inputs = 0 if inputs is None else inputs.shape[1]
    weights = np.zeros(shape=num_inputs + 1)

    """ YOUR CODE HERE """
    N = len(outputs)

    # Constructing the augmented inputs with a column of ones for w0 (intercept)
    augmented_inputs = np.ones((N, num_inputs + 1))
    if inputs is not None:
        augmented_inputs[:, 1:] = inputs

    # Compute matrix A using the augmented inputs
    A = augmented_inputs.T @ augmented_inputs

    # Compute vector b using the augmented inputs and outputs
    b = augmented_inputs.T @ outputs

    # Solve the system Ax = b to find the weights
    weights = np.linalg.solve(A, b)
    """ END YOUR CODE HERE """

    return weights


def _learn_node_parameter_var(outputs, weights, inputs):
    """
    Returns the variance i.e. sigma^2 for the node. Learns variance for one node only. Call once for each node.

    Args:
        outputs: numpy array of N output observations of the node
        weights: numpy array of (I + 1) weights of the linear Gaussian model
        inputs:  N x I numpy array of input observations to the linear Gaussian model.

    Returns:
        variance of the node's Linear Gaussian model
    """
    var = 0.

    """ YOUR CODE HERE """
    if inputs is None:
        residuals = outputs - weights[0]
    else:
        # Calculate the estimated outputs using weights and inputs
        estimated_outputs = inputs @ weights[1:] + weights[0]
        # Calculate the residuals between the observed and estimated outputs
        residuals = outputs - estimated_outputs

    # Compute the variance
    var = (residuals ** 2).mean()

    """ END YOUR CODE HERE """

    return var


def _get_learned_parameters(nodes, edges, observations):
    """
    Learns the parameters for each node in nodes and returns the parameters as a dictionary. The nodes are given in
    ascending numerical order e.g. [1, 2, ..., V]

    Args:
        nodes: numpy array V nodes in the graph e.g. [1, 2, 3, ..., V]
        edges: numpy array of edges in the graph e.g. [i, j] implies i -> j where i is the parent of j
        observations: dictionary of node: observations pair where observations[1] returns a list of
                    observations for node 1.

    Returns:
        dictionary of parameters e.g.
        parameters = {
            "1": {  // first node
                "bias": w0 weight for node "1",
                "variance": variance for node "1"

                "2": weight for node "2", who is the parent of "1"
                ...
                // weights for other parents of "1"
            },
            ...
            // parameters of other nodes.
        }
    """
    parameters = {}

    """ YOUR CODE HERE """
    # Iterate over each node to get its parameters
    for node in nodes:

        # Identify parent nodes for the current node based on the edges
        parent_nodes = [edge[0] for edge in edges if edge[1] == node]

        # Extract output observations for the current node
        output_observations = np.array(observations[node])

        # If node has parents, collect their observations
        if parent_nodes:
            input_observations = np.array([observations[p_node] for p_node in parent_nodes]).T
        else:
            input_observations = None

        # Learn weights and variance for the node
        node_weights = _learn_node_parameter_w(output_observations, input_observations)
        node_variance = _learn_node_parameter_var(output_observations, node_weights, input_observations)

        # Store learned parameters in the dictionary
        node_parameters = {
            "variance": node_variance,
            "bias": node_weights[0]
        }
        for idx, p_node in enumerate(parent_nodes):
            node_parameters[p_node] = node_weights[idx + 1]
        parameters[node] = node_parameters
    """ END YOUR CODE HERE """

    return parameters


def main():
    """
    Helper function to load the observations, call your parameter learning function and save your results.
    DO NOT EDIT THIS FUNCTION.
    """
    argparser = ArgumentParser()
    argparser.add_argument('--case', type=int, required=True,
                           help='case number to create observations e.g. 1 if 1.json')
    args = argparser.parse_args()

    case = args.case
    observation_file = os.path.join(OBSERVATION_DIR, '{}.json'.format(case))
    with open(observation_file, 'r') as f:
         observation_config = json.load(f)

    nodes = observation_config['nodes']
    edges = observation_config['edges']
    observations = observation_config['observations']

    # solution part
    parameters = _get_learned_parameters(nodes=nodes, edges=edges, observations=observations)
    # end solution part

    # json only recognises floats, not np.float, so we need to cast the values into floats.
    for node, node_params in parameters.items():
        for param, val in node_params.items():
            node_params[param] = float(val)
        parameters[node] = node_params

    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)
    prediction_file = os.path.join(PREDICTION_DIR, '{}.json'.format(case))

    with open(prediction_file, 'w') as f:
        json.dump(parameters, f, indent=1)
    print('INFO: Results for test case {} are stored in {}'.format(case, prediction_file))


if __name__ == '__main__':
    main()
