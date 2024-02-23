""" CS5340 Lab 4 Part 1: Importance Sampling
See accompanying PDF for instructions.

Name: Zhang Enhao
Email: e1132290@u.nus.edu
Student ID: A0276657M
"""

import os
import json
from functools import reduce

import numpy as np
import networkx as nx
from factor_utils import factor_evidence, factor_product, assignment_to_index
from factor import Factor, index_to_assignment
from argparse import ArgumentParser
from tqdm import tqdm

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'inputs')
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')


""" ADD HELPER FUNCTIONS HERE """


""" END HELPER FUNCTIONS HERE """


def _sample_step(nodes, proposal_factors):
    """
    Performs one iteration of importance sampling where it should sample a sample for each node. The sampling should
    be done in topological order.

    Args:
        nodes: numpy array of nodes. nodes are sampled in the order specified in nodes
        proposal_factors: dictionary of proposal factors where proposal_factors[1] returns the
                sample distribution for node 1

    Returns:
        dictionary of node samples where samples[1] return the scalar sample for node 1.
    """
    samples = {}

    """ YOUR CODE HERE: Use np.random.choice """
    evidence = {}

    for node in nodes:
        conditioned_factor = factor_evidence(proposal_factors[node], evidence)
        sampled_value = np.random.choice(a=conditioned_factor.card[0], p=conditioned_factor.val)
        samples[node] = sampled_value
        evidence[node] = sampled_value
    """ END YOUR CODE HERE """

    assert len(samples.keys()) == len(nodes)
    return samples


def _get_conditional_probability(target_factors, proposal_factors, evidence, num_iterations):
    """
    Performs multiple iterations of importance sampling and returns the conditional distribution p(Xf | Xe) where
    Xe are the evidence nodes and Xf are the query nodes (unobserved).

    Args:
        target_factors: dictionary of node:Factor pair where Factor is the target distribution of the node.
                        Other nodes in the Factor are parent nodes of the node. The product of the target
                        distribution gives our joint target distribution.
        proposal_factors: dictionary of node:Factor pair where Factor is the proposal distribution to sample node
                        observations. Other nodes in the Factor are parent nodes of the node
        evidence: dictionary of node:val pair where node is an evidence node while val is the evidence for the node.
        num_iterations: number of importance sampling iterations

    Returns:
        Approximate conditional distribution of p(Xf | Xe) where Xf is the set of query nodes (not observed) and
        Xe is the set of evidence nodes. Return result as a Factor
    """
    out = Factor()

    """ YOUR CODE HERE """
    # Combine target factors into a single factor, accounting for evidence
    combined_target = reduce(lambda acc_factor, next_factor: factor_product(acc_factor, next_factor),
                             target_factors.values(), Factor())
    out = factor_evidence(combined_target, evidence)

    # Prepare proposal factors, excluding evidence nodes
    non_evidence_proposal_factors = {node: factor_evidence(factor, evidence)
                                     for node, factor in proposal_factors.items()
                                     if node not in evidence}

    # Get a list of non-evidence nodes
    non_evidence_nodes = list(non_evidence_proposal_factors.keys())

    # Number of configurations for the combined target factor
    config_count = len(out.val)
    importance_weights = np.zeros(config_count)
    sample_counts = np.zeros(config_count)

    # Calculate importance weights for each configuration
    for config_idx in range(config_count):
        config_assignment = index_to_assignment(config_idx, out.card)
        assignment_dict = dict(zip(non_evidence_nodes, config_assignment))

        # Calculate the proposal probability (q) for this configuration
        proposal_prob = np.prod([non_evidence_proposal_factors[node].val[
                                     assignment_to_index(
                                         [assignment_dict[var] for var in non_evidence_proposal_factors[node].var],
                                         non_evidence_proposal_factors[node].card)]
                                 for node in non_evidence_nodes])

        # Calculate the target probability (p) for this configuration
        target_prob = np.prod([target_factors[node].val[assignment_to_index(
            [evidence.get(var, assignment_dict.get(var)) for var in target_factors[node].var],
            target_factors[node].card)]
                               for node in target_factors])

        # Compute the importance weight for this configuration
        importance_weights[config_idx] = target_prob / proposal_prob

    # Perform importance sampling iterations
    total_weight_sum = 0
    for _ in range(num_iterations):
        sample_values = _sample_step(non_evidence_nodes, non_evidence_proposal_factors)
        sample_config = [sample_values[node] for node in non_evidence_nodes]
        config_index = assignment_to_index(sample_config, out.card)
        sample_counts[config_index] += 1
        total_weight_sum += importance_weights[config_index]

    # Compute the conditional probability distribution
    out.val = sample_counts * importance_weights / total_weight_sum

    """ END YOUR CODE HERE """

    return out


def load_input_file(input_file: str) -> (Factor, dict, dict, int):
    """
    Returns the target factor, proposal factors for each node and evidence. DO NOT EDIT THIS FUNCTION

    Args:
        input_file: input file to open

    Returns:
        Factor of the target factor which is the target joint distribution of all nodes in the Bayesian network
        dictionary of node:Factor pair where Factor is the proposal distribution to sample node observations. Other
                    nodes in the Factor are parent nodes of the node
        dictionary of node:val pair where node is an evidence node while val is the evidence for the node.
    """
    with open(input_file, 'r') as f:
        input_config = json.load(f)
    target_factors_dict = input_config['target-factors']
    proposal_factors_dict = input_config['proposal-factors']
    assert isinstance(target_factors_dict, dict) and isinstance(proposal_factors_dict, dict)

    def parse_factor_dict(factor_dict):
        var = np.array(factor_dict['var'])
        card = np.array(factor_dict['card'])
        val = np.array(factor_dict['val'])
        return Factor(var=var, card=card, val=val)

    target_factors = {int(node): parse_factor_dict(factor_dict=target_factor) for
                      node, target_factor in target_factors_dict.items()}
    proposal_factors = {int(node): parse_factor_dict(factor_dict=proposal_factor_dict) for
                        node, proposal_factor_dict in proposal_factors_dict.items()}
    evidence = input_config['evidence']
    evidence = {int(node): ev for node, ev in evidence.items()}
    num_iterations = input_config['num-iterations']
    return target_factors, proposal_factors, evidence, num_iterations


def main():
    """
    Helper function to load the observations, call your parameter learning function and save your results.
    DO NOT EDIT THIS FUNCTION.
    """
    argparser = ArgumentParser()
    argparser.add_argument('--case', type=int, required=True,
                           help='case number to create observations e.g. 1 if 1.json')
    args = argparser.parse_args()
    # np.random.seed(0)

    case = args.case
    input_file = os.path.join(INPUT_DIR, '{}.json'.format(case))
    target_factors, proposal_factors, evidence, num_iterations = load_input_file(input_file=input_file)

    # solution part
    conditional_probability = _get_conditional_probability(target_factors=target_factors,
                                                           proposal_factors=proposal_factors,
                                                           evidence=evidence, num_iterations=num_iterations)
    print(conditional_probability)
    # end solution part

    # json only recognises floats, not np.float, so we need to cast the values into floats.
    save__dict = {
        'var': np.array(conditional_probability.var).astype(int).tolist(),
        'card': np.array(conditional_probability.card).astype(int).tolist(),
        'val': np.array(conditional_probability.val).astype(float).tolist()
    }

    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)
    prediction_file = os.path.join(PREDICTION_DIR, '{}.json'.format(case))

    with open(prediction_file, 'w') as f:
        json.dump(save__dict, f, indent=1)
    print('INFO: Results for test case {} are stored in {}'.format(case, prediction_file))


if __name__ == '__main__':
    main()
