import argparse
import numpy as np
import logging
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from load_data import load_dataset
from evaluation import *
import pdb


def set_seed(seed=0):
    """Set seed for reproducibility."""
    np.random.seed(seed)


def initialize_logger(args):
    """Initialize the logger for experiment tracking."""
    log_format = '%(asctime)s %(name)s:%(levelname)s:  %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    log_file = f'{args.save_dir}/{args.save_name}_log.txt'
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    logger.info('Arguments')
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')

    return logger

def compute_sample_weights(true_labels, predicted_probabilities, protected_attributes, multipliers, eta, decision_boundary=0.5):
    num_samples = len(true_labels)
    exponential_term = np.exp(-eta * abs(predicted_probabilities - decision_boundary))
    weight_component = np.zeros(num_samples)

    for attr in protected_attributes:
        weight_component += exponential_term * np.sum(attr) / np.sum(exponential_term * attr) * attr

    combined_weights = np.zeros(num_samples)
    for i, multiplier in enumerate(multipliers):
        combined_weights += multiplier * protected_attributes[i]
    
    sample_weights = combined_weights * weight_component

    return sample_weights

def compute_group_weights(predictions, true_labels, protected_attributes, alpha):
    group_weights = []
    
    num_samples = len(true_labels)
    
    for attr in protected_attributes:
        protected_indices = np.where(attr > 0)
        
        positive_protected_prediction = np.sum(predictions[protected_indices])
        negative_protected_prediction = np.sum(1 - predictions[protected_indices])

        # Calculate weights for positive and negative protected predictions
        weight_positive = (len(protected_indices[0]) * np.sum(predictions) + alpha) / (num_samples * positive_protected_prediction)
        weight_negative = (len(protected_indices[0]) * np.sum(1 - predictions) + alpha) / (num_samples * negative_protected_prediction)
        
        group_weights.extend([weight_positive, weight_negative])

    return group_weights


def main(args):
    """Main function for the fairness experiment."""
    logger = initialize_logger(args)
    features_train, features_test, labels_train, labels_test, protected_attribute_train, protected_attribute_test = load_dataset()

    protected_attributes_list = [protected_attribute_train, 1 - protected_attribute_train]
    label_combinations = [protected_attribute_train * labels_train, protected_attribute_train * (1 - labels_train), \
        (1 - protected_attribute_train) * labels_train, (1 - protected_attribute_train) * (1 - labels_train)]
    fairness_multipliers = np.ones(len(label_combinations))
    sample_weights = np.array([1] * features_train.shape[0])

    for epoch in range(args.epoch):
        # Train logistic regression model
        classifier = LogisticRegression(max_iter=10000)
        classifier.fit(features_train, labels_train, sample_weights)
        
        predictions_train = classifier.predict(features_train)
        prediction_probabilities = classifier.predict_proba(features_train)[:, 0].astype(np.float32)

        # Compute weights and multipliers
        sample_weights = compute_sample_weights(labels_train, prediction_probabilities, label_combinations, fairness_multipliers, args.eta)
        group_fairness_weights = compute_group_weights(predictions_train, labels_train, protected_attributes_list, args.alpha)
        fairness_multipliers *= np.array(group_fairness_weights)

        # Evaluate model on test dataset
        predictions_test = np.squeeze(classifier.predict(features_test))
        test_accuracy = accuracy_score(labels_test, predictions_test)
        
        delta_equal_opportunity = equalized_odds_difference(predictions_test, labels_test, protected_attribute_test, target_label=1)
        delta_equalized_odds_negative = equalized_odds_difference(predictions_test, labels_test, protected_attribute_test, target_label=0)
        delta_demographic_parity = demographic_parity_difference(predictions_test, protected_attribute_test)

        delta_equalized_odds = max(delta_equal_opportunity, delta_equalized_odds_negative)

        # Log results
        log_message = (f'Epoch {epoch + 1}\t delta_equalized_odds {delta_equalized_odds:.2%}\t delta_equal_opportunity {delta_equal_opportunity:.2%}'
                       f'\t delta_demographic_parity {delta_demographic_parity:.2%}\t test_accuracy {test_accuracy:.2%}')
        
        logger.info(log_message)
        print(log_message)



def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--alpha', type=int, default=0)
    parser.add_argument('--eta', type=float, default=0)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--save-dir', type=str, default='./temp')
    parser.add_argument('--save-name', type=str, default='temp-name')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    set_seed()
    try:
        main(args)
    except Exception as e:
        logging.exception('Unexpected exception! %s', e)