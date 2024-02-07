# Boosting Fair Classifier Generalization through Adaptive Priority Reweighing

This repository contains the implementation of the novel approach presented in the paper "Boosting Fair Classifier Generalization through Adaptive Priority Reweighing." This approach addresses the critical challenge of enhancing algorithmic fairness in machine learning applications, especially in decision-making domains. By introducing an adaptive reweighing method that effectively accounts for distribution shifts between training and test data, our method significantly improves the generalizability of fair classifiers. Unlike traditional reweighing methods that assign uniform weights to each (sub)group, our technique granularly adjusts weights based on the distance of sample predictions from the decision boundary, prioritizing those closer to it. This strategy ensures enhanced model performance across accuracy and fairness metrics such as equal opportunity, equalized odds, and demographic parity, demonstrated through extensive experiments on tabular benchmarks and applied to language and vision models. The full paper is available at [arXiv](https://arxiv.org/abs/2309.08375).
## Getting Started

To get this project up and running on your local machine, follow these steps.

### Prerequisites

Before you begin, ensure you have the following software installed on your system:

- Python 3.6 or newer
- scikit-learn
- pandas
### Data

This project uses the UCI Adult dataset, which is divided into two parts: `adult.data` for training and `adult.test` for testing. These files are included in the repository.

## Usage

This project provides scripts to improve the fairness of machine learning models using the UCI Adult dataset.
### Improving Fairness

To enhance your model's fairness on specific metrics, use one of the following scripts. Each script preprocesses the data, trains the model, and evaluates its fairness based on the chosen metric:

- **Equal Opportunity:**
  To improve Equal Opportunity fairness, invoke the following command:

```
python equal_opportunity.py --eta 1.2
```

- **Equalized Odds:**
For advancing fairness with respect to Equalized Odds, execute this command:

```
python equalized_odds.py --alpha 1000 --eta 1
```

- **Demographic Parity:**
To enhance Demographic Parity fairness, use the command below:

```
python demographic_parity.py --alpha 1000 --eta 2.8
```

### Parameters

- `--eta`: Represents the step size in the adaptive reweighing process. A higher value increases the weight assigned to samples that are closer to the decision boundary, emphasizing the model's sensitivity to these instances during training.
- `--alpha`: A constant that regulates the magnitude of the weight adjustment across all samples.

Experimenting with different `--eta` and `--alpha` values is encouraged to discover the optimal balance between accuracy and fairness for your model.

## Citation

```
@article{hu2023boosting,
  title={Boosting Fair Classifier Generalization through Adaptive Priority Reweighing},
  author={Hu, Zhihao and Xu, Yiran and Du, Mengnan and Gu, Jindong and Tian, Xinmei and He, Fengxiang},
  journal={arXiv e-prints},
  pages={arXiv--2309},
  year={2023}
}
```
