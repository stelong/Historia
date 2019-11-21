# Historia

This library contains the tools needed for building a multi-scale model of bi-ventricular rat heart electromechanics. It also provides routines for constructing a Bayesian surrogate of the model, and it implements History Matching (HM) technique to fit the model. Emulators used as a statistical representation of the real model are obtained as composition of a mean function given by a linear regression model and a zero-mean Gaussian process (GP) regressor. HM efficacy relies on the use of GP emulators, each one built to predict a single output feature to match. 

---
## Information

**Status**: `Occasionally maintained`

**Type**: `Personal project`

**Author**: [stelong](https://github.com/stelong)

---
## Getting Started

To get a copy of the project on your local machine, type the following in you shell:

```
git clone https://github.com/stelong/Historia.git
```

### Prerequisites

- diversipy
- matplotlib
- numpy
- Python 3
- scikit-learn
- scipy
- seaborn

---

## Contributing

Stefano Longobardi is the only mantainer. Any contribution is welcome.
