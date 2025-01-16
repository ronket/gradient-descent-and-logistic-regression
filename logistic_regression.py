from typing import NoReturn
import numpy as np
from base_estimator import BaseEstimator
from gradient_descent import GradientDescent
from modules import L1, L2 ,LogisticModule, RegularizedModule



class LogisticRegression(BaseEstimator):
    """
    Logistic Regression Classifier

    Attributes
    ----------
    solver_: GradientDescent, default=GradientDescent()
        Descent method solver to use for the logistic regression objective optimization

    penalty_: str, default="none"
        Type of regularization term to add to logistic regression objective. Supported values
        are "none", "l1", "l2"

    lam_: float, default=1
        Regularization parameter to be used in case `self.penalty_` is not "none"

    alpha_: float, default=0.5
        Threshold value by which to convert class probability to class value

    include_intercept_: bool, default=True
        Should fitted model include an intercept or not

    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by linear regression. To be set in
        `LogisticRegression.fit` function.
    """

    def __init__(self,
                 include_intercept: bool = True,
                 solver: GradientDescent = GradientDescent(),
                 penalty: str = "none",
                 lam: float = 1,
                 alpha: float = .5):
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        solver: GradientDescent, default=GradientDescent()
            Descent method solver to use for the logistic regression objective optimization

        penalty: str, default="none"
            Type of regularization term to add to logistic regression objective. Supported values
            are "none", "l1", "l2"

        lam: float, default=1
            Regularization parameter to be used in case `self.penalty_` is not "none"

        alpha: float, default=0.5
            Threshold value by which to convert class probability to class value

        include_intercept: bool, default=True
            Should fitted model include an intercept or not
        """
        super().__init__()
        self.include_intercept_ = include_intercept
        self.solver_ = solver
        self.lam_ = lam
        self.penalty_ = penalty
        self.alpha_ = alpha

        if penalty not in ["none", "l1", "l2"]:
            raise ValueError("Supported penalty types are: none, l1, l2")

        self.coefs_ = None




    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Logistic regression model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model using specified `self.solver_` passed when instantiating class and includes an intercept
        if specified by `self.include_intercept_
        """
        n_samples, n_features = X.shape

        if self.include_intercept_:
            X = np.c_[X,np.ones(n_samples)]
            n_features+=1
      
        init = np.random.randn(n_features) / (n_features ** 0.5)
    
        if self.penalty_ == "none":
            log_module = LogisticModule(weights=init)
            module_to_fit = log_module
        else:
            log_module = LogisticModule()
            if self.penalty_ == "l1":
                reg_module = L1()
            elif self.penalty_ == "l2":
                reg_module = L2()
            regularized_module = RegularizedModule(fidelity_module=log_module,
                                                regularization_module=reg_module,
                                                lam=self.lam_,
                                                weights=init)
            module_to_fit = regularized_module
        
        self.coefs_ = self.solver_.fit(module_to_fit, X, y)
     

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        y_predict = self.predict_proba(X)
        return np.where(y_predict > self.alpha_, 1, 0)


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities of samples being classified as `1` according to sigmoid(Xw)

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict probability for

        Returns
        -------
        probabilities: ndarray of shape (n_samples,)
            Probability of each sample being classified as `1` according to the fitted model
        """
        if self.include_intercept_:
            X = np.c_[X, np.ones(X.shape[0])]
        return 1 / (1 + np.exp(-X @ self.coefs_))
    
    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification error

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under misclassification error
        """
        from loss_functions import misclassification_error
        return misclassification_error(y, self._predict(X))





