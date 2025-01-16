import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from base_module import BaseModule
from base_learning_rate import  BaseLR
from gradient_descent import GradientDescent
from learning_rate import FixedLR
from cross_validate import cross_validate
from utils import split_train_test, confusion_matrix
from plotly.subplots import make_subplots

# from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from modules import L1, L2
from logistic_regression import LogisticRegression
from utils import split_train_test
from sklearn.metrics import roc_curve, auc

import plotly.graph_objects as go
import plotly.express as px
from loss_functions import misclassification_error, accuracy

def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))



def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights_list = []

    def callback(val, weight, **kwargs):
        values.append(val)
        weights_list.append(weight)

    return callback, values, weights_list

def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    models_and_names = {'L1': L1, 'L2': L2}
    min_val = np.inf
    min_eta = 0

    for name, model in models_and_names.items():
        fig = go.Figure()
        for eta in etas:
            callback,values, weights = get_gd_state_recorder_callback()
            gd = GradientDescent(learning_rate = FixedLR(eta), callback=callback)
            f = model(np.array(init))
            gd.fit(f, np.empty(0), np.empty(0))

            if np.min(values) < min_val:
                min_val = np.min(values)
                min_eta = eta

            descent_path_fig = plot_descent_path(model, np.array(weights), title=f"Model: {name}, Learning Rate:{eta}")
            descent_path_fig.show()

            fig.add_trace(go.Scatter(x=np.arange(1, len(values) + 1), y=values, mode='lines', name=f"Learning Rate:{eta}"))

        fig.update_layout(
            title=f"Convergence Rate of {name} Model",
            xaxis_title='Number of Iterations',
            yaxis_title='Norm'
        )
        fig.show()

        print(f"Lowest loss achieved for {name} model: {min_val} with learning rate: {min_eta}")


def load_data(path: str = "SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """

    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)



def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Convert df to numpy array
    X_train, y_train = X_train.values, y_train.values
    X_test, y_test = X_test.values, y_test.values

    # Plotting convergence rate of logistic regression over SA heart disease data
    callback , values, weights = get_gd_state_recorder_callback()
    t = 20000
    lr = 1e-4
    gd = GradientDescent(learning_rate=FixedLR(lr), max_iter=t, callback=callback)
    model = LogisticRegression(solver=gd)
    model.fit(X_train,y_train)

    y_pred = model.predict_proba(X_test)
    fpr,tpr,ths = roc_curve(y_test, y_pred)



    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, line=dict(color='orange'), mode='lines', name='ROC Curve'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='blue'), name='random'))

    fig.update_layout(
        title=f'ROC Curve - logistic regression ( AUC : {auc(fpr, tpr):.3f} )',
        xaxis_title='FPR',
        yaxis_title='TPR',
        showlegend=False
    )

    fig.show()


    # q6 - determine maximal alpha against the criteria
    criteria = tpr - fpr
    idx = np.argmax(criteria)
    alpha_opt = ths[idx]
    error_opt = model.loss(X_test, y_test)

    print(f"Optimal alpha: {alpha_opt}, with test error of : {error_opt}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter

    lamb = [0.001,0.002,0.005,0.01,0.02,0.05,0.1]

    best_lamb = None
    best_score =np.inf
    for i in range(len(lamb)):
        cur_lamb = lamb[i]
        gd = GradientDescent(learning_rate=FixedLR(lr), max_iter=t, callback=callback)
        model = LogisticRegression(solver=gd, penalty='l1', alpha=0.5, lam=cur_lamb)
        train_score,validation_score = cross_validate(estimator=model, X=X_train, y=y_train, scoring=misclassification_error)

        if validation_score < best_score:
            best_lamb = cur_lamb
            best_score = validation_score

    gd = GradientDescent(learning_rate=FixedLR(lr), max_iter=t, callback=callback)
    model = LogisticRegression(solver=gd, penalty='l1', alpha=0.5, lam=best_lamb)
    model.fit(X_train, y_train)

    best_lamb_error = model.loss(X_test, y_test)
    print(f"Best lambda: {best_lamb}, with test error of : {best_lamb_error}")




if __name__ == '__main__':

    np.random.seed(0)
    # compare_fixed_learning_rates()
    fit_logistic_regression()
