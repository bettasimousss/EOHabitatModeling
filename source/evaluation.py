import sklearn.metrics as skm
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binned_statistic
from sklearn.metrics import RocCurveDisplay

'''
Y_score: pandas DataFrame with columns = classes in alphabetical order
Y_true: pandas Series

Use class codes as EUNIS codes, not label encoder to avoid mistakes and increase interpretability
'''
'''
Y_score: pandas DataFrame with columns = classes in alphabetical order
Y_true: pandas Series

Use class codes as EUNIS codes, not label encoder to avoid mistakes and increase interpretability
'''

def eval_classifier(y_score,y_true, k_list=[3,5,10], model_name=None, super_class=None, classes=None):
    y_hat = y_score.idxmax(axis=1)
    perfs = pd.DataFrame.from_dict(skm.classification_report(y_true=y_true,y_pred=y_hat, output_dict=True, zero_division=0))
    
    if len(classes)>2:
        for k in k_list:
            perfs['top%d'%k] = skm.top_k_accuracy_score(y_true=y_true, y_score=y_score, k=k, labels=classes)
        
    perfs['adj_balanced_accuracy'] = skm.balanced_accuracy_score(y_true=y_true, y_pred=y_hat, adjusted=True)
    perfs['balanced_accuracy'] = skm.balanced_accuracy_score(y_true=y_true, y_pred=y_hat, adjusted=False)
    
    ##### Computing coverage: does the correct class rank well on average at least ? 
    ohe = OneHotEncoder(categories=[y_score.columns.tolist()],sparse_output=False)
    ohe.fit(y_true.values.reshape(-1,1))
    y_1h = pd.DataFrame(data=ohe.transform(y_true.values.reshape(-1,1)), columns=y_score.columns.tolist())
    
    perfs['coverage'] = skm.coverage_error(y_true=y_1h, y_score=y_score)
    
    #### Confusion matrix
    cm = pd.DataFrame(data=skm.confusion_matrix(y_pred=y_hat,y_true=y_true,labels=classes),columns=classes,index=classes)
    
    ##### Add model metadata
    perfs['nb_classes'] = y_score.shape[1]
    
    if model_name is not None:
        perfs['model_name']=model_name
        
    if super_class is not None:
        perfs['super_class']=super_class
        
    return perfs, cm


def mcroc_eval(Y_hat,y_true, title):
    Y_true = pd.get_dummies(y_true)
    fig, ax = plt.subplots(1,1)
    for class_id in Y_true.columns.tolist():
        RocCurveDisplay.from_predictions(
            Y_true[class_id].values,
            Y_hat[class_id].values,
            name=f"ROC curve for : %s"%class_id,
            ax=ax
        )

    fig.suptitle(title)
    
    return fig

def plot_confusion_matrix(conf_mat, title='',gs=10):
    fig, ax = plt.subplots(1,1,figsize=(2*gs,gs))
    conf_norm = conf_mat.apply(lambda x: x/sum(x), axis=1)
    sns.heatmap(data=conf_norm,cmap='Reds',vmin=0,vmax=1,ax=ax)
    fig.suptitle(title)
    plt.close()
    
    return fig

def expected_calibration_error(y_true, prob_pred, n_bins=10):
    """
    Compute Expected Calibration Error (ECE)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_edges[:-1]
    bin_uppers = bin_edges[1:]

    confidences = np.max(prob_pred, axis=1)  # Max predicted probability
    predictions = np.argmax(prob_pred, axis=1)
    correctness = (predictions == y_true).astype(float)

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if np.sum(in_bin) > 0:
            acc_bin = np.mean(correctness[in_bin])
            conf_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_bin - conf_bin) * np.sum(in_bin) / len(y_true)

    return ece

def plot_calibration_curve(y_true, prob_pred, title="Calibration Plot",out_file="calibration.png"):
    """
    Reliability diagram (Calibration curve)
    """
    bin_means, bin_edges, _ = binned_statistic(np.max(prob_pred, axis=1), (np.argmax(prob_pred, axis=1) == y_true),
                                               statistic="mean", bins=10)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, ax = plt.subplots(1,1,figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    ax.plot(bin_centers, bin_means, "s-", label="Model Calibration")
    ax.set_xlabel("Predicted Confidence")
    ax.set_ylabel("Actual Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid()

    fig.savefig(out_file)
