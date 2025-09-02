# -- coding: gbk --
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef,
    precision_recall_curve, roc_curve, auc, log_loss
)
from sklearn.calibration import calibration_curve
from scipy.stats import entropy
import warnings
import joblib
import os
import sys
import datetime
from attention import EnhancedTemporalAttention  # Ensure to import your custom layer

# Disable warnings
warnings.filterwarnings("ignore")

# Set global font size
plt.rcParams.update({
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'font.family': 'DejaVu Sans'  # Ensure Chinese character support
})

def specificity_score(y_true, y_pred):
    """Calculate specificity (true negative rate)"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def calculate_metrics(y_true, y_pred, probs=None):
    """Calculate comprehensive evaluation metrics"""
    metrics = {}
    
    # Basic metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = tn + fp + fn + tp
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['specificity'] = specificity_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # Additional metrics
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False positive rate
    metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False negative rate
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative predictive value
    metrics['fdr'] = fp / (fp + tp) if (fp + tp) > 0 else 0.0  # False discovery rate
    metrics['informedness'] = metrics['recall'] + metrics['specificity'] - 1  # Youden's J statistic
    metrics['markedness'] = metrics['precision'] + metrics['npv'] - 1
    metrics['support'] = total  # Sample size
    
    # Probability-related metrics
    if probs is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, probs)
        except ValueError:
            metrics['roc_auc'] = 0.5  # Fallback when only one class
            
        # Add PR AUC calculation
        try:
            metrics['pr_auc'] = average_precision_score(y_true, probs)
        except:
            metrics['pr_auc'] = 0.0
            
        metrics['log_loss'] = log_loss(y_true, probs)
        
        # Calculate calibration error
        try:
            prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10, strategy='quantile')
            metrics['calibration_error'] = np.mean(np.abs(prob_true - prob_pred))
        except ValueError:
            metrics['calibration_error'] = 0.0
        
        # Calculate prediction entropy
        try:
            entropy_values = entropy(np.column_stack([1-probs, probs]), axis=1)
            metrics['avg_entropy'] = np.mean(entropy_values)
        except:
            metrics['avg_entropy'] = 0.0
    
    return metrics

def model_agreement(all_preds):
    """Calculate model agreement"""
    # Convert predictions to numpy array
    preds_array = np.array(all_preds)
    
    # Calculate agreement for each sample (proportion of models that agree)
    agreements = np.mean(preds_array == np.round(np.mean(preds_array, axis=0)), axis=0)
    
    # Group by agreement level
    bins = [0, 0.5, 0.7, 0.9, 1.0]
    labels = ['Low(<50%)', 'Medium(50-70%)', 'High(70-90%)', 'Very High(>90%)']
    agreement_groups = pd.cut(agreements, bins=bins, labels=labels)
    
    # Calculate group proportions (compatible with all Pandas versions)
    group_counts = pd.Series(agreement_groups).value_counts()
    total = group_counts.sum()
    
    # Manually calculate normalized proportions
    normalized_counts = {}
    for group in labels:
        if group in group_counts:
            normalized_counts[group] = group_counts[group] / total
    
    return {
        'overall': np.mean(agreements),
        'distribution': normalized_counts
    }

def uncertainty_analysis(all_probs, y_true, y_pred):
    """Uncertainty analysis"""
    # Calculate standard deviation of probabilities for each sample
    prob_std = np.std(np.array(all_probs), axis=0)
    
    # Uncertainty grouping
    uncert_bins = [0, 0.1, 0.2, 0.3, 1.0]
    uncert_labels = ['Low', 'Medium', 'High', 'Very High']
    uncert_groups = pd.cut(prob_std, bins=uncert_bins, labels=uncert_labels)
    
    # Calculate performance for each group
    group_perf = {}
    for group in uncert_labels:
        mask = (uncert_groups == group)
        if mask.sum() > 0:
            acc = accuracy_score(y_true[mask], y_pred[mask])
            group_perf[group] = {
                'accuracy': acc,
                'size': mask.sum(),
                'proportion': mask.sum() / len(y_true)
            }
    
    return group_perf

def comprehensive_evaluation(models, test_x, test_y, chromname):
    """
    Comprehensive evaluation of Bagging ensemble model
    Returns result dictionary with all evaluation metrics
    """
    # 1. Initialize result storage
    results = {
        'avg_strategy': {},
        'agreement': {},
        'uncertainty': {},
        'single_models': []
    }
    
    # 2. Get true labels
    test_labels = np.argmax(test_y, axis=1)
    
    # 3. Model predictions
    all_preds = []
    all_probs = []
    
    for model in models:
        # Get prediction probabilities
        prob = model.predict(test_x, verbose=0)
        pos_probs = prob[:, 1]  # Positive class probability
        all_probs.append(pos_probs)
        
        # Get predicted classes
        pred = np.argmax(prob, axis=1)
        all_preds.append(pred)
    
    # 4. Ensemble strategy: average probability
    avg_probs = np.mean(all_probs, axis=极0)
    ensemble_pred_avg = (avg_probs > 0.5).astype(int)
    
    # 5. Calculate metrics - average probability strategy
    results['avg_strategy'] = calculate_metrics(
        test_labels, ensemble_pred_avg, avg_probs
    )
    
    # 7. Calculate single model metrics
    single_metrics = []
    for i in range(len(models)):
        metrics = calculate_metrics(
            test_labels, all_preds[i], all_probs[i]
        )
        single_metrics.append(metrics)
        results['single_models'].append(metrics)
    
    # 8. Calculate model agreement
    results['agreement'] = model_agreement(all_preds)
    
    # 9. Uncertainty analysis
    results['uncertainty'] = uncertainty_analysis(all_probs, test_labels, ensemble_pred_avg)
      
    return results

def save_full_report(results, chromname, report_path):
    """Save comprehensive evaluation report (including ablation study data)"""
    with open(report_path, "w") as f:
        f.write(f"=== {chromname} Chromatin Loop Detection Model Evaluation and Ablation Study Report ===\n\n")
        
        # Report basic information
        f.write(f"Evaluation Time: {datetime.datetime.now()}\n")
        f.write(f"Number of Test Samples: {results['avg_strategy']['support']}\n")
        f.write(f"Number of Single Models: {len(results['single_models'])}\n\n")
        
        # Single model performance statistics
        f.write("[Single Model Performance Statistics]\n")
        f.write("Model\tAccuracy\tPrecision\tRecall\tF1\tAUC\tMCC\n")
        for i, model in enumerate(results['single_models']):
            f.write(f"Model{i+1}\t{model['accuracy']:.4f}\t{model['precision']:.4f}\t"
                    f"{model['recall']:.4f}\t{model['f1']:.4f}\t"
                    f"{model.get('roc_auc', 0):.4f}\t{model['mcc']:.4f}\n")
        
        # Calculate average single model performance
        avg_metrics = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'mcc']:
            values = [m[metric] for m in results['single_models'] if metric in m]
            if values:
                avg_metrics[metric] = np.mean(values)
        
        f.write(f"\nSingle Avg\t{avg_metrics['accuracy']:.4f}\t{avg_metrics['precision']:.4f}\t"
                f"{avg_metrics['recall']:.4f}\t{avg_metrics['f1']:.4f}\t"
                f"{avg_metrics.get('roc_auc', 0):.4f}\t{avg_metrics['mcc']:.4f}\n\n")
        
        # Ensemble strategy results
        f.write("[Average Probability Ensemble Strategy]\n")
        f.write("Accuracy\tPrecision\tRecall\tF1\tAUC\tMCC\tSpecificity\tInformedness\n")
        f.write(f"{results['avg_strategy']['accuracy']:.4f}\t{results['avg_strategy']['precision']:.4f}\t"
                f"{results['avg_strategy']['recall']:.4f}\t{results['avg_strategy']['f1']:.4f}\t"
                f"{results['avg_strategy'].get('roc_auc', 0):.4f}\t{results['avg_strategy']['mcc']:.4f}\t"
                f"{results['avg_strategy']['specificity']:.4f}\t{results['avg_strategy']['informedness']:.4f}\n\n")
        
        # Ablation study findings
        f.write("[Ablation Study Findings]\n")
        f.write(f"? Average Single Model F1 Score: {avg_metrics['f1']:.4f}\n")
        f.write(f"? Ensemble Model F1 Score: {results['avg_strategy']['f1']:.4f}\n")
        improvement = (results['avg_strategy']['f1'] - avg_metrics['f1']) / avg_metrics['f1'] * 100
        f.write(f"? Relative Improvement: {improvement:.2f}%\n")
        
        f.write(f"\n? Average Single Model AUC: {avg_metrics['roc_auc']:.4f}\n")
        f.write(f"? Ensemble Model AUC: {results['avg_strategy']['roc_auc']:.4f}\n")
        auc_improvement = (results['avg_strategy']['roc_auc'] - avg_metrics['roc_auc']) / avg_metrics['roc_auc'] * 100
        f.write(f"? Relative Improvement: {auc_improvement:.2f}%\n")
        
        # Model agreement
        f.write("\n[Model Agreement]\n")
        f.write(f"Overall Agreement: {results['agreement']['overall']:.4f}\n")
        f.write("Agreement Distribution:\n")
        for group, prop in results['agreement']['distribution'].items():
            f.write(f"  ? {group}: {prop:.2%}\n")
        
        # Uncertainty analysis
        f.write("\n[Uncertainty Analysis]\n")
        for level, data in results['uncertainty'].items():
            f.write(f"? {level} Uncertainty Region: {data['proportion']:.2%} samples, "
                    f"Accuracy={data['accuracy']:.4f}\n")
        
        # Model performance improvement analysis
        f.write("\n[Model Performance Improvement Analysis]\n")
        if improvement > 0:
            f.write(f"Ensemble strategy successfully improved model performance, F1 score increased by {improvement:.2f}%.\n")
            f.write("This indicates that predictions from different models are complementary, and ensemble learning effectively reduces variance.\n")
        else:
            f.write("Ensemble strategy did not significantly improve model performance.\n")
            f.write("It is recommended to examine model diversity or try different ensemble methods.\n") 
    
    print(f"Comprehensive evaluation report saved to: {report_path}")

def main():
    # Configuration parameters
    test_chrom = 'chr15'  # Chromosome to evaluate
    model_dir = "/home/yanghao/Loopnetae/chr15/"  # Model storage directory
    test_data_path = "/home/yanghao/Loopnetae/testset/chr15.pkl" # Pre-saved test set
    report_dir = "/home/yanghao/Loopnetae/modelevaluation/"
    
    # 1. Load test set
    try:
        test_data = joblib.load(test_data_path)
        
        # Fix: Process tuple-formatted test data
        if isinstance(test_data, tuple):
            print("Test set is in tuple format, using index access")
            # According to the typical structure, the first element of the tuple is features, the second is labels
            test_x = test_data[0]
            test_y = test_data[1]
            
            # Check if there is a third element (e.g., features, labels, other metadata)
            if len(test_data) >= 3:
                print(f"Test set contains extra elements ({len(test_data)}")
                
        elif isinstance(test_data, dict):
            print("Test set is in dictionary format, using key access")
            test_x = test_data['features']
            test_y = test_data['labels']
        else:
            raise ValueError(f"Unknown test set format: {type(test_data)}")
            
        print(f"Successfully loaded test set: {test_data_path}")
        print(f"Test set shape: {test_x.shape}, Labels shape: {test_y.shape}")
    except Exception as e:
        print(f"Failed to load test set: {str(e)}")
        return
    
    # 2. Prepare test data format
    # Assume test set is 2D array (samples, features)
    # Reshape to 3D format (samples, features, 1)
    test_x_3d = test_x.reshape(test_x.shape[0], test_x.shape[1], 1)
    
    # Convert labels to categorical format (if not already converted)
    if len(test_y.shape) == 1 or (len(test_y.shape) == 2 and test_y.shape[1] == 1):
        test_y_cat = to_categorical(test_y, num_classes=2)
    else:
        test_y_cat = test_y
    
    # 3. Load all models
    models = []
    for i in range(0, 10):  # 10 models
        model_path = f'{model_dir}/{test_chrom}_{i}.h5'
        try:
            # Fix: Ensure custom layer reference is correct
            model = load_model(
                model_path,
                custom_objects={'EnhancedTemporalAttention': EnhancedTemporalAttention}
            )
            models.append(model)
            print(f"Successfully loaded model: {model_path}")
        except Exception as e:
            print(f"Failed to load model {model_path}: {str(e)}")
    
    if not models:
        print("Error: No models loaded")
        return
    
    # 4. Perform comprehensive evaluation
    results = comprehensive_evaluation(
        models=models,
        test_x=test_x_3d,
        test_y=test_y_cat,
        chromname=test_chrom
    )
    
    # 5. Save full report
    report_path = os.path.join(report_dir, f"{test_chrom}_full_report.txt")
    save_full_report(results, test_chrom, report_path)
    print(f"{test_chrom} evaluation completed")

if __name__ == "__main__":
    # Set random seed
    np.random.seed(42)
    # Run main function
    main()