import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve, recall_score


# Define your trained models (rf, adab, gbm, model)
#estimators = [rf, adab, gbm, model, GNB]
# classifiers = ['RandomForestClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'XGBoostClassifier','Gaussian Naive Bayes Classifier']

# Custom labels for the confusion matrix and classification report
# labels = ['No Out of Stock', 'Out of Stock']  # Assuming binary classification with 0 and 1
def evaluate_models(estimators, classifiers, labels, X_test, y_test):
    # Initialize an empty list to store the results
    metrics_list = []

    # Loop over models to calculate and store metrics
    for i, estimator in enumerate(estimators):
        # Make predictions on the test set
        y_pred = estimator.predict(X_test)
        y_pred_prob = estimator.predict_proba(X_test)[:, 1]  # Probabilities for ROC Curve
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Extract TN, FP, FN, TP
        tn, fp, fn, tp = cm.ravel()
        
        # Compute metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        
        # Compute ROC AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        # Append metrics to the list
        metrics_list.append({
            'Model': classifiers[i],
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score,
            'AUC': roc_auc
        })

    # Convert the list of metrics to a DataFrame
    metrics_df = pd.DataFrame(metrics_list)

    # Display the DataFrame with all metrics
    print(metrics_df)

    # Plot metrics for comparison
    metrics_df.set_index('Model').plot(kind='bar', figsize=(14, 8))
    plt.title('Comparison of Model Performance Metrics')
    plt.xlabel('Model')
    plt.ylabel('Metric Value')
    plt.xticks(rotation=45)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # Plot ROC Curves
    plt.figure(figsize=(10, 8))
    for i, estimator in enumerate(estimators):
        y_pred_prob = estimator.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        plt.plot(fpr, tpr, label=f'{classifiers[i]} (AUC = {metrics_df["AUC"][i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='best')
    plt.show()

    # Loop over models to calculate and display confusion matrices and classification reports
    for i, estimator in enumerate(estimators):
        # Make predictions on the test set
        y_pred = estimator.predict(X_test)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Create a ConfusionMatrixDisplay object
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        
        # Plot confusion matrix with custom labels
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')  # Display raw counts
        
        # Annotate the matrix with additional metrics
        tn, fp, fn, tp = cm.ravel()
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                value = cm[j, k]
                # Annotate each cell with TP, TN, FP, FN
                if j == 0 and k == 0:
                    annotation = f'TN: {tn}'
                elif j == 0 and k == 1:
                    annotation = f'FP: {fp}'
                elif j == 1 and k == 0:
                    annotation = f'FN: {fn}'
                elif j == 1 and k == 1:
                    annotation = f'TP: {tp}'
                else:
                    annotation = str(value)
                
                # Center the text in the cell with adjusted position
                ax.text(k, j, annotation, ha='center', va='center', color='black', fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3'))
        
        # Add title
        plt.title(f'Confusion Matrix for {classifiers[i]}')
        
        # Show the plot
        plt.show()
        
        # Generate classification report
        report_dict = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
        
        # Convert classification report to a DataFrame
        report_df = pd.DataFrame(report_dict).transpose()
        
        # Plot classification report as a heatmap
        plt.figure(figsize=(10, 5))
        sns.heatmap(report_df.iloc[:-1, :].T, annot=True, cmap="Blues", cbar=False, fmt=".2f")
        plt.title(f'Classification Report for {classifiers[i]}')
        plt.show()

    # Initialize lists to store the TP, TN, FP, FN for each model
    true_positives = []
    true_negatives = []
    false_positives = []
    false_negatives = []

    # Loop over models to calculate confusion matrices and extract metrics
    for i, estimator in enumerate(estimators):
        # Make predictions on the test set
        y_pred = estimator.predict(X_test)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Ensure the confusion matrix is 2x2 for binary classification
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            # Append values to the lists
            true_positives.append(tp)
            true_negatives.append(tn)
            false_positives.append(fp)
            false_negatives.append(fn)
        else:
            raise ValueError(f"Unexpected confusion matrix shape: {cm.shape}")

    # Plot histograms
    x = np.arange(len(classifiers))  # The label locations
    width = 0.2  # The width of the bars

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot bars for each metric
    bars_tp = ax.bar(x - 1.5 * width, true_positives, width, label='True Positives', color='lightblue')
    bars_tn = ax.bar(x - 0.5 * width, true_negatives, width, label='True Negatives', color='lightyellow')
    bars_fp = ax.bar(x + 0.5 * width, false_positives, width, label='False Positives', color='lightcoral')
    bars_fn = ax.bar(x + 1.5 * width, false_negatives, width, label='False Negatives', color='lightsalmon')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Models', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Comparison of TP, TN, FP, and FN Across Models', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(classifiers, rotation=45, ha='right')
    ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()





def evaluate_model(model,X,y_true):

  """
  This function takes trained model , X(input) , y_true(true label) as input and 
  evaluates model on different metrics
  """
  th = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
  pred = model.predict_proba(X)[:,1]
  scores = []
  tpr = []
  fpr = []

  for i in th:
    pred_labels =[]
    for j in pred:
      if j>=i:
        pred_labels.append(1)
      else:
        pred_labels.append(0)
    scores.append([recall_score(y_true,pred_labels,pos_label=1),precision_score(y_true,pred_labels,pos_label=1)])

    pred_labels = np.array(pred_labels)

    fp = np.sum((pred_labels == 1) & (y_true == 0))
    tp = np.sum((pred_labels == 1) & (y_true == 1))
    fn = np.sum((pred_labels == 0) & (y_true == 1))
    tn = np.sum((pred_labels == 0) & (y_true == 0))
    
    fpr.append(fp / (fp + tn))
    tpr.append(tp / (tp + fn))
  f1score = f1_score(y_true, model.predict(X),average = None)
  print("The F1 scores of each class and Macro F1 score are : " , f1score[0] ,f1score[1] , (f1score[0]+f1score[1])/2)

  xx = [X[0] for X in scores]
  yy = [Y[1] for Y in scores]
  fig = plt.figure(figsize=(6,8))

  ax1 = fig.add_subplot(311)
  ax1.plot(xx,yy,label = 'AUC PR curve'+str(np.round(auc(xx,yy),3)))

  ax1.set_title("Precision - Recall curve")
  ax1.set_xlabel("Recall")
  ax1.set_ylabel("Precision")
  ax1.legend()

  plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.9)

  ax2 = fig.add_subplot(312)
  ax2.plot(fpr,tpr,label = "Model ROC AUC on test data : "+str(np.round(auc(fpr,tpr),3)))
  ax2.plot([0, 1], ls="--",label='No Skill')
  ax2.set_title("ROC-AUC curve")
  ax2.set_xlabel("FPR")
  ax2.set_ylabel("TPR")
  ax2.legend()
  plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.9)



