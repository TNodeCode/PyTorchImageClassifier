# Datasets and Metrics

## Traing, Validation and Test datasets

In machine learning, it is common to split your available data into three separate datasets: the training dataset, the validation dataset, and the test dataset.

- ***Training Dataset***: The training dataset is the largest portion of your data and is used to train the neural network model. It contains labeled examples (input data and corresponding output labels) that the model uses to learn patterns and relationships. During training, the model adjusts its parameters to minimize the difference between its predicted outputs and the true labels in the training dataset.

- ***Validation Dataset***: The validation dataset is used to tune the hyperparameters of the neural network and assess its performance during the training process. Hyperparameters are configuration settings that are not learned by the model but affect its learning process, such as the learning rate or the number of layers in the network. By evaluating the model's performance on the validation dataset, you can make adjustments to the hyperparameters to improve the model's generalization abilities. The validation dataset helps prevent overfitting, which is when the model becomes too specialized to the training data and performs poorly on new, unseen data.

- ***Test Dataset***: The test dataset is used to evaluate the final performance of the trained model after all the training and hyperparameter tuning are completed. It represents new, unseen data that the model has not encountered before. The test dataset is important to assess how well the model generalizes to real-world examples. By evaluating the model on the test dataset, you can obtain an unbiased estimate of its performance in practical scenarios. It helps you make informed decisions about deploying or further refining the model.

It is essential to keep the test dataset separate from the training and validation datasets throughout the development process to avoid any bias in evaluating the model's performance. The test dataset should only be used as a final evaluation step, and no modifications to the model or hyperparameters should be made based on its results.

## True positives, true negatives, false positives and false negatives

### Binary classification

In the context of binary classification tasks (where there are two possible class labels), the terms "true positives," "true negatives," "false positives," and "false negatives" are used to describe the outcomes of the classification process. Let's break down each term:

- ***True Positives (TP)***: True positives refer to the cases where the model correctly predicts the positive class. In other words, the model correctly identifies the presence of the target condition or class.
- ***True Negatives (TN)***: True negatives refer to the cases where the model correctly predicts the negative class. In other words, the model correctly identifies the absence of the target condition or class.
- ***False Positives (FP)***: False positives occur when the model incorrectly predicts the positive class. In other words, the model predicts the presence of the target condition or class when it is actually absent.
- ***False Negatives (FN)***: False negatives occur when the model incorrectly predicts the negative class. In other words, the model predicts the absence of the target condition or class when it is actually present.

To illustrate these terms, let's consider an example of a binary classification task: classifying whether an email is spam or not spam.

A true positive (TP) would occur when the model correctly identifies an email as spam.
A true negative (TN) would occur when the model correctly identifies an email as not spam.
A false positive (FP) would occur when the model incorrectly classifies a non-spam email as spam.
A false negative (FN) would occur when the model incorrectly classifies a spam email as not spam.
These concepts are commonly used to evaluate the performance of a classification model. Metrics such as accuracy, precision, recall, and F1 score are calculated based on these four outcomes to quantify the model's performance in terms of correctly and incorrectly classified instances.

Understanding true positives, true negatives, false positives, and false negatives allows for a more nuanced evaluation of a model's effectiveness in differentiating between classes and identifying patterns correctly.

### Multiclass classification

In multiclass classification problems, where there are more than two possible classes, the definitions of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN) are extended to accommodate the multiple classes. Let's explore how these terms are defined in the context of multiclass classification:

- ***True Positives (TP)***: For a particular class, true positives refer to the cases where the model correctly predicts that class. In other words, the model correctly identifies instances belonging to that specific class.
- ***True Negatives (TN)***: For a particular class, true negatives refer to the cases where the model correctly predicts the absence of that class. In other words, the model correctly identifies instances not belonging to that specific class.
- ***False Positives (FP)***: For a particular class, false positives occur when the model incorrectly predicts that class. In other words, the model predicts the presence of the class when it is actually absent.
- ***False Negatives (FN)***: For a particular class, false negatives occur when the model incorrectly predicts the absence of that class. In other words, the model predicts the absence of the class when it is actually present.

To further illustrate these terms, let's consider an example of a multiclass classification task: classifying images of animals into categories of "cat," "dog," and "bird."

A true positive (TP) for the "cat" class would occur when the model correctly identifies an image as a cat.
A true negative (TN) for the "cat" class would occur when the model correctly identifies an image as not a cat (i.e., as a dog or a bird).
A false positive (FP) for the "cat" class would occur when the model incorrectly classifies an image as a cat when it is actually a dog or a bird.
A false negative (FN) for the "cat" class would occur when the model incorrectly classifies an image as not a cat when it is actually a cat.
Similarly, these terms can be defined for each class in the multiclass problem.

In multiclass classification, the evaluation metrics are often extended to consider performance across all classes. Some commonly used metrics include accuracy, precision, recall (sensitivity), specificity, and F1 score, which provide an overall assessment of the model's performance in distinguishing between multiple classes.

It's worth noting that different evaluation metrics and techniques can be used depending on the specific requirements and characteristics of the multiclass problem at hand.

## Accuracy, Precision and Recall

Certainly! Accuracy, precision, and recall are commonly used evaluation metrics in classification tasks. Let's delve into the definitions of each metric:

Accuracy: Accuracy measures the overall correctness of the model's predictions by calculating the ratio of correctly predicted instances (both true positives and true negatives) to the total number of instances in the dataset. It provides an indication of how well the model performs across all classes and is defined as follows:

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Accuracy is a useful metric when the classes in the dataset are balanced, meaning they have roughly equal representation. However, accuracy can be misleading when the dataset is imbalanced, as a high accuracy value can still occur if the model performs well on the majority class but poorly on the minority class.

Precision: Precision is a metric that quantifies the proportion of correctly predicted positive instances (true positives) out of all instances that the model predicted as positive (both true positives and false positives). Precision focuses on the soundness of positive predictions and is defined as follows:

```
Precision = TP / (TP + FP)
``` 

Precision provides insights into the model's ability to avoid false positives. It is particularly important in scenarios where false positives are undesirable or have significant consequences. For example, in medical diagnosis, precision is valuable as it represents the probability of a positive prediction being correct.

Recall (Sensitivity): Recall, also known as sensitivity or true positive rate, measures the proportion of correctly predicted positive instances (true positives) out of all instances that truly belong to the positive class (true positives and false negatives). Recall emphasizes the model's ability to capture all positive instances and is defined as follows:

```
Recall = TP / (TP + FN)
```

Recall is particularly important when the cost of false negatives is high. For instance, in disease detection, recall is crucial because missing a positive case (false negative) can have severe consequences. Therefore, a high recall indicates that the model is good at identifying positive instances, minimizing false negatives.

It's important to note that accuracy, precision, and recall are not independent of each other. Depending on the specific use case and requirements, you might prioritize one metric over the others. Additionally, it's common to use these metrics in conjunction with other evaluation measures, such as F1 score, which combines precision and recall into a single value to provide a balanced assessment of the model's performance.


### F-beta and F1 score

The F1 score is a commonly used evaluation metric in classification tasks that combines precision and recall into a single value. To understand the F1 score, it's helpful to first introduce the concept of the F-beta score, which is a generalized form of the F1 score.

The F-beta score is defined as the weighted harmonic mean of precision and recall, where beta determines the weight given to precision versus recall. The beta parameter allows you to adjust the emphasis on either precision or recall, depending on the specific requirements of your problem.

The formula for the F-beta score is as follows:

```
F-beta Score = (1 + beta^2) * (precision * recall) / ((beta^2 * precision) + recall)
```

When beta = 1, it reduces to the F1 score, which weighs precision and recall equally. The F1 score is often used when you want to balance the importance of precision and recall.

The formula for the F1 score is therefore:

```
F1 Score = 2 * (precision * recall) / (precision + recall)
```

The F1 score ranges between 0 and 1, with 1 being the best score indicating perfect precision and recall, and 0 being the worst score indicating poor performance in either precision or recall.

The F1 score is particularly useful when you want to find a balance between precision and recall, as it considers both metrics simultaneously. It is commonly used in scenarios where false positives and false negatives are equally important and need to be minimized.

It's worth noting that if you have a specific preference for precision or recall based on your problem's requirements, you can select a different beta value to compute the F-beta score accordingly. For example, if recall is more critical than precision, you can use a higher beta value (e.g., beta = 2) to give more weight to recall in the evaluation.

In summary, the F1 score is a widely used metric that combines precision and recall into a single value, providing a balanced assessment of a model's performance in classification tasks. The F-beta score is a generalization of the F1 score that allows you to adjust the weight given to precision and recall based on the specific needs of your problem.