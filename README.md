# Using Machine Learning to Predict Student Performance

This project uses the data from Cortez,Paulo. (2014). Student Performance. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T. This dataset tabulates student achievement in secondary education with numerous demographic, social, and school related information.

The code main.py uses a simple linear regression with numerous inputs deemed relevant for the prediction task.

The model is then retrained using only 5 features with the highest weight from the previous model. These were 'G2', 'higher', 'paid', 'failures', 'schoolsup', standing for second period grade, whether they want to pursue higher education, if they take extra paid classes, number of past class failures, and if extra educational support. Doing this tends to slightly improve the model's $r^2$ by an average of 0.05 points, indicating the relationships in the dataset tend to be relatively simple and mainly dependent on a handful of features.

One must note that high scores were only possible when using the 'G1' and 'G2' features, which contain the students' midterm grades. As one would expect, students with higher midterm grades will tend to have higher final grades (though that's not always true from personal experience 😭). When those are rremoved, the model always performs poorly, struggling to reach a $r^2$ value higher than 0.4.

## Future Work
Future work could investigate using different model architectures as those used in the [2008 paper](https://www.semanticscholar.org/paper/Using-data-mining-to-predict-secondary-school-Cortez-Silva/61d468d5254730bbecf822c6b60d7d6595d9889c) which used the dataset for its results.
