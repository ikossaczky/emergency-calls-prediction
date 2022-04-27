
# Emergency call prediction

### Implementation:
- process_data.sh : data preprocessing


- data_pipeline.py: data pipeline and training + evaluation of GradientBoostingRegressor as pipeline test


- test_data_pipeline.py: unit tests for data pipeline


- encoders.py: collection of data encoders I was testing in pipelines with the models. Encoders follow (and partly inherit) sklearn transformer protocol.


- neural_network.py: training + evaluation of simple neural network


- eval_tools.py: helper for evaluation and plotting.

### Chosen metrics:

- R2 score, as it compares the model with dummy model predicting mean it ais a good metric for regression task


- Mean Poisson deviance: the number of calls can be seen as approximately Poisson distributed, allthough with varying parameters in time. We are implicitly estimating these parameters.
 

- I evaluated both of these metrics also for the "Mean-Lookup model", which just takes for a given day a mean across past years as prediction. Our models performed better than the mean-lookup model.


- Outlook: to reduce variance in the target it probably makes sense to compare predictions and targets accumulated over some larger time window




