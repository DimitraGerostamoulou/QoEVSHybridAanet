# QoEVSHybridAanet
Machine Learning-Based QoE Prediction in In- Flight Video over Hybrid AANETs

The code is written in Python following a case study using combined methological
approach of quantitative and qualitative research methods using a six-phase sectioning.

Phase A: QoS data generation
Phase B: Calculation of QoE based on IQX Hypothesis
Phase C: Data normalization 
Phase D: Data preprocessing
Phase E: Training
Phase F: Evaluation of the forecasting model using QA metrics 


Each supported condition scenario reflects the behaviour of an IFC over hybrid AANET architecture, depending on adjustments in the QoS metrics. Thus, the functionality of the proposed model portrays three scenario types:

•	Positive scenario: Conditions are ideal with no variations or sudden negative alterations. All QoS parameters are optimal, with bandwidth and buffer ratio having a constant and smooth upward trend, and the rest following a downward trend. 
•	Negative scenario: QoS indicators are featured and adjusted based on interventions applied in a way that congestion and other low performance conditions are obvious.
•	Provocative scenario: This type of scenario includes cases where there are severe fluctuations in the trends of the network performance indicators. Behaviour is irregular with unpredictable events.

You will be asked to give as input the scenario type: positive/negative/provocative, 
how many samples you want to be generated and to choose between the available loss functions.

Output is a series of graphs and results of QA metrics: MAW, RMSE and R2
