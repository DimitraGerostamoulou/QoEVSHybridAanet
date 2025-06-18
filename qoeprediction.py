import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import Huber
from tensorflow.keras.layers import Attention

def generateQosValues(scenario, time_steps, bandwidth, latency, rtt, jitter, packet_loss, buffer_ratio):
    if scenario == "negative":    
        for t in range(1, time_steps):

            worsenFactor = np.random.normal(0, 0.3)
            lowRandomFactor = np.random.uniform(0.01, 0.05)
            bigRandomFactor = np.random.uniform(0.5, 1.0)
            hugeRandomFactor = np.random.uniform(5, 15)
        
            #using min&max to secure values from unexpected ones
            newBandwidth = max(0, bandwidth[-1] - 0.1 * bigRandomFactor + worsenFactor)
            if np.random.random() > 0.9:  #random drops
                newBandwidth -= hugeRandomFactor

            newLatency = max(0, latency[-1] + 0.2 * (1 / max(1, newBandwidth)) + worsenFactor)
            newRTT = max(0, rtt[-1] + 0.1 * (1 / max(1, newBandwidth)) + worsenFactor) 

            if np.random.random() > 0.95:
               jitterFactor = hugeRandomFactor #random high spikes for jitter
            else:
                jitterFactor = 0
            newJitter = max(0, jitter[-1] + bigRandomFactor + jitterFactor)
  

            newPacketLoss = min(1, packet_loss[-1] + 0.02 * newLatency)
            if np.random.random() > 0.95:
                newPacketLoss += bigRandomFactor #random congestion

            newBufferRatio = max(0, buffer_ratio[-1] - 0.02 * newLatency + 0.01 * newBandwidth + worsenFactor)
            if np.random.random() > 0.95:  #random flush
                newBufferRatio = max(0, newBufferRatio - bigRandomFactor)

            bandwidth.append(newBandwidth)
            latency.append(newLatency)
            rtt.append(newRTT)
            jitter.append(newJitter)
            packet_loss.append(newPacketLoss)
            buffer_ratio.append(newBufferRatio)

    elif scenario == "positive":
        for t in range(1, time_steps):
            
            improvingFactor = np.random.normal(0, 0.1)
            lowRandomFactor = np.random.uniform(0.01, 0.05)
            bigRandomFactor = np.random.uniform(0.1, 0.5)
            
            #using min&max to secure values from unexpected ones
            newBandwidth = max(0, bandwidth[-1] + 0.2 * bigRandomFactor + improvingFactor)
            newLatency = max(0, latency[-1] - 0.2 * (1 / max(1, newBandwidth)) )
            newRTT = max(0, rtt[-1] - 0.1 * (1 / max(1, newBandwidth)))
            newJitter = max(0, jitter[-1] - bigRandomFactor)
            newPacketLoss = max(0, packet_loss[-1] - lowRandomFactor)
            if np.random.random() > 0.95:  #random recovery
                newPacketLoss = 0
               
            newBufferRatio = min(1, buffer_ratio[-1] + 0.05 * newBandwidth - 0.02 * newLatency + lowRandomFactor)
    
            bandwidth.append(newBandwidth)
            latency.append(newLatency)
            rtt.append(newRTT)
            jitter.append(newJitter)
            packet_loss.append(newPacketLoss)
            buffer_ratio.append(newBufferRatio)

    else: #provocative scenario
        for t in range(1, time_steps):
                    
            minorFluctuationFactor = np.random.uniform(-5, 5)
            majorFluctuationFactor = np.random.uniform(-10, 10) 
            lowRandomFactor = np.random.uniform(0.1, 0.3) 
            bigRandomFactor = np.random.uniform(20, 50)   
            recoveryFactor = np.random.uniform(0, 1)      

            #using min&max to secure values from unexpected ones
            newBandwidth = max(0, bandwidth[-1] + majorFluctuationFactor + np.random.uniform(-30, 30)) #fluctuations
            if np.random.random() > 0.7:  #often high drops
                newBandwidth = max(0, newBandwidth - bigRandomFactor)

           
            newLatency = max(0, latency[-1] + 0.1 * max(1, newBandwidth) + minorFluctuationFactor)
            if np.random.random() > 0.8:  
                newLatency = 10 * bigRandomFactor  #fluctuations           

            newRTT = max(0, rtt[-1] + 0.1 * (1 / max(1, newBandwidth)) + majorFluctuationFactor )
            newJitter = max(0, jitter[-1] + minorFluctuationFactor )
            if np.random.random() > 0.9:  #often high spikes
                newJitter += bigRandomFactor

            newPacketLoss = min(1, packet_loss[-1] + minorFluctuationFactor * 0.01 )
            if np.random.random() > 0.5:  #bursts of severe loss
                newPacketLoss = min(1, newPacketLoss + lowRandomFactor)

            newBufferRatio = max(0, buffer_ratio[-1] + 0.03 * newBandwidth - 0.03 * newLatency + minorFluctuationFactor)
            if np.random.random() > 0.5:  #random but unstable recovery
                newBufferRatio = recoveryFactor

            bandwidth.append(newBandwidth)
            latency.append(newLatency)
            rtt.append(newRTT)
            jitter.append(newJitter)
            packet_loss.append(newPacketLoss)
            buffer_ratio.append(newBufferRatio)





#main
print("\nQuality Assurance of LSTM multivariate timeseries for forecasting QoE in hybrid AANET\n")

scenario = input("Please provide network scenario type [positive/negative/provocative]: ").strip().lower()
samples = int(input("Please provide number of samples: "))
functionLoss = input("Please provide LSTM function loss [mse/huber/mae]: ").strip().lower()

if scenario not in ["positive", "negative", "provocative"] or not isinstance(samples, (int)) or functionLoss not in ["mse", "huber", "mae"]:
    print("Invalid input. Try again")
else: 
    print("  Phase A: Creating QoS dataset   ")
    print("------------------------------------------------------------------------------")

    np.random.seed(42)
    
    bandwidth = [np.random.uniform(0.002, 1000)]  
    latency = [np.random.uniform(20, 750)]  
    rtt = [np.random.uniform(40, 1500)]  
    jitter = [np.random.uniform(1, 50)]  
    packet_loss = [np.random.uniform(0.01, 0.10)]  
    buffer_ratio = [np.random.uniform(0.04, 1.5)]  

    generateQosValues(scenario, samples, bandwidth, latency, rtt, jitter, packet_loss, buffer_ratio)

    data = pd.DataFrame({
        'Bandwidth': bandwidth,
        'Latency': latency,
        'RTT': rtt,
        'Jitter': jitter,
        'Packet loss': packet_loss,
        'Buffer ratio': buffer_ratio
    })

    print("Done\n")
    print("  Phase B: Calculation of synthetic QoE (ground truth) using IQX hypothesis   ")
    print("------------------------------------------------------------------------------")

    # Exponential QoE calculation based on IQX hypothesis
    alpha = 5  #max QoE value
    beta = 0.01  
  
    QoS = data['Latency'] + data['Jitter'] + data['Packet loss']
    exponent = -beta * QoS
    base = np.exp(exponent)
    buffer_component = 0.5 * data['Buffer ratio'] #positive impact to qoe
    bandwidth_component = 0.25 * data['Bandwidth'] #positive impact to qoe but less postive than buffer_ratio
    gamma = buffer_component + bandwidth_component 
    data['QoE'] = (alpha * base) + gamma

    correlations = data[['Bandwidth', 'Latency', 'RTT', 'Jitter', 'Packet loss', 'Buffer ratio', 'QoE']].corr()
    print(correlations['QoE'])  # Correlation of QoE with each QoS parameter

    #scale again to [1,5] 
    QoEmin = data['QoE'].min()
    QoEmax = data['QoE'].max()
    data['QoE'] = (data['QoE'] - QoEmin) / (QoEmax - QoEmin) * 4 + 1  

    #distribution of QoS Parameters and QoE in a grid before LSTM training
    fig, axes = plt.subplots(3, 3, figsize=(10, 10), squeeze=False)
    columns = ['Bandwidth', 'Latency', 'RTT', 'Jitter', 'Packet loss', 'Buffer ratio']
    for i, col in enumerate(columns):
        row, col_idx = divmod(i, 3)
        axes[row, col_idx].plot(data[col].values[:1000])
        axes[row, col_idx].set_title(col)
        axes[row, col_idx].set_xlabel('Samples')
        axes[row, col_idx].set_ylabel('Value')

    if len(columns) < 9:
        for i in range(len(columns), 9):
            row, col_idx = divmod(i, 3)
            fig.delaxes(axes[row, col_idx])

    plt.tight_layout()
    plt.show()

    # Plot QoE before LSTM training
    plt.figure(figsize=(12, 6))
    plt.plot(data['QoE'], label="QoE Before Training (IQX)", color='blue')
    plt.title("QoE Over Time Before LSTM Training")
    plt.xlabel("Time Steps")
    plt.ylabel("QoE Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    print("Done\n")
    print("  Phase C: Normalising QoS data with Z-Score and QoE data with MinMax[1,5]   ")
    print("-----------------------------------------------------------------------------")

    scaler = StandardScaler() #z-score normalization
    scaledQoSdata = scaler.fit_transform(data.iloc[:, :-1]) 

    scaler= MinMaxScaler(feature_range=(1, 5)) #min-max normalization
    scaledQoEdata = scaler.fit_transform(data[['QoE']])  

    #combine and convert to array
    dataScaled = np.hstack((scaledQoSdata, scaledQoEdata))


    print("Done\n")
    print("  Phase D: Data preprocessing   ")
    print("-----------------------------------------------------------------------------")
    
    sequence_length = 10 
    X, y = [], []
    for i in range(len(dataScaled) - sequence_length):
        X.append(dataScaled[i:i+sequence_length, :-1])  
        y.append(dataScaled[i+sequence_length, -1]) 

    X = np.array(X)
    y = np.array(y)

    print("--> Splitting data to 80-10-10\n")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Done\n")
    print("  Phase E: LSTM setup & training  ")
    print("-----------------------------------------------------------------------------")


    #defining model
    model = Sequential()
    model.add(LSTM(100, input_shape=(sequence_length, X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(1))

    
    if functionLoss == "huber":
        model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber(delta=0.5), metrics=['mae'])
    elif functionLoss == "mae":
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mae", metrics=['mae'])
    else: 
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=['mae'])

    print("--> LSTM training starts...\n")

    epochs = 50
    batch_size = 32
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    #fit model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[reduce_lr])

    predictions = model.predict(X_test)

    #change predictions and ground truth to original scale sto that the have the same shape
    y_testOrigin = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    predictionsOrigin = scaler.inverse_transform(predictions).flatten()

    print("Done\n")
    print("  Phase F: LSTM Evaluation & visualisation ")
    print("-----------------------------------------------------------------------------")

    loss, mae = model.evaluate(X_test, y_test)
    rmse = np.sqrt(mean_squared_error(y_testOrigin, predictionsOrigin))
    r2 = r2_score(y_testOrigin, predictionsOrigin)
    print(f"Mean Absolute Error-ΜΑΕ: {mae}") 
    print(f"Root Mean Square-RMSE: {rmse}") 
    print(f"R² Score: {r2}") 



    predictionErrors = y_testOrigin - predictionsOrigin
    plt.figure(figsize=(8, 6))
    plt.hist(predictionErrors, bins=50, alpha=0.7)
    plt.title("Prediction Errors Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.show()


    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


    plt.figure(figsize=(8, 6))
    plt.scatter(y_testOrigin, predictionsOrigin, color='blue', alpha=0.6, label='Predictions vs Actual', edgecolors='black', linewidth=1.2)
    plt.plot([1, 5], [1, 5], 'r--', label='Ideal', linewidth=2.5)  
    plt.xlim(1, 5)
    plt.ylim(1, 5)
    plt.title('Scatter Plot of Actual vs Predicted QoE')
    plt.xlabel('Actual QoE', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted QoE', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3, linewidth=1.2)
    plt.show()


    print("Done\n")
    print("-----------------------------------------------------------------------------")
    print("  Execution completed!\n  ")
    print("  Exiting\n  ")
    