import pandas as pd

def trainTable(file = "templog.CSV"):
    mapping = {
        "Total CPU Usage [%]" : "CPU Usage", 
        "Core Clocks (avg) [MHz]" : "CPU Clock", 
        "Core C0 Residency (avg) [%]" : "CPU C0", 
        "Core C1 Residency (avg) [%]" : "CPU C1", 
        "Core C6 Residency (avg) [%]" : "CPU C6", 
        "CPU Core [°C]" : "CPU Core Temp", 
        "CPU SOC [°C]" : "CPU SOC Temp", 
        "CPU Package Power [W]" : "CPU Power", 
        "GPU Temperature [°C]" : "GPU Temp", 
        "GPU Clock [MHz]" : "GPU Clock", 
        "GPU ASIC Power [W]" : "GPU Power", 
        "GPU Memory Clock [MHz]" : "GPU Memory Clock", 
        "GPU Utilization [%]" : "GPU Usage", 
        "Read Rate [MB/s]" : "Drive Read", 
        "Write Rate [MB/s]" : "Drive Write", 
        "Current DL rate [KB/s]" : "Network Download", 
        "Current UP rate [KB/s]" : "Network Upload"
    }
    trainingdata = pd.read_csv(file, encoding = "latin1")
    trainingdata.drop(trainingdata.index[-2:], inplace=True)
    trainingdata = trainingdata.get(["Total CPU Usage [%]", "Core Clocks (avg) [MHz]", "Core C0 Residency (avg) [%]", "Core C1 Residency (avg) [%]", "Core C6 Residency (avg) [%]", "CPU Core [°C]", "CPU SOC [°C]", "CPU Package Power [W]", "GPU Temperature [°C]", "GPU Clock [MHz]", "GPU ASIC Power [W]", "GPU Memory Clock [MHz]", "GPU Utilization [%]", "Read Rate [MB/s]", "Write Rate [MB/s]", "Current DL rate [KB/s]", "Current UP rate [KB/s]"])
    trainingdata = trainingdata.rename(columns = mapping)
    trainingdata = trainingdata.astype(float)
    trainingdata['CPU Clock'] /= 400
    return trainingdata
