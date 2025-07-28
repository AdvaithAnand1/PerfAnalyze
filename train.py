import pandas as pd, normalize, requests

cachedTrainTable = None
def trainTable(file = "templog.CSV"):
    global cachedTrainTable
    if cachedTrainTable is None:    
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
        cachedTrainTable = trainingdata
    return cachedTrainTable

cachedGeekFrame = None
def getGeekFrame():
    global cachedGeekFrame
    if cachedGeekFrame is None:
        url = "https://browser.geekbench.com/processor-benchmarks.json"
        response  = requests.get(url, timeout = 10)
        response.raise_for_status()
        benchmarks = response.json()["devices"]
        geekframe = pd.DataFrame(benchmarks)
        geekframe = pd.json_normalize(benchmarks, sep = "_")
        geekframe["name"] = geekframe["name"].apply(normalize.normalizeCPUName)
        cachedGeekFrame = geekframe
    return cachedGeekFrame

def getGeekScores():
    row = normalize.getDetails(getGeekFrame(), normalize.getCPU())
    return (row.get("score").item(), row.get("multicore_score").item())

cachedPassFrame = None

def getPassFrame():
    global cachedPassFrame
    if cachedPassFrame is None:
        url = "https://www.videocardbenchmark.net/gpu_list.php"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        gpu_df = tables[1]
        gpu_df = gpu_df[["Videocard Name", "Passmark G3D Mark (higher is better)"]]
        gpu_df = gpu_df.rename(columns = {"Videocard Name": "name", "Passmark G3D Mark (higher is better)": "score"})
        gpu_df["score"] = (gpu_df["score"].astype(str).str.replace(",", "").astype(int))
        gpu_df["name"] = gpu_df["name"].apply(normalize.normalizeGPUName)
        cachedPassFrame = gpu_df
    return cachedPassFrame

def getPassScores():
    row = normalize.getDetails(getPassFrame(), normalize.getGPU())
    return row.get("score").item()


class BottleneckRNN(nn.Module):
    def __init__(
        self,
        feat_dim = 17,       # placeholder for number of features per timestep
        const_dim = 3,      # placeholder for number of constant inputs (e.g. 2)
        hidden_size=128,
        num_layers=2,
        const_hidden=32,
        fc_hidden=64,
        num_labels=7,
        bidirectional=True,
        dropout=0.2,
    ):
        super().__init__()
        # — input embedding for time‑series features
        self.input_emb = nn.Linear(feat_dim, hidden_size)

        # — stacked bidirectional LSTM
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # — small MLP branch for constant inputs
        self.const_branch = nn.Sequential(
            nn.Linear(const_dim, const_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # — classifier head: fuse RNN + constant embeddings
        rnn_output_dim = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_output_dim + const_hidden, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_labels),
            # for multi‑label use sigmoid in training/validation, 
            # but omit here if you plan to use BCEWithLogitsLoss
        )

    def forward(self, x, constants):
        """
        x:         Tensor of shape (batch, seq_len, feat_dim)\n
        constants: Tensor of shape (batch, const_dim)
        """
        # 1) embed time‑series features
        x_emb = self.input_emb(x)  # → (batch, seq_len, hidden_size)

        # 2) pass through RNN
        rnn_out, (h_n, _) = self.rnn(x_emb)
        # h_n: (num_layers * num_directions, batch, hidden_size)

        # 3) summarize sequence via final hidden states
        if self.rnn.bidirectional:
            h_forward = h_n[-2]  # last layer forward
            h_backward = h_n[-1] # last layer backward
            rnn_feat = torch.cat([h_forward, h_backward], dim=1)
        else:
            rnn_feat = h_n[-1]   # last layer

        # 4) process constants
        c_feat = self.const_branch(constants)  # → (batch, const_hidden)

        # 5) fuse and classify
        fused = torch.cat([rnn_feat, c_feat], dim=1)
        logits = self.classifier(fused)  # → (batch, num_labels)
        return logits