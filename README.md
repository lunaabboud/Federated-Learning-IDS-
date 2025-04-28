# Federated-Learning-IDS-
# Federated Learning Intrusion Detection System (FL-IDS)

A distributed, privacy-preserving intrusion detection system that uses federated learning to train machine learning models across multiple nodes without sharing sensitive data.

## 📋 Overview

This project implements a Federated Learning approach for Network Intrusion Detection Systems. Instead of centralizing all network data (which may contain sensitive information), the system uses Flower framework to train models across distributed clients while only sharing model parameters.

### Key Features

- **Privacy-Preserving Learning**: Train effective IDS models without centralizing sensitive network data
- **Distributed Architecture**: Docker-based containerized deployment for server and clients
- **Real-time Monitoring**: Capture and visualize training progress
- **Performance Analysis**: Generate comprehensive visualizations and metrics reports

## 🔧 System Architecture

The system consists of:

1. **Server Node**: Coordinates the federated learning process and evaluates global model performance
2. **Client Nodes**: Train models on local data and share only model parameters
3. **Visualization Tools**: Monitor training progress and generate performance reports

## 📊 Visualization Features

The system includes comprehensive visualization tools:
- Global model accuracy tracking
- Client-specific performance metrics
- Loss convergence analysis
- Before/after improvement measurements
- Detailed summary reports

## 🚀 Installation & Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Pandas, Scikit-learn, TensorFlow, and other dependencies (see requirements.txt)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/federated-learning-ids.git
   cd federated-learning-ids
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Build Docker containers:
   ```bash
   docker-compose build
   ```

## 💻 Usage

### Running the System

1. **Start the federated learning system with visualization**:
   ```bash
   python3 run_visualization.py
   ```
   This will start the server and client containers, capture logs, and generate visualizations automatically.

2. **Visualize existing logs**:
   ```bash
   python3 visualize_existing.py
   ```
   This allows you to choose from previously captured logs and generate new visualizations.

3. **Manual execution**:
   ```bash
   # Start the federated learning system
   docker-compose up
   
   # In a separate terminal, to capture logs:
   python3 capture_logs.py --output logs/my_training.log
   
   # Generate visualizations after training:
   python3 visualize.py --log logs/my_training.log --output visualizations
   ```

### Data Preparation

The system expects data in CSV format with features and a "Label" column indicating benign or malicious traffic. Use the `preprocess.py` script to prepare your own data:

```bash
python3 preprocess.py
```

## 📈 Results & Outputs

After running the system, you'll find:

- **Log files**: Detailed logs of the training process in the `logs/` directory
- **Visualizations**: Performance charts and diagrams in the `visualizations/` directory
- **Summary reports**: Text-based analysis of model performance

## 🔍 Project Structure

```
federated-learning-ids/
├── client.py               # Client node implementation
├── server.py               # Server node implementation
├── visualize.py            # Visualization generation tool
├── visualize_existing.py   # Tool to visualize previous logs
├── run_visualization.py    # Main script to run system with visualization
├── capture_logs.py         # Log capture utility
├── preprocess.py           # Data preprocessing script
├── Dockerfile              # Server container definition
├── Dockerfile.client       # Client container definition
├── docker-compose.yml      # Container orchestration
├── requirements.txt        # Python dependencies
├── data/                   # Data directory (not included in repo)
├── logs/                   # Training logs
└── visualizations/         # Generated visualizations
```

## 🔧 Customization

### Adding New Clients

1. Update `docker-compose.yml` to add new client containers
2. Prepare client-specific datasets in the `data/` directory

### Model Architecture

Modify the `create_model()` function in `client.py` and `server.py` to experiment with different model architectures.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

- [Flower: A Friendly Federated Learning Framework](https://flower.dev/)
- [TensorFlow](https://www.tensorflow.org/)
- [Scikit-learn](https://scikit-learn.org/)
