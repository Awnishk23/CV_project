markdown
# Siamese Neural Network for Face Verification

A real-time face verification system built using Siamese Neural Networks with TensorFlow and OpenCV. This system can learn to verify if two face images belong to the same person by computing similarity scores between facial embeddings.

##  Features

- **Real-time face verification** using webcam input
- **Siamese Neural Network architecture** for one-shot learning
- **Custom data collection** interface for building personalized datasets
- **Live verification system** with adjustable thresholds
- **GPU acceleration** support with TensorFlow

##  Requirements


pip install tensorflow>=2.8.0
pip install opencv-python
pip install numpy
pip install matplotlib
pip install uuid


##  Project Structure


├── data/
   ├── positive/     
   ├── negative/         
├── application_data/
   ├── input_image/
   └── verification_images/
├── training_checkpoints/
└── siamesemodelv2.h5


##  How It Works

### 1. **Siamese Network Architecture**

The model consists of:
- **Embedding Network**: CNN that extracts 4096-dimensional feature vectors
  - 4 Convolutional blocks with increasing filter sizes (64→128→128→256)
  - MaxPooling layers for dimensionality reduction
  - Dense layer with sigmoid activation for final embeddings

- **Distance Layer**: Custom L1 distance calculation between embeddings
- **Classification Head**: Binary classifier to determine if faces match

### 2. **Training Process**

The network learns to:
- **Minimize distance** between anchor and positive pairs (same person)
- **Maximize distance** between anchor and negative pairs (different people)
- Use **binary cross-entropy loss** with Adam optimizer

##  Quick Start

### 1. Data Collection


# Run the data collection script
# Press 'a' to capture anchor images
# Press 'p' to capture positive images  
# Press 'q' to quit
python collect_data.py


### 2. Prepare Negative Samples

Download the [LFW dataset](http://vis-www.cs.umass.edu/lfw/) and place it in the project directory. The script will automatically move images to the negative folder.

### 3. Train the Model


# Train for 20 epochs
EPOCHS = 20
train(train_data, EPOCHS)


### 4. Real-time Verification


# Run verification system
# Press 'v' to verify current face
# Press 'q' to quit
python verify_face.py


## ⚙ Configuration

### Verification Thresholds

- **Detection Threshold** (0.5): Minimum similarity score for positive detection
- **Verification Threshold** (0.5): Proportion of positive detections required for verification

### Model Parameters

- **Input Size**: 100×100×3 RGB images
- **Batch Size**: 16
- **Learning Rate**: 1e-4
- **Embedding Dimension**: 4096

##  Model Performance

The model uses precision and recall metrics during training:
- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual positive cases

##  Customization

### Adjust Network Architecture


def make_embedding():
    # Modify layers, filters, or dimensions
    inp = Input(shape=(100,100,3), name='input_image')
    # Add/remove layers as needed
    return Model(inputs=[inp], outputs=[d1], name='embedding')


### Modify Verification Logic


def verify(model, detection_threshold, verification_threshold):
    # Adjust thresholds based on your requirements
    # detection_threshold: 0.1-0.9 (lower = more lenient)
    # verification_threshold: 0.1-0.9 (lower = easier verification)


##  Use Cases

- **Access Control Systems**
- **Identity Verification**
- **Security Applications**
- **Attendance Systems**
- **Personal Photo Organization**

##  Important Notes

1. **Data Quality**: Ensure good lighting and clear face visibility during data collection
2. **Dataset Balance**: Collect similar numbers of anchor and positive samples
3. **GPU Memory**: The script includes GPU memory growth configuration for optimal performance
4. **Model Saving**: Model is saved as 'siamesemodelv2.h5' with custom objects

##  Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


