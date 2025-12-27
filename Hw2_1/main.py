import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QFileDialog, QScrollArea
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from torchvision import transforms
from torchsummary import summary

from model import LeNet5


class ModelDemoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LeNet5 Model Demo")
        self.setGeometry(100, 100, 1200, 700)

        # Initialize model with device detection (MPS support for Apple Silicon)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = LeNet5(activation_type='relu').to(self.device)

        # Load pretrained weights
        try:
            weights_path = Path(__file__).parent / 'model' / 'Weight_Relu.pth'
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.model.eval()
            print(f"Model loaded from {weights_path}")
        except Exception as e:
            print(f"Warning: Could not load model weights: {e}")
            print("Attempting to load from alternative paths...")

        # Image preprocessing pipeline with better handling for hand-written digits
        self.transform = transforms.Compose([
            # First convert RGBA to RGB if needed
            transforms.Lambda(lambda img: img.convert('RGB') if img.mode == 'RGBA' else img),
            # Convert to grayscale
            transforms.Grayscale(num_output_channels=1),
            # Resize to 32x32 (LeNet5 expects this)
            transforms.Resize((32, 32)),
            # Convert to tensor
            transforms.ToTensor(),
            # Invert colors if needed (for hand-written images where black is foreground)
            # This helps match MNIST training data where digits are darker
            transforms.Lambda(lambda x: 1.0 - x if x.mean() > 0.5 else x),
            # Normalize with MNIST statistics
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Store loaded image
        self.loaded_image = None
        self.loaded_image_path = None
        self.prediction_result = None

        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()

        # Left panel: Buttons
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)

        # Load Image Button
        self.load_btn = QPushButton("Load Image")
        self.load_btn.setMinimumHeight(50)
        self.load_btn.clicked.connect(self.load_image)
        left_panel.addWidget(self.load_btn)

        # Show Architecture Button
        self.arch_btn = QPushButton("1.1 Show Architecture")
        self.arch_btn.setMinimumHeight(50)
        self.arch_btn.clicked.connect(self.show_architecture)
        left_panel.addWidget(self.arch_btn)

        # Show Acc Loss Button
        self.loss_btn = QPushButton("1.2 Show Acc Loss")
        self.loss_btn.setMinimumHeight(50)
        self.loss_btn.clicked.connect(self.show_loss_acc)
        left_panel.addWidget(self.loss_btn)

        # Predict Button
        self.predict_btn = QPushButton("1.3 Predict")
        self.predict_btn.setMinimumHeight(50)
        self.predict_btn.clicked.connect(self.predict_image)
        left_panel.addWidget(self.predict_btn)

        # Prediction Result Label
        left_panel.addSpacing(20)
        self.result_label = QLabel("Predict: -")
        result_font = QFont()
        result_font.setPointSize(14)
        result_font.setBold(True)
        self.result_label.setFont(result_font)
        self.result_label.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(self.result_label)

        left_panel.addStretch()

        # Wrap left panel with margins
        left_container = QWidget()
        left_container.setLayout(left_panel)
        left_container.setMaximumWidth(300)

        # Right panel: Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid gray; background-color: white;")
        self.image_label.setMinimumSize(500, 500)

        # Scroll area for image
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)

        # Main layout assembly
        main_layout.addWidget(left_container)
        main_layout.addWidget(scroll_area, 1)

        central_widget.setLayout(main_layout)

    def load_image(self):
        """Load an image from file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            str(Path(__file__).parent / "Q1_TestData"),
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            self.loaded_image_path = file_path
            image = Image.open(file_path)

            # Store original image for prediction
            self.loaded_image = image.copy()

            # Display image
            pixmap = QPixmap(file_path)
            # Scale image to fit in label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaledToWidth(400, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)

            # Reset prediction when new image is loaded
            self.prediction_result = None
            self.result_label.setText("Predict: -")

    def show_architecture(self):
        """Display model architecture using torchsummary"""
        try:
            # Create a figure for displaying the summary
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.axis('off')

            # Capture the summary as a string
            summary_str = self._get_model_summary()

            ax.text(0.05, 0.95, summary_str, transform=ax.transAxes,
                   fontfamily='monospace', fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error showing architecture: {e}")

    def _get_model_summary(self):
        """Get model summary as a string"""
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            summary(self.model, (1, 32, 32))
        return f.getvalue()

    def show_loss_acc(self):
        """Display the Loss & Accuracy curves"""
        try:
            loss_acc_path = Path(__file__).parent / "Loss&Acc_Sigmoid.png"
            if loss_acc_path.exists():
                img = Image.open(loss_acc_path)
                plt.figure(figsize=(10, 8))
                plt.imshow(img)
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            else:
                print(f"Loss & Accuracy image not found at {loss_acc_path}")
        except Exception as e:
            print(f"Error showing Loss & Accuracy: {e}")

    def predict_image(self):
        """Predict the digit in the loaded image and show probability distribution"""
        if self.loaded_image is None:
            print("Please load an image first!")
            return

        try:
            # Ensure model is in evaluation mode (disables Dropout and BatchNorm)
            self.model.eval()

            # Preprocess image
            input_tensor = self.transform(self.loaded_image).unsqueeze(0).to(self.device)

            # Make prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                probs = probabilities[0].cpu().numpy()

            # Store prediction result
            self.prediction_result = predicted_class
            self.result_label.setText(f"Predict: {predicted_class}")

            # Display probability distribution as bar plot
            self._show_probability_distribution(probs)

        except Exception as e:
            print(f"Error during prediction: {e}")

    def _show_probability_distribution(self, probabilities):
        """Display the probability distribution as a bar plot"""
        fig, ax = plt.subplots(figsize=(10, 6))

        classes = list(range(10))
        colors = ['red' if i == np.argmax(probabilities) else 'blue' for i in range(10)]

        ax.bar(classes, probabilities, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Digit Class', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(classes)
        ax.set_ylim(0, 1)

        # Add value labels on bars
        for i, prob in enumerate(probabilities):
            ax.text(i, prob + 0.02, f'{prob:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.show()


def main():
    app = QApplication(sys.argv)
    window = ModelDemoApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
