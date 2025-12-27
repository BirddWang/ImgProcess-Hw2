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

from model import ResNet18


class ModelDemoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ResNet18 Model Demo")
        self.setGeometry(100, 100, 1200, 700)

        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNet18().to(self.device)

        # Load pretrained weights
        try:
            weights_path = Path(__file__).parent / 'model' / 'weight.pth'
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.model.eval()
        except Exception as e:
            print(f"Warning: Could not load model weights: {e}")

        # Image preprocessing pipeline (CIFAR10 with RGB channels)
        self.transform = transforms.Compose([
            # Convert RGBA to RGB if needed
            transforms.Lambda(lambda img: img.convert('RGB') if img.mode == 'RGBA' else img),
            # Resize to 32x32 (CIFAR-10 standard)
            transforms.Resize((32, 32)),
            # Convert to tensor
            transforms.ToTensor(),
            # Normalize with CIFAR-10 statistics
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        # CIFAR10 class names
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']

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

        # Load and Show Image Button
        self.load_btn = QPushButton("2.1 Load and Show Image")
        self.load_btn.setMinimumHeight(50)
        self.load_btn.clicked.connect(self.load_image)
        left_panel.addWidget(self.load_btn)

        # Show Architecture Button
        self.arch_btn = QPushButton("2.2 Show Architecture")
        self.arch_btn.setMinimumHeight(50)
        self.arch_btn.clicked.connect(self.show_architecture)
        left_panel.addWidget(self.arch_btn)

        # Show Acc Loss Button
        self.loss_btn = QPushButton("2.3 Show Acc Loss")
        self.loss_btn.setMinimumHeight(50)
        self.loss_btn.clicked.connect(self.show_loss_acc)
        left_panel.addWidget(self.loss_btn)

        # Inference Button
        self.predict_btn = QPushButton("2.4 Inference")
        self.predict_btn.setMinimumHeight(50)
        self.predict_btn.clicked.connect(self.predict_image)
        left_panel.addWidget(self.predict_btn)

        left_panel.addStretch()

        # Wrap left panel with margins
        left_container = QWidget()
        left_container.setLayout(left_panel)
        left_container.setMaximumWidth(300)

        # Right panel: Image display and prediction result
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        # Scroll area for image
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid gray; background-color: white;")
        self.image_label.setMinimumSize(500, 500)

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)

        right_panel.addWidget(scroll_area, 1)

        # Prediction Result Label (below image)
        self.result_label = QLabel("Predict: -")
        result_font = QFont()
        result_font.setPointSize(14)
        result_font.setBold(True)
        self.result_label.setFont(result_font)
        self.result_label.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(self.result_label)

        right_container = QWidget()
        right_container.setLayout(right_panel)

        # Main layout assembly
        main_layout.addWidget(left_container)
        main_layout.addWidget(right_container, 1)

        central_widget.setLayout(main_layout)

    def load_image(self):
        """Load an image from file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            str(Path(__file__).parent.parent / "Hw2_2" / "Q2_inference_img"),
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            self.loaded_image_path = file_path
            image = Image.open(file_path).convert('RGB')

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
            summary(self.model, (3, 32, 32))
        return f.getvalue()

    def show_loss_acc(self):
        """Display the Loss & Accuracy curves"""
        try:
            loss_acc_path = Path(__file__).parent / "Loss&Acc.png"
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
        """Predict the object in the loaded image and show probability distribution"""
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
            class_name = self.class_names[predicted_class]
            self.result_label.setText(f"Predict: {predicted_class} ({class_name})")

            # Display probability distribution as bar plot
            self._show_probability_distribution(probs)

        except Exception as e:
            print(f"Error during prediction: {e}")

    def _show_probability_distribution(self, probabilities):
        """Display the probability distribution as a bar plot"""
        fig, ax = plt.subplots(figsize=(12, 6))

        classes = list(range(10))
        colors = ['red' if i == np.argmax(probabilities) else 'blue' for i in range(10)]

        ax.bar(classes, probabilities, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title('Prediction Probability Distribution (CIFAR10)', fontsize=14, fontweight='bold')
        ax.set_xticks(classes)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
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
