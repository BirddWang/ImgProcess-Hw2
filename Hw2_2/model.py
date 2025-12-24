import torch
import torch.nn as nn
from torchvision import models

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        # 載入原本的 ResNet18 架構 (不使用預訓練權重，因為我們要改結構)
        self.model = models.resnet18(weights=None)
        
        # 1. 修改第一層卷積 (對應圖片中的第 1 點)
        # 原本是: kernel_size=7, stride=2, padding=3
        # 修改為: kernel_size=3, stride=1, padding=1 (為了保持 32x32 的特徵圖大小)
        self.model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 2. 移除 Max Pooling (對應圖片中的第 1 點 remove max pooling)
        # 原本 ResNet 在 conv1 後面接了一個 maxpool，會讓圖片變更小。
        # 在 PyTorch 中最簡單的移除方法是用 nn.Identity() 取代它，這樣訊號就會直接通過不做處理。
        self.model.maxpool = nn.Identity()
        
        # 3. 修改全連接層 (對應圖片中的第 2 點)
        # ResNet18 最後一層的輸入特徵數通常是 512
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        # 因為我們是把修改後的 resnet18 存成 self.model
        # 所以 forward 只需要呼叫 self.model(x) 即可
        return self.model(x)

# --- 測試與顯示結構 (對應圖片中的 "Run the function to show the structure in the terminal") ---
if __name__ == "__main__":
    # 實例化模型
    my_resnet = ResNet18(num_classes=10)
    
    # 顯示模型結構
    print(my_resnet)
    
    # 簡單測試一下輸入輸出是否正常 (Batch Size=1, Channel=3, Height=32, Width=32)
    input_tensor = torch.randn(1, 3, 32, 32)
    output = my_resnet(input_tensor)
    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}") # 應該要是 [1, 10]