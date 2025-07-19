# pyQT-CSITool
This is an open-source project developed in Python for real-time CSI signal collection, labeling, and playback.
<img width="762" height="744" alt="image" src="https://github.com/user-attachments/assets/2b7d16ce-e594-4691-8ef4-46c74b5fcb88" />
# Run
```
python main.py
```

# Download Dataset
All raw data (".dat" file) can be downloaded at：
* https://pan.baidu.com/s/1La99unNH-6KYhxrV5LsyVQ?pwd=c46j (Full Dataset)
* https://ieee-dataport.org/documents/tcs-fall (Partial Dataset)

# Package dependencies
CSITool can be accessed from https://github.com/dhalperi/linux-80211n-csitool-supplementary.

Python 3.8+ is required, and the main dependency packages are as follows:
```
pip install jupyter
pip install torch==1.11.0 torchvision==0.12.0
pip install numpy==1.23.5 numba==0.56.4 PyWavelets=1.4.1 scipy==1.9.3 matplotlib==3.6.2 pandas==1.5.2
pip install onnxruntime pyqtgraph==0.11.1 PyQt5==5.15.7
```

# Cite
```
@article{zhou2024tcs,
  title={TCS-Fall: Cross-individual fall detection system based on channel state information and time-continuous stack method},
  author={Zhou, Ziyu and Liu, Zhaoqing and Liu, Yujie and Zhao, Yan and Wang, Jiarui and Zhang, Bowen and Xia, Youbing and Zhang, Xiao and Li, Shuyan},
  journal={Digital Health},
  volume={10},
  pages={20552076241259047},
  year={2024},
  publisher={SAGE Publications Sage UK: London, England}
}
```
