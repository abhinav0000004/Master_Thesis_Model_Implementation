# Master Thesis – 3D Human Pose Estimation Model Implementation

This repository contains modified implementation of five popular transformer-based models for 3D Human Pose Estimation. It was developed as part of my master's thesis to compare joint angle accuracy across different models using real-world video data.

---

## 📁 Models Included

This repo integrates the following five models:

| Model Name               | Original Repository |
|--------------------------|---------------------|
| StridedTransformer-Pose3D | https://github.com/Vegetebird/StridedTransformer-Pose3D |
| GraphMLP                 | https://github.com/Vegetebird/GraphMLP |
| MHFormer                 | https://github.com/Vegetebird/MHFormer |
| MotionAGFormer           | https://github.com/TaatiTeam/MotionAGFormer |
| PoseFormerV2             | https://github.com/QitaoZhao/PoseFormerV2 |

> ✅ Please check the original repositories for folder structure, dependencies, and pretrained weights.  
> 💡 Use this repository for the actual code implementation as it includes necessary modifications for unified processing and evaluation.

---

## 🔧 Folder Structure

```bash
Master_Thesis_Model_Implementation/
│
├── model_name/                # One folder for each model (e.g., PoseFormerV2/)
│   └── demo/
│       ├── video/             # Input videos go here
│       ├── output/            # Output predictions (2D & 3D) saved here
│       └── vis.py             # Script to run pose estimation
│
├── data/
│   └── Ground Truth/          # Ground truth files go here (for evaluation)
│
├── Calculate_Angles.py        # Script to compute joint angles
├── Calculate_MPJAE.py         # Script to compute MPJAE metric
└── README.md
```

---

## 🚀 How to Run

### 1. Setup

- Clone the repository:

```bash
git clone https://github.com/abhinav0000004/Master_Thesis_Model_Implementation.git
cd Master_Thesis_Model_Implementation
```

- Install required libraries as per the original model repositories. Each model may have its own dependencies.

> ⚠️ There's no `requirements.txt` here because dependencies vary model to model. Refer to the original repositories for exact versions.

### 2. Download Pretrained Weights

- Download the weights from the respective original repos.
- Place them in the correct folder (e.g., `checkpoints/`, `weights/`, etc.) as described in those repos.

### 3. Add Input Videos and Ground Truth

- Place your `.mp4` videos inside:  
  `model_name/demo/video/`

- Place your **ground truth** files inside:  
  `data/Ground Truth/`

> 🎥 You can process **multiple videos** at once — all files inside the video folder will be handled automatically.

### 4. Run Pose Estimation

Run the following command based on the model you’re using:

```bash
python model_name/demo/vis.py
```

This will generate:
- 2D keypoints: `demo/output/videoName/xxx_2D.npz`
- 3D keypoints: `demo/output/videoName/xxx_3D.npz`

If you face any errors:
- Double-check folder structure and naming
- Ensure weights are correctly placed
- Refer to the original model’s README for help

### 5. Calculate Angles
- Execute `Calculate_Angles.py` to compute various angles.
   - The results are saved directly in the Data folder.
   - Individual Action Files: For each action, an individual file is created that contains the computed angles.
   - Consolidated Files: Two consolidated files are created:
   - Ground_Truth_Angles_Processed/Model_Angles_Processed: Contains data processed from Model's output, where the first frame's data has been subtracted from the rest of the frames in each action sequence to reduce initial bias.
   - Ground_Truth_Angles/Model_Angles: Absolute Angles.

### 6. Evaluate MPJAE
   - Run `Calculate_MPJAE.py` to calculate the Mean Per Joint Angle Error (MPJAE) between ground truth and Model's predicted results. You can choose whether you want to use processed angles or absolute angles and modify the code accordingly (For thesis I used processed angles). The output will be stored in the Data folder in Excel format.

**NOTE**: All five models share a similar flow.
---

## 🙏 Acknowledgements

Special thanks to the authors and repositories that made this project possible:

- StridedTransformer-Pose3D: https://github.com/Vegetebird/StridedTransformer-Pose3D
- GraphMLP: https://github.com/Vegetebird/GraphMLP
- MHFormer: https://github.com/Vegetebird/MHFormer
- MotionAGFormer: https://github.com/TaatiTeam/MotionAGFormer
- PoseFormerV2: https://github.com/QitaoZhao/PoseFormerV2
---

## 📜 License

This repository is licensed under the **MIT License**.  
Please refer to each model’s original license in their respective repositories.

---
