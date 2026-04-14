# radar-obstacle-classification
Radar obstacle classification and hazard level assessment using traditional machine learning.
Radar Obstacle Classification and Hazard Assessment
This is the official open-source repository for the paper "Traditional Machine Learning based Radar Obstacle Classification and Hazard Level Assessment".
We have preprocessed the original large-scale CARRADA dataset into a small, easy-to-use feature dataset, allowing you to reproduce all the experimental results in the paper with just one click, without downloading the 10GB+ raw radar data.


📁 Project Structure


File	Description
load_carrada.py	Raw dataset preprocessing script. Extract features from the original CARRADA dataset to generate the small CSV file.
train_models.py	Model training & evaluation script. Load the preprocessed dataset and reproduce all paper results.
carrada_radar_4d_dataset.csv	Preprocessed small dataset (~1MB). Contains 16683 valid samples with 4D radar features and labels.
requirements.txt	Python dependencies list.
results/paper_results.txt	Generated after running the training script. Contains all 5 tables and statistical test results for your paper.

🚀 Quick Start
1. Clone the repository

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2. Install dependencies

pip install -r requirements.txt
3. One-click Reproduce Paper Results
You don't need the original large dataset! We have already preprocessed it for you. Just run:

python train_models.py
This will:
1.Load the preprocessed small dataset automatically
2.Run 5-fold cross-validation for 4 models (KNN/SVM/RF/XGBoost)
3.Evaluate both obstacle classification and hazard assessment tasks
4.Generate all the tables you need for your paper in results/paper_results.txt


(Optional) Reprocess from Raw Dataset
If you want to process the raw CARRADA dataset by yourself:
1.Download the original CARRADA dataset from Kaggle:
https://www.kaggle.com/datasets/ghammoud/carrada
2.Modify the DATA_ROOT path in load_carrada.py to point to your extracted dataset folder.
3.Run the preprocessing script:

python load_carrada.py
This will generate the same carrada_radar_4d_dataset.csv file we provided.


📊 Experimental Results
After running train_models.py, you will get results/paper_results.txt, which contains:
1.Table 1: Dual-task performance of all models (5-fold cross-validation)
2.Table 2: Feature importance ranking (based on XGBoost)
3.Table 3: Confusion matrix for obstacle classification
4.Table 4: Per-class performance for hazard level assessment
5.Table 5: Model inference time
6.Statistical Significance Test: McNemar's test results

📄 Dataset Description
The original CARRADA dataset contains ~10GB of raw radar range-Doppler and range-Angle heatmaps. We extracted 4 key features from the raw data:
•distance: Target distance
•velocity: Relative velocity
•angle: Azimuth angle
•rcs: Radar Cross Section (echo intensity)
This reduces the dataset size from 10GB+ to only ~1MB, making it extremely easy to open-source and share.



📝 Citation
If you use this code or dataset in your research, please cite our paper:

@article{your_paper_2026,
  title={Traditional Machine Learning based Radar Obstacle Classification and Hazard Level Assessment},
  author={XUKE},
  journal={//},
  year={2026}
}


📜 License
This project is licensed under the MIT License - see the LICENSE file for details.


📧 Contact
Email: 19118551049@163.com
If you have any questions, please open an issue or contact the authors.
