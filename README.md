# YOLOv8 for Weed Detection and Segmentation

![GitHub banner for a weed detection project](https://i.imgur.com/7g8h4kT.png)

This project uses the **YOLOv8-seg** model to perform instance segmentation for identifying and outlining weeds in agricultural imagery. The primary goal is to distinguish between sorghum crops, grass weeds, and broad-leaf weeds to aid in precision farming and automated weed management.

The repository includes a Jupyter Notebook (`.ipynb`) that documents the entire workflow from data preparation to training and evaluation, as well as a clean Python script (`.py`) for reusability.

---

## Key Features

-   **Data Processing:** Includes scripts to convert COCO annotations to the YOLOv8 segmentation format.
-   **Model Training:** Demonstrates how to train a `YOLOv8s-seg` model from scratch on a custom dataset.
-   **Evaluation & Prediction:** Shows how to evaluate model performance and run predictions on test images.
-   **Hyperparameter Tuning:** Contains a workflow for using YOLOv8's built-in tuner to find optimal training parameters.

---

## Dataset: Sorghum Weed Dataset

This project is trained on the **Sorghum Weed Dataset**, an excellent public dataset available on Mendeley Data. It provides high-resolution images of sorghum fields with detailed annotations for three classes.

-   **Classes:** `Sorghum`, `Grasses`, `Broad-leaf weeds`
-   **Source:** The dataset was created by researchers from the University of Southern Queensland, Australia.
-   **Link:** You can access and download the dataset here: [Sorghum Weed Dataset on Mendeley Data](https://data.mendeley.com/datasets/g335n3f524/2)

Properly preparing this dataset is a key part of the project, and the included scripts automate the conversion from its native COCO format to the required YOLOv8 `.txt` format.

---

## Getting Started

Follow these steps to set up the environment and run the project.

### 1. Prerequisites

-   Python 3.8 or later
-   Access to Google Colab or a local machine with a GPU (recommended for training).
-   The source dataset downloaded from the Mendeley link above.

### 2. Installation

Clone the repository and install the required Python packages:

```bash
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name
pip install -r requirements.txt
```

*(Note: You may need to create a `requirements.txt` file containing `ultralytics` and `pycocotools`)*

### 3. Data Setup

1.  Download the `SorghumWeedDataset_Segmentation.zip` file from Mendeley Data.
2.  Create a project folder in your Google Drive (e.g., `Sorghum_Project`).
3.  Upload the `.zip` file to a subfolder within your project directory (e.g., `Sorghum_Project/source_dataset/`).
4.  Update the paths in the `.ipynb` notebook or `.py` script to point to your file locations.

---

## Usage

You can run the project using either the Jupyter Notebook or the Python script.

### Using the Notebook (`weed_segmentation.ipynb`)

Open the notebook in Google Colab or Jupyter Lab and run the cells sequentially. The notebook is documented and guides you through each step:
1.  **Setup & Dependencies:** Mounts Google Drive and installs libraries.
2.  **Data Preparation:** Unzips the dataset and converts annotations.
3.  **Training & Evaluation:** Runs baseline and augmented training runs.
4.  **Hyperparameter Tuning:** Executes the automated tuning process.

### Using the Python Script (`train.py`)

Execute the script from your terminal. It will automatically perform the data setup and then proceed with training.

```bash
python train.py
```

---

## Results

The model performs well in identifying and creating precise segmentation masks for both crops and weeds. Below are some example predictions on the test set.

*(Here you should add your own images! Take screenshots of the prediction results from your `runs/segment/predict` folder and upload them to your repository.)*

| Input Image                                    | Model Prediction                               |
| ---------------------------------------------- | ---------------------------------------------- |
| ![Test Image 1](placeholder_image_1.jpg)       | ![Prediction Image 1](placeholder_pred_1.jpg)  |
| ![Test Image 2](placeholder_image_2.jpg)       | ![Prediction Image 2](placeholder_pred_2.jpg)  |

The final model achieved a mean Average Precision (mAP) of **XX.X%** on the validation set.

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

## Acknowledgments

-   A big thank you to the creators of the **Sorghum Weed Dataset** for making their valuable data publicly available.
-   The Ultralytics team for the powerful and easy-to-use YOLOv8 library.
