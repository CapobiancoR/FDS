## Step 1: Data Preparation
- analyze data and visualize 
- Image Preprocessing:
- Resize images to a standard dimension 
- Normalize pixel values 
- Augment data ( if needed )
## Step 2: Model Selection
- pre-trained CNN model or Custom CNN arch ( if needed) 
## Step 3:Model Training
- Optimize hyperparameters and apply suitable optimizer and loss function
- hyperparameter search 
## Step 4:  Model Evaluation
- Our Metrics: Accuracy, Recall, Confusion Matrix 
- dropout layers or early stopping  can apply ( if  overfits ) and cross-validation can be used 

## Notes from slides : 
analyze and add comment for each step 
add standart graphs as much as possible :  loss curves, accuracy chart, simple architecture graphics...


# Important

Git desktop is giving errors with the the 3Gb dataset (also having a .gitignore), so you must have the dataset just outside the directory.

Folder structure:

FP_FDS/ (or however you prefer)
	/FDS (github repo)
	/Datset
		/New Plant Diseases Dataset(Augmented)
			/train
				/[jpeg fiolders]
			/valid
				/[jpeg fiolders]
		/test
			/*.JPEG
	