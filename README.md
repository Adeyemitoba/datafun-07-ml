# datafun-07-ml 
# Project title: Implementing Simple Linear Regression (Supervised Machine Learning) and analyzing Time Series Data
# Project Overview
Using supervised machine learning, simple linear regression, to train a model using all available data and use the resulting model (a "best-fit" straight line) to make predictions.
# Project Requirements include:
# Developing a Linear Regression Model
1. Preparing and Training the Model: We initiate our analysis by constructing a straightforward linear regression model. The process encompasses organizing the dataset, dividing it into training and testing subsets, and employing the training subset to fit our model.
2. Making Predictions and Evaluation:After the model has been adequately trained, it will be deployed to generate predictions on the testing subset. We assess the model's efficacy by juxtaposing its predictive output against the actual observations within the testing subset.
3. Visualization of the Model's Performance:To further comprehend the model's alignment with the data, we plan to illustrate the linear regression model. This will be achieved by delineating the regression line that best fits, alongside the actual data points derived from the testing subset. Such visual representation serves as an insightful tool in evaluating the model's accuracy.
4. Insight Publication and Application:We will encapsulate the key findings and insights gleaned from this endeavor. The discourse will extend to the ramifications of our model's performance and its conceivable utility in addressing real-world challenges. This final phase aims to elucidate the practical implications and the broader impact of our analysis.

# Deliverables
datafun-07-ml: Project repository
README.md: Provides an overview of the project, as well as instructions for setting up and executing the envir
requirements.txt: Lists all packages required for the project.
Toba_ml.ipynb: Jupyter Notebook file

# How to Install and Run the Projec
Create a new GitHub repository named datafun-07-ml with a default README.md.
Clone the repository to your local machine.
git clone https://www.your-repository.com
Create a Project Virtual Environment in the .venv folder.
Activate the Project Virtual Environment.
py -m venv .venv
.\.venv\Scripts\Activate.ps1
Install dependencies into your .venv and freeze into your requirements.txt.
pip install pandas
pip install pyarrow
pip install scipy
pip install seaborn
pip install matplotlib
pip install scikit-learn
pip freeze > requirements.txt
Add a useful .gitignore to the root project folder with .vsode/ and .venv/ to prevent adding to repository
Git add and commit with a useful message (e.g. "initial commit") and push to GitHub.
git add .
git commit -m "initial commit"
git push origin main
Create examples subfolder in the root project repository folder: datafun-07-applied
Add downloaded associated files and data to the examples subfolder
# Start the Project
Open datafun-07-ml root project repository in VS code
Open a terminal in your root project repository folder and run git pull to make sure you have the latest changes from GitHub
Create new notebook in root project repository named: Toba_ml.ipynb
Add Markdown cell at the top of new notebook with Title, Author, and clickable link to project repository
Add Python cell with import statements
import matplotlib
from matplotlib import pyplot as plt
import pandas
import pyarrow
import scipy
from scipy import stats
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
Analysis Workflow
CC 7.5: Chart a Straight Line (Part 1)
Complete section per Project 7 requirements
CC 7.6: Predict Avg High Temp in NYC in January (Part 2)
Complete section per Project 7 requirements for Object-Oriented Programming
Build a model
Make predictions
Visualize the model
CC 7.7: Predict Avg High Temp in NYC in January (Part 3)
Complete section per Project 7 requirements for Supervised Machine Learning
Build a model
Make predictions
Visualize the model
CC 7.8: Insights (Part 4)
Complete section per Project 7 requirements
Publish insights
Optional Bonus (Part 5)
Complete section per Project 7 requirements
Loading the data
Training and testing the data
Visualizing the data
Choosing the best model 


# References
Guided projects in 10.16 and 15.4 of textbook: Intro to Python for Computer Science and Data Science: Learning to Program with AI, Big Data and the Cloud.
2. ChartGBT AI: This project likely involves creating graphical representations and applying machine learning techniques using the Gradient Boosting Tree (GBT) algorithm.
3. JUPYTER.md for Jupyter Notebook keyboard shortcuts and recommendations.
4. MARKDOWN.md for Markdown syntax and recommendations.
5. Data Visualization using Python, Matplotlib and Seaborn for visualization project ideas.
6. Linear Regression in Python using Jupyter Notebooks! to create forecasting projections.
