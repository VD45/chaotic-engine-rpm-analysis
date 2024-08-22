# chaotic-engine-rpm-analysis
Analysis of chaotic behavior in the rotational speed of internal combustion engines using Python. This repository includes scripts for calculating Lyapunov exponents, correlation dimensions, and other chaos-related metrics
Overview
This repository contains the Python code and data used to analyze the chaotic behavior in the rotational speed (RPM) of internal combustion engines. The analysis is based on the research article "Chaotic Behavior in the Rotational Speed of Internal Combustion Engines" by Lomakin and co-authors.

Features
Lyapunov Exponents Calculation: Estimate the sensitivity to initial conditions, a key indicator of chaos.
Correlation Dimension: Measure the fractal dimension of the system, reflecting its complexity.
Phase Space Reconstruction: Visualize the system's dynamics in a reconstructed phase space.
Sample Entropy: Determine the complexity and predictability of the RPM data.
Power Spectral Density (PSD) Analysis: Analyze the distribution of power across frequencies in the RPM signal.
Average Mutual Information (AMI): Identify the optimal time delay for phase space reconstruction.
Requirements
To run the analysis, you need the following Python libraries:
numpy
pandas
matplotlib
scipy
nolds
joblib
tqdm
You can install the required packages using pip:
"
bash
pip install numpy pandas matplotlib scipy nolds joblib tqdm
Usage
Clone the repository:
"
"
bash
git clone https://github.com/yourusername/chaotic-engine-rpm-analysis.git
cd chaotic-engine-rpm-analysis
Run the main analysis script:
"
"
bash
python chaos_analysis_v1_3.py
View the output:
"
The results, including plots and calculated values (e.g., Lyapunov Exponents, Correlation Dimensions), will be saved in the repository directory.
Data Files
The following data files are required to run the analysis:

550.txt
1000.txt
2000.txt
3100.txt
high.txt
low.txt
Ensure these files are in the same directory as the script before running it.

Contribution
Feel free to fork this repository and contribute by submitting pull requests. If you encounter any issues, please open an issue on GitHub.

License
This project is licensed under the MIT License - see the LICENSE file for details.

You can copy and paste this text into the README file when you initialize your repository. Let me know if you'd like any adjustments!
