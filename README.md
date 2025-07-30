# Reproducibility Platform Evaluation

This repository accompanies the bachelor's thesis:

**"Evaluation of Reproducibility Platforms – A Comparative Study on Usability, Performance and Application Areas"**  
**Author:** Rocco Hauptkorn  
**Institution:** Technical University of Munich  
**Submission Date:** August 1, 2025

---

## Thesis Abstract

Reproducibility is fundamental in scientific computing, ensuring that computational and experimental results can be replicated independently of time, system, and user. This thesis compares two reproducibility platforms – Binder and CodeOcean – based on usability, performance, limitations, and suitability for different application areas. The study focuses on three domains: Business Analytics, Machine Learning and Scientific Computing. The thesis will implement standardized "Hello World" test programs tailored to each domain to systematically assess these platforms. These programs will be deployed and executed on all platforms to analyze reproducibility challenges, including dependency management, execution consistency, and ease of use. The final results will provide insights into the strengths and weaknesses of different reproducibility solutions and recommend the right platform based on specific research and industry needs.


---

## Repository Structure

ba_output/ (Output artifacts from the business analytics notebook)\
ml_output/ (Output artifacts from the machine learning notebook)\
sc_output/ (Output artifacts from the scientific computing notebook)

business_analytics.ipynb (Notebook for business analytics domain)\
ml_pipeline.ipynb (Notebook for machine learning pipeline)\
scientific_pi_simulation.ipynb (Notebook for Monte Carlo $\pi$ estimation)\

requirements.txt (Unpinned dependencies for all notebooks)\
LICENSE (Open-source license (MIT))\
.gitignore (Files excluded from version control)\
README.md (This file)


---

## How to Run the Notebooks

### Option 1: Launch Online with Binder
Click below to open the notebooks in a temporary, cloud-based environment:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/RoccoHauptkornTUM/reproducibility_project_notebooks/HEAD)

---
### Option 2: Run on CodeOcean

To run the project on [CodeOcean](https://codeocean.com/dashboard):

1. **Create a free account** at [https://codeocean.com](https://codeocean.com/login)
2. **Click "Create New Compute Capsule"**
3. Choose:
   - **"Copy from public Git"**
   - Paste the repo URL:  
     ```
     https://github.com/RoccoHauptkornTUM/reproducibility_project_notebooks.git
     ```

4. After importing:
   - Select either the **JupyterLab** or **VSCode** interface
   - Go to the **Environment** tab:
     - Manually add dependencies based on `requirements.txt`
     - Or use the **postInstall** script to auto-install:
       ```bash
       pip install -r requirements.txt
       ```

5. Once the environment is ready:
   - Launch the capsule
---

### Option 3: Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RoccoHauptkornTUM/reproducibility_project_notebooks.git
   cd reproducibility_project_notebooks
2. **Create a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
4. **Launch Jupyter Notebook**
    ```bash
    jupyter notebook
---


## 🧪 Test Domains and Notebooks

| Domain              | Notebook                         | Description |
|---------------------|----------------------------------|-------------|
| **Business Analytics**  | `business_analytics.ipynb`         | Customer segmentation and sales pattern detection using synthetic retail data |
| **Machine Learning**    | `ml_pipeline.ipynb`                | A full ML pipeline predicting student placement using classification models |
| **Scientific Computing**| `scientific_pi_simulation.ipynb`   | Monte Carlo simulation to estimate π and assess convergence behavior |

---

## 📊 Platform Evaluation Summary

| Platform    | Strengths | Best Use Case |
|-------------|-----------|----------------|
| **Binder**      | Simple, browser-based, no install | Teaching, quick sharing |
| **CodeOcean**   | Versioning, DOI support, auditability | Reproducible research & publishing |
| **Local (VSCode)** | Full control, flexible | Development, debugging, offline work |

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).

---

## 📬 Contact

For questions, feedback, or collaboration:

**Rocco Hauptkorn**  
`rocco.hauptkorn@tum.de`
