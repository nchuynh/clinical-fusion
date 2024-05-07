# Team 70
* **Team members:** Nhan Huynh (nchuynh2@illinois.edu)
* **Paper:** [10] Combining structured and unstructured data for predictive models: a deep learning approach
* **GitHub repository:** https://github.com/nchuynh/clinical-fusion
* **Video presentation:** https://www.youtube.com/watch?v=tBxqkE2sRhs

## Steps to run the code
1. Setup the environment
2. Obtain the MIMIC-III dataset
3. Preprocess the MIMIC-III dataset
4. Train the models
5. Test the models

## 1. Setup the environment
* Requirements:
   * Python = 3.6.10
   * Gensim = 3.8.0
   * Matplotlib = 3.1.3
   * NLTK = 3.4.5
   * Numpy = 1.14.2
   * Pandas = 0.25.3
   * Scikit-learn = 0.20.1
   * Tqdm = 4.42.1
   * PyTorch = 1.4.0
* Create the Conda environment
   * Use requirements.yml: `conda env create -f environment.yml`
   * Activate the clinical-fusion environment: `conda activate clinical-fusion`

## 2. Obtain the MIMIC-III dataset
1. Request access to MIMIC-III dataset by:
  * Creating a PhysioNet account on: `https://physionet.org/login/`
  * Completing the training for CITI Data or Specimens Only research: `https://physionet.org/about/citi-course/`
  * Submitting the training completion certificate: `https://physionet.org/settings/training/`
  * Signing the data use agreement: `https://physionet.org/sign-dua/mimiciii/1.4/`
  * Becoming a credentialed user: `https://physionet.org/settings/credentialing/`
2. Download the files for the MIMIC-III dataset
3. Place the files in: `./data/mimic`

## 3. Preprocess the MIMIC-III dataset
1. Create a MIMIC-III database using PostgreSQL by:
  * Downloading and installing PostgreSQL: `http://www.postgresql.org/download/`
  * Cloning the mimic-code repo:
    * `git clone https://github.com/MIT-LCP/mimic-code.git`
  * Making the database from the MIMIC-III files: 
    * `make create-user mimic-gz datadir="/path/to/mimic-iii-clinical-database-1.4/"` 
2. Run SQL queries and export views by:
  * Connecting to the MIMIC-III database:
    * `psql -d mimic`
  * Setting the default scheme to mimiciii:
    * `SET search_path TO mimiciii;`
  * Executing the .sql files in `./query` and exporting the views to .csv files:
    ```
    \i adm_details.sql
    \copy (SELECT * FROM adm_details) TO 'adm_details.csv' WITH CSV HEADER;

    \i pivoted-lab.sql
    \copy (SELECT * FROM pivoted_lab) TO 'pivoted-lab.csv' WITH CSV HEADER;

    \i pivoted-vital.sql
    \copy (SELECT * FROM pivoted_vital) TO 'pivoted-vital.csv' WITH CSV HEADER;
    ```
  * Transferring the .csv files to `./data/mimic`
3. Run the preproessing scripts with:
   ```
   python 00_define_cohort.py # define patient cohort and collect labels
   python 01_get_signals.py # extract temporal signals (vital signs and laboratory tests)
   python 02_extract_notes.py --firstday # extract first day clinical notes
   python 03_merge_ids.py # merge admission IDs
   python 04_statistics.py # run statistics
   python 05_preprocess.py # run preprocessing
   python 06_doc2vec.py --phase train # train doc2vec model
   python 06_doc2vec.py --phase infer # infer doc2vec vectors
   ```

## 4. Train the models
* Use the helper functions provided in `DL4H_Team_40.ipynb`
   * run_train(): Run training and validation. Best model is saved to '<model>_<task>_<use_unstructure>.ckpt'.
* Parameters:
   * model: 'cnn', 'lstm'
   * task: 'mortality', 'readmit', 'llos'
   * use_unstructure: 0, 1
      * 0: Only use structured data (temporal signals and demographics)
      * 1: Use unstructured data(clinical notes) and structured data
   * epochs: Number of epochs for training
* Output: Each epoch, training loss, training AUC, validation loss, and validation AUC. Best model (highest validation AUC) is saved to '<model>_<task>_<use_unstructure>.ckpt'.

## 5. Test the models
* Use the helper functions provided in `DL4H_Team_40.ipynb`
   * run_test(): Evaluates on test split. Uploads best model for selected parameters.
* Parameters:
   * model: 'cnn', 'lstm'
   * task: 'mortality', 'readmit', 'llos'
   * use_unstructure: 0, 1
      * 0: Only use structured data (temporal signals and demographics)
      * 1: Use unstructured data(clinical notes) and structured data
* Output: Testing AUC
