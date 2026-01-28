Project: MyDiet_AI - Intelligent Diet Planner
Submitted by: Devireddy Lohith Reddy

OVERVIEW:
-----------------------
MyDiet_AI is an end-to-end intelligent system that automates diet planning using:
1. OCR & Parsing (Milestone 1) for medical data extraction.
2. Random Forest Classification (Milestone 2) for diet prediction.
3. BioBERT NER (Milestone 3) for medical entity recognition.
4. Generative AI (OpenAI/GPT) for personalized meal planning.

FILES INCLUDED:
-----------------------
1. app.py: Main Streamlit web application.
2. train_model.py: ML pipeline used to train the model.
3. diet_model.pkl: The pre-trained Random Forest model.
4. training_dataset.csv: The dataset used for training.
5. patient_report_1.pdf & patient_report_2.pdf: Sample reports for testing.

HOW TO RUN:
-----------------------
1. Install dependencies:
   pip install -r requirements.txt

2. Run the application:
   streamlit run app.py

3. Test the App:
   - Drag and drop 'patient_report_1.pdf' into the sidebar.
   - The app will extract vitals, detect medical terms, and generate a diet plan.