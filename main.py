from fastapi import FastAPI, HTTPException
from catboost import CatBoostClassifier
import pandas as pd


app = FastAPI()


class Model:
    def __init__(self, path, json_file):
        self.json_file = self.json_processing(json_file)
        self.model = CatBoostClassifier().load_model(path)

    def json_processing(self, json_file):
        self.json_file = pd.read_json(json_file, orient='index').transpose()
        data_types = {
            "Age": int,
            "Education": int,
            "MonthlyIncome": float,
            "num_companies_worked": int,
            "over_time": int,
            "TotalWorkingYears": int,
            "YearsAtCompany": int,
            "resume_on_job_search_site": int,
            "sent_messages": int,
            "company_years_ratio": float,
            "received_messages": int,
            "message_recipients": int,
            "bcc_message_count": int,
            "cc_message_count": int,
            "late_read_messages": int,
            "days_between_received_read": int,
            "replied_messages": int,
            "sent_message_characters": int,
            "off_hours_sent_messages": int,
            "received_sent_ratio": float,
            "received_sent_bytes_ratio": float,
            "unanswered_questions": int
        }
        self.json_file = self.json_file.astype(data_types)
        x = pd.get_dummies(self.json_file)
        self.json_file = pd.DataFrame(self.json_file,
                                      columns=['Age', 'Education', 'MonthlyIncome', 'num_companies_worked',
                                               'over_time', 'TotalWorkingYears', 'YearsAtCompany',
                                               'resume_on_job_search_site', 'sent_messages', 'company_years_ratio',
                                               'received_messages', 'message_recipients', 'bcc_message_count',
                                               'cc_message_count', 'late_read_messages', 'days_between_received_read',
                                               'replied_messages', 'sent_message_characters',
                                               'off_hours_sent_messages', 'received_sent_ratio',
                                               'received_sent_bytes_ratio', 'unanswered_questions',
                                               'matrial_status_Divorced', 'matrial_status_Married',
                                               'matrial_status_Single'])
        self.json_file.fillna(0, inplace=True)
        return self.json_file.merge(x, 'right')

    def predict(self):
        return self.model.predict_proba(pd.get_dummies(self.json_file))


@app.post("/predict/")
async def get_prediction(json_file):
    try:
        model = Model('attr_w', json_file)
        res = model.predict()

        return {'proba': res[0][1]}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

