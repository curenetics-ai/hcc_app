import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import boto3
from dotenv import load_dotenv
# Load dotenv first
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)


BUCKET_NAME = "hcc-app-model-weights-2026"
NER_MODEL_KEY = "HCC_APP_NER_Model_files/model.safetensors"
TOKENIZER_FILES = [
    "HCC_APP_NER_Model_files/config.json",
    "HCC_APP_NER_Model_files/tokenizer.json",
    "HCC_APP_NER_Model_files/tokenizer_config.json",
    "HCC_APP_NER_Model_files/vocab.txt",
    "HCC_APP_NER_Model_files/special_tokens_map.json"
]

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION")
)

def download_from_s3(bucket: str, key: str, local_path: str):
    s3.download_file(bucket, key, local_path)
    return local_path

def load_tokenizer_from_s3(bucket: str, tokenizer_files: list, local_dir="./tmp_ner_model"):
    os.makedirs(local_dir, exist_ok=True)
    for key in tokenizer_files:
        local_file = os.path.join(local_dir, os.path.basename(key))
        download_from_s3(bucket, key, local_file)
    return AutoTokenizer.from_pretrained(local_dir, truncation=True, padding="max_length", model_max_length=512)


def load_model_from_s3(bucket: str, model_key: str, local_dir="./tmp_ner_model"):
    os.makedirs(local_dir, exist_ok=True)
    local_file = os.path.join(local_dir, os.path.basename(model_key))
    download_from_s3(bucket, model_key, local_file)
    return AutoModelForTokenClassification.from_pretrained(local_dir)

liver_disease_flag = [
    "Liver Cirrhosis",
    "Cirrhotic",
    "Chronic liver disease",
    "Fatty Liver",    # optional
    "Hepatic"         # optional, may be too generic
]

# 2️⃣ Portal hypertension / complications
portal_hypertension_flag = [
    "Portal Hypertension",
    "Esophageal Varices",
    "Gastroesophageal varices",
    "Portal Vein Thrombosis",
    "Hepatic ascites"
]

# 3️⃣ Liver tumor / neoplasm
liver_tumor_flag = [
    "Liver carcinoma",
    "Lesion of liver",
    "Liver mass",
    "Neoplasms",
    "Malignant Neoplasms",
    "Neoplastic",
    "Liver and Intrahepatic Biliary Tract Carcinoma",
    "Primary Malignant Liver Neoplasm",
    "Hepatocellular"
]

# 4️⃣ Biliary / duct issues
biliary_flag = [
    "Congenital Biliary Dilatation",
    "biliary dilatation",
    "Intrahepatic bile duct dilatation",
    "Obstruction of biliary tree",
    "Bile Ducts, Extrahepatic"
]

# 5️⃣ Symptoms / general clinical signs
symptoms_flag = [
    "Ascites",       # also in liver disease
    "Splenomegaly",
    "Abdominal Pain",
    "Weight Loss",
    "Fatigue"
]


class NERService:
    def __init__(self):
        self.tokenizer = load_tokenizer_from_s3(BUCKET_NAME, TOKENIZER_FILES)
        self.model = load_model_from_s3(BUCKET_NAME, NER_MODEL_KEY)
        self.ner = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple"
        )

    def split_chunks(self, clinical_notes: str, chunk_size=512, overlap=50):
        tokens = self.tokenizer.encode(clinical_notes, add_special_tokens=False)

        chunks = []
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk = tokens[start:end]
            chunks.append(chunk)
            start += (chunk_size - overlap)

        return [self.tokenizer.decode(chunk) for chunk in chunks]

    def extract_symptoms(self, clinical_notes: str):
        chunks = self.split_chunks(clinical_notes)

        all_results = []
        batch_size = 8

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            results = self.ner(batch)
            all_results.extend(results)

        extracted = set()
        for chunk_result in all_results:
            for ent in chunk_result:
                if ent["entity_group"] == "Disease":
                    word = ent["word"].strip()
                    extracted.add(word)

        return list(extracted)

    def generate_flags(self, clinical_notes: str):
        symptoms = self.extract_symptoms(clinical_notes)

        flags = [
            liver_tumor_flag,
            liver_disease_flag,
            portal_hypertension_flag,
            biliary_flag,
            symptoms_flag
        ]

        ner_list = [0] * len(flags)
        symptoms_set = {s.lower().strip() for s in symptoms}

        for i, flag_keywords in enumerate(flags):
            flag_set = {f.lower().strip() for f in flag_keywords}
            if symptoms_set & flag_set:
                ner_list[i] = 1

        return ner_list