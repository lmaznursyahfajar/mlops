# Imports

import os
import random
import re
import warnings

import evaluate
import numpy as np
import onnx
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from nltk.tokenize import word_tokenize
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          RobertaTokenizer)

warnings.filterwarnings("ignore")

# Cleaning functions

stopword = []
with open("stopwords.txt", "r", encoding="utf-8") as file:
    for word in file:
        stopword.append(word.strip())

slangwords = {}
with open("slangwords.txt", "r", encoding="utf-8") as file:
    for line in file:
        words = line.split(",")
        old = words[0].strip()
        new = words[1].strip()
        slangwords[old] = new


def convertToSlangword(review):
    review = review.split()
    content = []
    for kata in review:
        if kata in slangwords:
            new_kata = slangwords[kata]
        else:
            new_kata = kata
        content.append(new_kata.lower())
    return " ".join(content)


def filtering(review, remove_numbers=True):
    # Menghapus URL
    review = re.sub(r"https?://\S+", " ", review)
    review = re.sub(r"\S*\.(com|org|co)/\w*\b", " ", review)

    review = re.sub(r"@[\w\.]+\b", " ", review)
    review = re.sub(r"@\w+", " ", review)

    review = re.sub(r"[.,:;]", " ", review)

    # Menghapus kata setelah tanda pagar (#) hanya jika jumlah hashtag tepat 3
    hashtags = re.findall(r"#([^\s]+)", review)
    for hashtag in hashtags:
        review = re.sub(r"#" + re.escape(hashtag) + r"\b", " ", review)

    review = re.sub(r"\d", " ", review) if remove_numbers else review
    review = re.sub(r"[.,:;+!\-_<^/=?\"'\(\)\*]", " ", review)
    review = re.sub(r"[^\x00-\x7f]", r" ", review)
    review = re.sub(r"(\\u[0-9A-Fa-f]+)", r" ", review)
    review = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", review)
    review = re.sub(r"\\u\w\w\w\w", " ", review)
    review = re.sub(r"@\w+\b", " ", review)
    review = re.sub(r"\s+", " ", review)

    # remove some words
    rmv = ["a", "href", "quot"]
    review = word_tokenize(review)
    review = [str(token).strip() for token in review if token not in rmv]
    review = " ".join(review)

    return review.strip()


def replaceThreeOrMore(review):
    pattern = re.compile(r"(\w)\1{2,}")
    return pattern.sub(r"\1", review)


def remove_stopwords(text):
    text = text.split()
    text = [token for token in text if token not in stopword]
    return " ".join(text)


def clean(review, slang=True, sw=True, num=True):
    review = review.lower()  # casefolding
    review = filtering(review, remove_numbers=num)
    review = replaceThreeOrMore(review)
    review = convertToSlangword(review) if slang else review
    review = remove_stopwords(review) if sw else review
    return review


# Common functions


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def count_param(module, trainable=False):
    if trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def metrics_to_string(metric_dict):
    string_list = []
    for key, value in metric_dict.items():
        string_list.append("{}:{:.2f}".format(key, value))
    return " ".join(string_list)


set_seed(2024)

# Util classes


class ABSADataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {"positive": 0, "neutral": 1, "negative": 2}
    INDEX2LABEL = {0: "positive", 1: "neutral", 2: "negative"}
    NUM_LABELS = 3

    def load_dataset(self, path):
        df = pd.read_csv(path, sep="\t", header=None)
        df.columns = ["aspect", "text", "sentiment"]
        df["aspect"] = df["aspect"].astype(str)
        df["text"] = df["text"].astype(str)

        def safe_label_convert(lab):
            if pd.isna(lab):
                return -1
            label_str = str(lab).strip().lower()
            return self.LABEL2INDEX.get(label_str, -1)

        df["sentiment"] = df["sentiment"].apply(safe_label_convert)
        df = df[df["sentiment"] != -1]  # Hapus baris dengan label tidak valid
        df = df.reset_index(drop=True)
        return df

    def __init__(
        self, dataset_path, tokenizer, no_special_token=False, *args, **kwargs
    ):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token

    def __getitem__(self, index):
        data = self.data.loc[index, :]
        aspect, text, sentiment = data["aspect"], data["text"], data["sentiment"]
        subwords = self.tokenizer.encode(
            aspect, text, add_special_tokens=not self.no_special_token
        )
        return (
            np.array(subwords),
            np.array(sentiment),
            f"({data['aspect']}) {data['text']}",
        )

    def __len__(self):
        return len(self.data)


class ABSADataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(ABSADataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len

    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)

        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        sentiment_batch = np.zeros((batch_size, 1), dtype=np.int64)

        seq_list = []
        for i, (subwords, sentiment, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_batch[i, : len(subwords)] = subwords
            mask_batch[i, : len(subwords)] = 1
            sentiment_batch[i, 0] = sentiment

            seq_list.append(raw_seq)

        return subword_batch, mask_batch, sentiment_batch, seq_list


def prepare_batch(batch_data, device="cpu"):
    batch_data = batch_data[:-1]
    if len(batch_data) == 3:
        (subword_batch, mask_batch, label_batch) = batch_data
        token_type_batch = None
    elif len(batch_data) == 4:
        (subword_batch, mask_batch, token_type_batch, label_batch) = batch_data

    subword_batch = torch.LongTensor(subword_batch).to(device)
    mask_batch = torch.FloatTensor(mask_batch).to(device)
    token_type_batch = (
        torch.LongTensor(token_type_batch).to(device)
        if token_type_batch is not None
        else None
    )
    label_batch = torch.LongTensor(label_batch).to(device)

    return {
        "input_ids": subword_batch,
        "attention_mask": mask_batch,
        "token_type_ids": token_type_batch,
        "labels": label_batch,
    }


# Main trainer class


class ABSAModelTrainer:

    # Print function for logging
    def print_and_log(self, msg):
        self.log_file.write(f"{msg}\n")
        self.log_file.flush()
        os.fsync(self.log_file.fileno())
        print(msg)

    def __init__(
        self,
        csv_input_path,
        model_name,
        pretrained_model="indobenchmark/indobert-base-p1",
        model_path_prefix="models",
        csv_output_path_prefix="dataset-fix",
        csv_input_sep=";",
        csv_output_sep=";",
        csv_train_frac=0.8,
        num_epochs=2,
        learning_rate=6e-6,
        postmade_file=None,
        is_onm=False,
    ):
        # Load parameters
        self.csv_input_path = csv_input_path
        self.csv_output_path = f"{csv_output_path_prefix}/{model_name}"
        self.model_name = model_name
        self.model_save_path = f"{model_path_prefix}/{model_name}"
        self.num_epochs = num_epochs
        self.pretrained_model = pretrained_model
        self.csv_input_sep = csv_input_sep
        self.csv_output_sep = csv_output_sep
        self.csv_train_frac = csv_train_frac
        self.learning_rate = learning_rate
        self.postmade_file = postmade_file
        self.is_onm = is_onm
        # tambahan
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Testing files
        self.test_result = None
        log_file_path = f"{self.model_save_path}/log_file.txt"
        os.makedirs(self.model_save_path, exist_ok=True)
        self.log_file = open(log_file_path, "w")
        print(f"Log saved in file: {log_file_path}")

        media_type = "online media / ONM" if self.is_onm else "social media"
        self.print_and_log(f"Media type: {media_type}")
        self.print_and_log(f"- Model training parameters -")
        self.print_and_log(f"Model name: {self.model_name}")
        self.print_and_log(f"Model save location: {self.model_save_path}")
        self.print_and_log(f"Pretrained model: {self.pretrained_model}")
        self.print_and_log(f"Num of training epochs: {self.num_epochs}")
        self.print_and_log(f"Learning rate: {self.learning_rate}")
        self.print_and_log(f"- Dataset creation parameters -")
        self.print_and_log(f"Input CSV file path: {self.csv_input_path}")
        self.print_and_log(f"Training dataset save location: {self.csv_output_path}")
        self.print_and_log(f"Input CSV column separators: {self.csv_input_sep}")
        self.print_and_log(f"Output CSV column separators: {self.csv_output_sep}")
        self.print_and_log(
            f"Dataset train fraction (for splitting): {self.csv_train_frac}"
        )

    # Dataset functions

    def load_raw_dataset(self):
        # Load input csv (raw) as dataframe
        print("Loading (raw data) CSV file into dataframe...")
        try:
            self.df = pd.read_csv(self.csv_input_path, sep=self.csv_input_sep)
        except:
            self.df = pd.read_csv(
                self.csv_input_path,
                sep=self.csv_input_sep,
                engine="python",
                on_bad_lines="skip",
            )
        print("Finished loading raw data!")

    def create_dataset_socmed(self):
        self.load_raw_dataset()
        df = self.df

        # Show raw data details
        print(f"Original (talk only) dataset size: {len(df)} rows")
        for sentiment in ["positive", "neutral", "negative"]:
            print(f"- {sentiment}: {len(df[df['final_sentiment'] == sentiment])}")

        # Clean dataset
        print("Preparing to clean and create training datasets...")
        print("Removing unnecessary columns...")
        df = df.rename(columns={"content_x": "content", "link_x": "link"})
        df = df[
            [
                "object_name",
                "content",
                "final_sentiment",
                "reply_to_original_id",
                "link",
            ]
        ]
        print("Renaming columns...")
        # df = df.rename(columns={'content':'raw_text', 'final_sentiment':'sentiment', 'object_name':'aspect'})
        for col in df.columns:
            df[col] = df[col].astype(str).apply(lambda x: x[1:] if x[0] == "'" else x)
        print("Applying text cleaning function to raw text...")
        df["clean_text"] = df["content"].apply(
            lambda x: clean(x, sw=False, slang=False, num=False)
        )
        df = df[["object_name", "content", "clean_text", "final_sentiment", "link"]]
        print("Dropping duplicate and empty text rows...")
        df = df.drop_duplicates(subset=["object_name", "clean_text", "final_sentiment"])
        df = df.dropna()
        df = df.drop(df[df["clean_text"] == ""].index)
        for col in df.columns:
            df[col] = df[col].astype(str)
        df["clean_text"] = df["clean_text"].astype(str)
        print("Finished cleaning dataset!")

        # Show cleaned dataset details
        print(f"Cleaned dataset size: {len(df)} rows")
        for sentiment in ["positive", "neutral", "negative"]:
            print(f"- {sentiment}: {len(df[df['final_sentiment'] == sentiment])}")

        # Split dataset
        print("Splitting cleaned dataset into train/test/validation...")
        self.df_train = df.sample(frac=0.8)
        self.df_test = df.drop(self.df_train.index).sample(frac=0.5)
        self.df_valid = df.drop(self.df_train.index).drop(self.df_test.index)
        # Match test dataset with original post
        print("Matching test dataset with original postmade (if exists)... # TBI #")
        # TBI

        print("Finished splitting datasets!")

        print(f"Train split dataset size: {len(self.df_train)}")
        print(f"Test split dataset size: {len(self.df_test)}")
        print(f"Valid split dataset size: {len(self.df_valid)}")

    def create_dataset_onm(self):
        self.load_raw_dataset()
        df = self.df

        # Show raw data details
        print(f"Original (talk only) dataset size: {len(df)} rows")
        for sentiment in ["positive", "neutral", "negative"]:
            print(f"- {sentiment}: {len(df[df['final_sentiment'] == sentiment])}")

        # Clean dataset
        print("Preparing to clean and create training datasets...")
        print("Removing unnecessary columns...")
        df = df[["clipping_name", "title", "body", "final_sentiment", "link"]]
        df["content"] = df["title"] + "\n" + df["body"]
        print("Renaming columns...")
        # df = df.rename(columns={'content':'raw_text', 'final_sentiment':'sentiment', 'object_name':'aspect'})
        for col in df.columns:
            df[col] = df[col].astype(str).apply(lambda x: x[1:] if x[0] == "'" else x)
        print("Applying text cleaning function to raw text...")
        df["clean_text"] = df["content"].apply(lambda x: str(x).strip().lower())
        df = df[
            [
                "clipping_name",
                "title",
                "body",
                "content",
                "clean_text",
                "final_sentiment",
                "link",
            ]
        ]
        print("Dropping duplicate and empty text rows...")
        df = df.drop_duplicates(
            subset=["clipping_name", "clean_text", "final_sentiment"]
        )
        df = df.dropna()
        df = df.drop(df[df["clean_text"] == ""].index)
        for col in df.columns:
            df[col] = df[col].astype(str)
        df["clean_text"] = df["clean_text"].astype(str)
        print("Finished cleaning dataset!")

        # Show cleaned dataset details
        print(f"Cleaned dataset size: {len(df)} rows")
        for sentiment in ["positive", "neutral", "negative"]:
            print(f"- {sentiment}: {len(df[df['final_sentiment'] == sentiment])}")

        # Split dataset
        print("Splitting cleaned dataset into train/test/validation...")
        self.df_train = df.sample(frac=0.8)
        self.df_test = df.drop(self.df_train.index).sample(frac=0.5)
        self.df_valid = df.drop(self.df_train.index).drop(self.df_test.index)
        # Match test dataset with original post
        print("Matching test dataset with original postmade (if exists)... # TBI #")
        # TBI

        print("Finished splitting datasets!")

        print(f"Train split dataset size: {len(self.df_train)}")
        print(f"Test split dataset size: {len(self.df_test)}")
        print(f"Valid split dataset size: {len(self.df_valid)}")

    def create_dataset(self):
        if self.is_onm:
            self.create_dataset_onm()
        else:
            self.create_dataset_socmed()

    def save_dataset(self, force_overwrite=False):
        print("Preparing to save training datasets...")
        # Create folder if not exist
        print("Creating folder to save datasets...")
        if not force_overwrite:
            if os.path.isdir(f"{self.csv_output_path}"):
                raise Exception(
                    f"{self.csv_output_path} already exists, set force_overwrite=True to create new dataset and replace the existing one."
                )

        os.makedirs(f"{self.csv_output_path}", exist_ok=True)

        # Save unsplit CSV
        print("Saving cleaned dataset (no split)...")
        self.df_train.to_csv(
            f"{self.csv_output_path}/{self.model_name}.csv",
            index=False,
            sep=self.csv_output_sep,
        )

        # Save as CSV
        print("Saving split datasets as CSVs...")
        self.df_train.to_csv(
            f"{self.csv_output_path}/train.csv", index=False, sep=self.csv_output_sep
        )
        self.df_test.to_csv(
            f"{self.csv_output_path}/test.csv", index=False, sep=self.csv_output_sep
        )
        self.df_valid.to_csv(
            f"{self.csv_output_path}/valid.csv", index=False, sep=self.csv_output_sep
        )

        # Save as TSV (for training dataset)
        print("Saving split datasets as TSVs for training...")
        if self.is_onm:
            aspect_col = "clipping_name"
        else:
            aspect_col = "object_name"
        self.df_train[[aspect_col, "clean_text", "final_sentiment"]].to_csv(
            f"{self.csv_output_path}/train.tsv", index=False, header=False, sep="\t"
        )
        self.df_test[[aspect_col, "clean_text", "final_sentiment"]].to_csv(
            f"{self.csv_output_path}/test.tsv", index=False, header=False, sep="\t"
        )
        self.df_valid[[aspect_col, "clean_text", "final_sentiment"]].to_csv(
            f"{self.csv_output_path}/valid.tsv", index=False, header=False, sep="\t"
        )

        print("Finished saving datasets!")

    def load_dataset(self):
        # Load dataset
        train_dataset_path = f"{self.csv_output_path}/train.tsv"
        valid_dataset_path = f"{self.csv_output_path}/valid.tsv"

        print("Loading datasets for training...")
        train_dataset = ABSADataset(train_dataset_path, self.tokenizer, lowercase=True)
        valid_dataset = ABSADataset(valid_dataset_path, self.tokenizer, lowercase=True)

        print(f"Loading train dataset from {train_dataset_path}...")
        self.train_loader = ABSADataLoader(
            dataset=train_dataset,
            max_seq_len=512,
            batch_size=16,
            num_workers=4,
            shuffle=True,
        )
        print(f"Loading validation dataset from {valid_dataset_path}")
        self.valid_loader = ABSADataLoader(
            dataset=valid_dataset,
            max_seq_len=512,
            batch_size=16,
            num_workers=4,
            shuffle=False,
        )
        print("Finished loading!")

    def prepare_dataset(self, force_overwrite=False):
        self.create_dataset()
        self.save_dataset(force_overwrite)

    # Training functions
    def load_model(self):
        # Load model and tokenizer
        print("Loading model and tokenizer...")
        w2i, i2w = ABSADataset.LABEL2INDEX, ABSADataset.INDEX2LABEL
        num_labels = len(w2i)
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        print("Finished loading tokenizer!")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model, num_labels=num_labels, id2label=i2w, label2id=w2i
        )
        # Check for GPU
        if torch.cuda.is_available():
            print(f"Device yang digunakan: {self.device}")
            self.model = self.model.to(self.device)
        print("Finished loading model!")

    def train_model(self):
        # Load training functions (accelerator, eval metric, optimizer)
        print(f"Preparing training functions...")
        accelerator = Accelerator()
        print("Loading evaluation metric...")
        metric = evaluate.load("accuracy")
        print(f"Initializing optimizer...")
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=3, threshold=1e-4, min_lr=1e-6
        )
        model, optimizer, train_loader, eval_loader = accelerator.prepare(
            self.model, optimizer, self.train_loader, self.valid_loader
        )

        print("Starting training!")
        # Main training loop
        try:
            best_acc = 0.0
            best_epoch = 0
            epochs_without_improvement = 0  # Track epochs without improvement

            for epoch in range(self.num_epochs):
                # Training phase
                model.train()
                train_pbar = tqdm(
                    train_loader, position=0, leave=True, total=len(train_loader)
                )
                total_train_loss = 0

                for i, batch in enumerate(train_pbar):
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    inputs = prepare_batch(batch, device=self.device)
                    outputs = self.model(**inputs)
                    loss = outputs.loss
                    accelerator.backward(loss)

                    optimizer.step()
                    optimizer.zero_grad()

                    total_train_loss += loss.item()
                    train_pbar.set_description(
                        f"(Epoch {epoch+1}) TRAIN LOSS:{total_train_loss/(i+1):.4f} LR:{get_lr(optimizer):.8f}"
                    )

                # Validation phase
                model.eval()
                eval_pbar = tqdm(
                    eval_loader, position=0, leave=True, total=len(eval_loader)
                )
                total_eval_loss = 0

                for i, batch in enumerate(eval_pbar):
                    with torch.no_grad():
                        inputs = prepare_batch(batch, device=self.device)
                        outputs = model(**inputs)

                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    metric.add_batch(
                        predictions=predictions, references=inputs["labels"]
                    )
                    total_eval_loss += outputs.loss.item()
                    eval_pbar.set_description(f"VALID LOSS:{total_eval_loss/(i+1):.4f}")

                results = metric.compute()
                print(f"(Epoch {epoch+1}) VALID ACCURACY:{results['accuracy']:.8f}")

                # Step the scheduler based on the validation loss
                scheduler.step(total_eval_loss)

                # Check for improvement
                if results["accuracy"] > best_acc:
                    best_acc = results["accuracy"]
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0

                    # Save and upload model
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            self.model_save_path, save_function=accelerator.save
                        )
                        self.tokenizer.save_pretrained(self.model_save_path)
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= 5:
                        break

        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Batch: {batch}")
            raise

        print("Finished training!")
        self.print_and_log(f"Best accuracy: {best_acc} (Epoch {best_epoch})")
        return best_acc  # <- Tambahkan ini

    # Testing functions
    def load_saved_model(self):
        print("Loading saved model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_save_path)
        print("Finished loading saved tokenizer!")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_save_path
        )
        # Check for GPU
        if torch.cuda.is_available():
            print(f"Device yang digunakan: {self.device}")
            self.model = self.model.to(self.device)
        print("Finished loading saved model!")

    def infer(self, aspect, text, debug=False):
        if len(text) > 0:
            encoding = self.tokenizer(
                aspect,
                text,
                truncation=True,
                max_length=64,
                return_tensors="pt",
                add_special_tokens=True,
            ).to(self.device)
            if debug:
                print(
                    self.tokenizer.convert_ids_to_tokens(
                        encoding.input_ids.squeeze().tolist()
                    )
                )
            output = self.model(**encoding)
            logits = output[0]
        else:
            logits = torch.Tensor([[0, 15, 0]]).to(self.device)
        label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()

        # print(f'Text: {text} | Label : {i2w[label]} ({F.softmax(logits, dim=-1).squeeze()[label] * 100:.3f}%)')
        return {
            "prediction": ABSADataset.INDEX2LABEL[label],
            "probability": float(F.softmax(logits, dim=-1).squeeze()[label]),
        }

    def run_inference(
        self,
        col_pred="prediction",
        col_prob="probability",
        aspect_col="object_name",
        text_col="content",
        with_clean=True,
        show_clean=False,
        col_clean="clean_text",
        **kwargs,
    ):
        df = pd.read_csv(
            f"{self.csv_output_path}/test.csv",
            sep=self.csv_output_sep,
            lineterminator="\n",
        )
        for col in df.columns:
            df[col] = df[col].astype(str)
        prediction = []
        probability = []
        if show_clean:
            clean_text = []
        for _, row in tqdm(df.iterrows()):
            text = clean(row[text_col], **kwargs) if with_clean else row[text_col]
            aspect = row[aspect_col]
            try:
                result = self.infer(aspect, text)
            except:
                print(f"text: {text}")
                print(f"aspect: {aspect}")
                raise
            prediction.append(result["prediction"])
            probability.append(result["probability"])
            if show_clean:
                clean_text.append(text)
        df[col_pred] = prediction
        df[col_prob] = probability
        if show_clean:
            if col_clean in df.columns:
                df[col_clean] = clean_text
            else:
                df.insert(2, col_clean, clean_text)
        print(f"Saving testing results to CSV...")
        df.to_csv(
            f"{self.csv_output_path}/{self.model_name}-test-result.csv",
            index=False,
            sep=self.csv_output_sep,
        )
        print(f"Testing results saved!")
        self.test_result = df

    def save_testing_result(self):
        print(f"Saving testing results to CSV...")
        self.test_result.to_csv(
            f"{self.csv_output_path}/{self.model_name}-test-result.csv",
            index=False,
            sep=self.csv_output_sep,
        )
        print(f"Testing results saved!")

    # Stats functions

    def distribution(self, col_prob="probability"):
        df = self.test_result
        distrib = []
        # print(f'{col_prob} distribution (% of total)')
        for rg in range(0, 10):
            inf = rg / 10
            sup = (rg + 1) / 10
            n = len(df[(df[col_prob] >= inf) & (df[col_prob] < sup)])
            rate = n / len(df) * 100 if len(df) > 0 else 0
            distrib.append(f"[{inf} , {sup}[ = {n} ({rate:.2f}%)")
        return distrib

    def distribution_correct(
        self, col_label="final_sentiment", col_prob="probability", col_pred="prediction"
    ):
        df = self.test_result
        distrib_correct = []
        # print(f'{col_prob} (% correct)')
        for rg in range(0, 10):
            inf = rg / 10
            sup = (rg + 1) / 10
            n = len(df[(df[col_prob] >= inf) & (df[col_prob] < sup)])
            correct = len(
                df[
                    (df[col_prob] >= inf)
                    & (df[col_prob] < sup)
                    & (df[col_label] == df[col_pred])
                ]
            )
            rate = correct / n * 100 if n > 0 else 0
            distrib_correct.append(f"[{inf} , {sup}[ = {correct}/{n} ({rate:.2f}%)")
        return distrib_correct

    def cumulative_correct(
        self, col_label="final_sentiment", col_prob="probability", col_pred="prediction"
    ):
        df = self.test_result
        correct_list = []
        for rg in range(9, -1, -1):
            inf = rg / 10
            correct = len(df[(df[col_prob] >= inf) & (df[col_label] == df[col_pred])])
            valid = len(df[df[col_prob] >= inf])
            rate = correct / valid * 100 if valid > 0 else 0
            if rg == 0:
                suf = f"> {inf} (all)"
            else:
                suf = f"> {inf}"
            correct_list.append(f"{col_prob} {suf} : {correct} / {valid} ({rate:.2f}%)")
        return correct_list

    def stats(
        self, col_label="final_sentiment", col_prob="probability", col_pred="prediction"
    ):
        # Distribution by range
        distrib = self.distribution(col_prob)
        print("Distribution by range:")
        for item in distrib:
            print(f"{item}")
        print("\n")
        # Distribution correct by range
        distrib_correct = self.distribution_correct(col_label, col_prob, col_pred)
        print("Distribution of correct by range:")
        for item in distrib_correct:
            print(f"{item}")
        print("\n")
        # Cumulative correct distribution
        cumul_correct = self.cumulative_correct(col_label, col_prob, col_pred)
        print("Cumulative correct distribution:")
        for item in cumul_correct:
            print(f"{item}")

    def export_to_onnx(self, export_dir="models_onnx"):
        print("Mengekspor model ke format ONNX...")

        self.model.eval()
        dummy_text = "project_name [SEP] aspect [SEP] text ini adalah contoh input"
        dummy_input = self.tokenizer(
            dummy_text, return_tensors="pt", truncation=True, max_length=512
        )

        os.makedirs(export_dir, exist_ok=True)
        output_path = os.path.join(export_dir, f"{self.model_name}.onnx")

        torch.onnx.export(
            self.model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            opset_version=14,
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "logits": {0: "batch_size", 1: "sequence"},
            },
        )

        # Validasi hasil ekspor
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"Model berhasil dikonversi ke ONNX → {output_path}")

    def run(self):
        self.prepare_dataset()
        self.load_dataset()
        print(f"[INFO] Menggunakan device: {self.device}")
        self.load_model()
        self.train_model()
