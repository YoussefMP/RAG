from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from data_processor import get_dataloaders_with_labels
from Utils.io_operations import load_jsonl_dataset
from sklearn.metrics import classification_report
from Source.Logging.loggers import get_logger
from sequence_classifier import RobertaCRF
from Utils.labels import *
from tqdm import tqdm
from utils import *
import datetime
import torch
import time
import gc


CONFIG = {
    "OUTPUT_DIR": paths.pretrained_classifiers_folder,
    "MODEL_NAME": 'FacebookAI/xlm-roberta-large',

    # Hyperparameters
    "DEVICE": torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    "BATCH_SIZE": 8,
    "MAX_LENGTH": 256,
    "NUM_CLASSES": 2,
    "EPOCHS": 3,
    "LEARNING_RATE": 2e-7,

    # Versioning
    "VERSION": "v1.0",
    "Comment": "After fine-tuning and forgetting to save the results this is the second fine-tuning",
    "TRAINING_DATASET": "Annotated_dataset",
    "DATASET_VERSION": "VR5.3",
    "VALIDATION_VERSION": "VRV5.3",
    "CHECKPOINT": [1, 2, 3],
    "LOAD_CHECKPOINT": "2",

    # Pipeline
    "PIPELINE": ["TRAIN", "EVAL", "VALIDATE"]
}

logger = get_logger("trainer_logger", "Training_logs.log")


def train(model, dataloader, device, epochs, learning_rate, val_set=None):
    # losses records
    losses = []
    # Training parameters
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Training loop
    model.to(device)

    logger.info(f"\tStarting training")
    for epoch in range(epochs):
        logger.info(f"\t\tStarting epoch {epoch+1}")
        start = time.time()
        model.train()
        total_loss = 0
        batch_count = 0
        for batch in tqdm(dataloader, total=len(dataloader), desc=f"Training Epoch {epoch}: "):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            loss = model(input_ids, attention_mask, labels)
            total_loss += loss.item()
            if batch_count % 50 == 0:
                logger.info(f'\t\t\tTotal loss for batch : {loss.item()}')

            loss.backward()
            optimizer.step()
            scheduler.step()

            # optimizing by emptying cache and collecting garbage
            torch.cuda.empty_cache()
            gc.collect()

            if (batch_count == 0 or batch_count == len(dataloader)//2 or batch_count+1 == len(dataloader)) and\
                    val_set is not None:
                logger.info(f'\t\t\tValidation round ...')
                evaluate_model(model, val_set, device)

            batch_count += 1

        end = time.time()
        avg_train_loss = total_loss / len(dataloader)
        logger.info(
            f'Epoch {epoch+1}, Loss: {avg_train_loss} - runtime for epoch: {datetime.timedelta(seconds=end-start)}'
        )
        losses.append(avg_train_loss)

        if epoch + 1 in CONFIG["CHECKPOINT"] and epoch != epochs-1:
            logger.info(f"Saving trained model with config")
            save_model(CONFIG, model, losses, checkpoint=epoch)

    return losses


def evaluate_model(model, dataloader, device):
    # Transferring model to device
    model.to(device)
    # Evaluation
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc=f"Evaluation "):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            output = model(input_ids, attention_mask)
            predictions.extend(output)
            true_labels.extend([labels[i].tolist()[:len(o)] for i, o in enumerate(output)])

    # Convert predictions and labels to tag names
    pred_tags = [[ID2TAG[id] for id in pred] for pred in predictions]
    true_tags = [[ID2TAG[id] for id in true] for true in true_labels]

    # Flatten the lists for evaluation
    flat_pred_tags = [item for sublist in pred_tags for item in sublist]
    flat_true_tags = [item for sublist in true_tags for item in sublist]

    print(classification_report(flat_true_tags, flat_pred_tags, digits=4))
    logger.info(classification_report(flat_true_tags, flat_pred_tags, digits=4))


if __name__ == '__main__':

    # load training data from json file
    logger.info(f"Loading training data from json file")
    dataset = load_jsonl_dataset(os.path.join(
        paths.annotations_folder,
        f"{CONFIG['TRAINING_DATASET']}_{CONFIG['DATASET_VERSION']}.jsonl")
    )

    # Loading validation set from the separate file
    logger.info(f"Loading training data from json file")
    validation_set = load_jsonl_dataset(os.path.join(
        paths.annotations_folder,
        f"{CONFIG['TRAINING_DATASET']}_{CONFIG['VALIDATION_VERSION']}.jsonl")
    )

    logger.info(f"Initializing tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["MODEL_NAME"])         # Initialize the tokenizer

    logger.info(f"Initializing model")
    # if model version already exits load it for evaluation
    if os.path.exists(os.path.join(
            CONFIG["OUTPUT_DIR"],
            f"{CONFIG['MODEL_NAME'].split('/')[1]}_{str(CONFIG['VERSION'])}")):
        logger.info(f"Loading pretrained model Weights {CONFIG['MODEL_NAME']} version {CONFIG['VERSION']}")

        if CONFIG["LOAD_CHECKPOINT"] is None:
            weights_path = os.path.join(CONFIG["OUTPUT_DIR"],
                                        f"{CONFIG['MODEL_NAME'].split('/')[1]}_{str(CONFIG['VERSION'])}",
                                        f"{CONFIG['MODEL_NAME'].split('/')[1]}_{str(CONFIG['VERSION'])}")
        else:
            checkpoint_name =\
                f"{CONFIG['MODEL_NAME'].split('/')[1]}_{str(CONFIG['VERSION'])}_checkpoint_{CONFIG['LOAD_CHECKPOINT']}"
            weights_path = os.path.join(CONFIG["OUTPUT_DIR"],
                                        f"{CONFIG['MODEL_NAME'].split('/')[1]}_{str(CONFIG['VERSION'])}",
                                        checkpoint_name
                                        )
        checkpoint = torch.load(weights_path)
        classifier = RobertaCRF(model_name=CONFIG['MODEL_NAME'], num_labels=checkpoint['num_labels'])
        classifier.load_state_dict(checkpoint['model_state_dict'])

    else:
        classifier = RobertaCRF(CONFIG["MODEL_NAME"], CONFIG["NUM_CLASSES"])

    if "EVAL" in CONFIG["PIPELINE"]:
        # Split dataset into train and validation sets (for demonstration)
        dataset = dataset.train_test_split(test_size=0.2)
        # initialize dataloaders
        logger.info(f"Initializing dataloaders")
        train_dataloader = get_dataloaders_with_labels(tokenizer,
                                                       dataset["train"], CONFIG["BATCH_SIZE"],
                                                       TAG2ID,
                                                       CONFIG["MAX_LENGTH"]
                                                       )
        eval_dataloader = get_dataloaders_with_labels(tokenizer,
                                                      dataset["test"],
                                                      CONFIG["BATCH_SIZE"],
                                                      TAG2ID,
                                                      CONFIG["MAX_LENGTH"]
                                                      )
    else:
        # initialize dataloaders
        logger.info(f"Initializing dataloaders")
        train_dataloader = get_dataloaders_with_labels(tokenizer,
                                                       dataset, CONFIG["BATCH_SIZE"],
                                                       TAG2ID,
                                                       CONFIG["MAX_LENGTH"]
                                                       )
        eval_dataloader = None

    validation_dataloader = None
    if "VALIDATE" in CONFIG["PIPELINE"]:
        logger.info(f"Initializing validation dataloader")
        validation_dataloader = get_dataloaders_with_labels(tokenizer,
                                                            validation_set,
                                                            CONFIG["BATCH_SIZE"],
                                                            TAG2ID,
                                                            CONFIG["MAX_LENGTH"]
                                                            )

    if "TRAIN" in CONFIG["PIPELINE"]:
        losses = train(classifier, train_dataloader, CONFIG["DEVICE"], CONFIG["EPOCHS"], CONFIG["LEARNING_RATE"],
                       validation_dataloader)

        logger.info(f"Saving trained model with config")
        save_model(CONFIG, classifier, losses)

    if "EVAL" in CONFIG["PIPELINE"]:
        logger.info(f"Evaluating trained model")
        evaluate_model(classifier, eval_dataloader, CONFIG["DEVICE"])

