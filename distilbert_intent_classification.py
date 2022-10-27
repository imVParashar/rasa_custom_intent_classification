import logging
from pyexpat import model
from typing import Dict, Text, Any, List

import numpy as np
import torch

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig
from transformers import Trainer, TrainingArguments
from transformers import pipeline

from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData

from model_config import MAX_LEN, NUM_LABELS, NUM_EPOCHS, TRAIN_BATCH_SIZE, WARMUP_STEPS, WEIGHT_DECAY

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_ID = 0 if torch.cuda.is_available() else -1

logger =logging.getLogger(__name__)

class PrepareDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        if DEVICE == 'cuda':
            loss_fct = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).cuda())
        else:
            loss_fct = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).cpu())
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(pred):
    """
    Helper function to compute aggregated metrics from predictions.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# TODO: Correctly register your component with its type
@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER], is_trainable=True
)
class CustomNLUComponent(GraphComponent):
    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["transformers"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {
            "model_name": "distilbert-base-uncased"
            }

    def __init__(
        self,
        config: Dict[Text, Any],
        name: Text,
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        self.name = name
        # load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['model_name'],
            do_lower_case=True
            )
        # load the model config
        self.model_config = AutoConfig.from_pretrained(
            config['model_name'],
            num_labels=NUM_LABELS
            )
        # load the model and pass to CUDA
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config['model_name'], 
            config=self.model_config,
            ).to(DEVICE)

        # We need to use these later when saving the trained component.
        self._model_storage = model_storage
        self._resource = resource

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        # TODO: Implement this
        return cls(config, execution_context.node_name, model_storage, resource)

    def _prepare_dataset(self, training_data: TrainingData) -> PrepareDataset:
        # Transforming the data into suitable format
        text_list = []
        intent_list = []
        for e in training_data.training_examples:
            if e.get('text'):
                text_list.append(e.get('text'))
                intent_list.append(e.get('intent'))

        # tokenize the dataset, truncate when passed `max_length`, 
        # and pad with 0's when less than `max_length`

        train_encodings = self.tokenizer(text_list, truncation=True, padding=True, max_length=MAX_LEN)

        # Doing label encoding of labels
        le = LabelEncoder()
        label_no = le.fit_transform(intent_list)

        # convert our tokenized data into a torch Dataset
        train_dataset = PrepareDataset(train_encodings, label_no)

        # calculating id2label and label2id
        id2label = {k:v for k, v in enumerate(le.classes_)}
        label2id = {v:k for k, v in id2label.items()}

        target_names = le.classes_.tolist()

        # Calculating class weights
        global class_weights
        class_weights = class_weight.compute_class_weight(
            class_weight = 'balanced',
            classes = np.unique(intent_list),
            y = intent_list
            )

        return id2label, label2id, target_names, train_dataset

    def persist(self) -> None:
        with self._model_storage.write_to(self._resource) as model_dir:
            self.tokenizer.save_pretrained(model_dir)
            self.trainer.save_model(model_dir)
        

    def train(self, training_data: TrainingData) -> Resource:
        # TODO: Implement this if your component requires training
        id2label, label2id, target_names, train_data = self._prepare_dataset(training_data)

        # Updating model configuration
        self.model.config.num_labels = len(target_names)
        self.model.config.id2label = id2label
        self.model.label2id = label2id

        training_args = TrainingArguments(
            output_dir='./results',                         # output directory
            evaluation_strategy="no",                       # Evaluation is done at the end of each epoch.
            num_train_epochs=NUM_EPOCHS,                    # total number of training epochs
            per_device_train_batch_size=TRAIN_BATCH_SIZE,   # batch size per device during training
            warmup_steps=WARMUP_STEPS,                      # number of warmup steps for learning rate scheduler
            weight_decay=WEIGHT_DECAY,                      # strength of weight decay
            save_total_limit=1,                             # limit the total amount of checkpoints. Deletes the older checkpoints.    
        )

        self.trainer = CustomTrainer(
            model=self.model,                     # the instantiated Transformers model to be trained
            args=training_args,                   # training arguments, defined above
            train_dataset=train_data,             # training dataset
            compute_metrics=compute_metrics,      # the callback that computes metrics of interest
        )

        # train the model
        self.trainer.train()

        self.persist()

        return self._resource

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        with model_storage.read_from(resource) as model_dir:
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                do_lower_case=True,
                add_special_tokens=True,
                max_length=MAX_LEN,
                truncation=True,
                padding=True
                )
            classifier = pipeline(
                "text-classification",
                model=model_dir.as_posix(),
                tokenizer=tokenizer,
                return_all_scores=True,
                device=DEVICE_ID
                )

            component = cls(
                config, execution_context.node_name, model_storage, resource
            )
            component.model = classifier
            return component
        

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        # TODO: Implement this if your component augments the training data with
        #       tokens or message features which are used by other components
        #       during training.
        self.process(training_data.training_examples)
        return training_data

    def process(self, messages: List[Message]) -> List[Message]:
        # TODO: This is the method which Rasa Open Source will call during inference.
        for message in messages:
            predictions = self.model(message.data.get('text'))
            for pred in predictions[0]:
                pred['name'] = pred.pop('label')
            intent_ranking = sorted(predictions[0], key=lambda d: d['score'], reverse=True) 
            message.set("intent", intent_ranking[0], add_to_output=True)
            message.set("intent_ranking", intent_ranking[1:], add_to_output=True)

        logger.info(messages[0].data)
        return messages
