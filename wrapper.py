import json
import random
from pathlib import Path
from typing import List

import torch
from fire import Fire
from pydantic.main import BaseModel
from tqdm import tqdm

from modeling import (RelationModel, ZETTTripletExtractor)
from utils import (RelationSentence, delete_checkpoints)

from transformers import AutoModelForSeq2SeqLM
from os.path import exists
import pandas as pd
import numpy as np
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def safe_divide(a: float, b: float) -> float:
    if a == 0 or b == 0:
        return 0
    return a / b


class Sentence(BaseModel):
    triplets: List[RelationSentence]

    @property
    def tokens(self) -> List[str]:
        return self.triplets[0].tokens

    @property
    def label(self) -> str:
        return self.triplets[0].label
    
    @property
    def all_labels(self) -> List[str]:
        return [x.label for x in self.triplets]

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    @property
    def head_entity(self) -> str:
        result = ""
        for x in self.triplets[0].head:
            result += self.triplets[0].tokens[x] + " "
        return result.strip()

    @property
    def tail_entity(self) -> str:
        result = ""
        for x in self.triplets[0].tail:
            result += self.triplets[0].tokens[x] + " "
        return result.strip()

    def assert_valid(self):
        assert len(self.tokens) > 0
        for t in self.triplets:
            assert t.text == self.text
            assert len(t.head) > 0
            assert len(t.tail) > 0
            assert len(t.label) > 0


class Dataset(BaseModel):
    sents: List[Sentence]
    label2sents: dict = {}
    label2triplets: dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for s in self.sents:
            for t in s.triplets:
                if t.label not in self.label2triplets:
                    self.label2triplets[t.label] = []
                self.label2triplets[t.label].append(t)


    def get_labels(self) -> List[str]:
        return sorted(set(t.label for s in self.sents for t in s.triplets))

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            sents = [Sentence(**json.loads(line)) for line in f]
        return cls(sents=sents)

    def save(self, path: str):
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            for s in self.sents:
                f.write(s.json() + "\n")

    def filter_labels(self, labels: List[str]):
        label_set = set(labels)
        sents = []
        for s in self.sents:
            triplets = [t for t in s.triplets if t.label in label_set]
            if triplets:
                _s = s.copy(deep=True)
                _s.triplets = triplets
                sents.append(_s)
        return Dataset(sents=sents)

            
class Extractor(BaseModel):
    load_dir: str
    save_dir: str
    threshold: float = 2.5 # for multi triplet evaluation
    label_constraint_th: float = 0.85
    model_kwargs: dict = {}
    relname2template : dict = {}

    templ_filepath: str
    relname2desc : dict = {}

    def __init__(self, **data):
        super().__init__(**data)
        for line in open(self.templ_filepath):
            tokens = line.strip().split("\t")
            rid, rel, templ, desc = tokens[:4]
            self.relname2desc[rel] = desc
            self.relname2template[rel] = templ

    def get_model(self, load_dir: str = None) -> RelationModel:
        if load_dir is not None:
            model_dir = load_dir
        else:
            model_dir = str(Path(self.save_dir) / "model")

        kwargs = dict(model_dir=model_dir, # output dir
                        model_name=self.load_dir,
                        data_dir=str(Path(self.save_dir) / "data"),
                        do_pretrain=False,
                        **self.model_kwargs,)
        model = ZETTTripletExtractor(**kwargs)
        return model

    def encode_to_line(self, sent: RelationSentence) -> str:
        template = self.relname2template[sent.label]
        x = f"{sent.text}</s>{self.get_templated_input(template)}</s>"

        s, r, o = sent.as_tuple()
        y = self.encode_y(template, s, o)
        return json.dumps(dict(text=x, summary=y)) + "\n"

    def write_data(self, data: Dataset, name: str) -> str:
        model = self.get_model()
        path_out = Path(model.data_dir) / f"{name}.json"
        if exists(path_out):
            return str(path_out)
        path_out.parent.mkdir(exist_ok=True, parents=True)
        lines = [self.encode_to_line(t) for s in data.sents for t in s.triplets]
        random.seed(model.random_seed)
        random.shuffle(lines)
        with open(path_out, "w") as f:
            f.write("".join(lines))
        return str(path_out)

    def fit(self, path_train: str, path_dev: str):
        model = self.get_model()
        if exists(model.model_dir + "/config.json"):
            return

        data_train = Dataset.load(path_train)
        data_dev = Dataset.load(path_dev)
        path_train = self.write_data(data_train, "train")
        path_dev = self.write_data(data_dev, "dev")
        model.fit(path_train=path_train, path_dev=path_dev)
        delete_checkpoints(model.model_dir)

    def get_templated_input(self, template):
        x_idx = template.find("[X]")
        y_idx = template.find("[Y]")
        if x_idx < y_idx:
            template = template.replace("[X]", "<extra_id_0>")
            template = template.replace(" [Y]", "<extra_id_1>")
        else:
            template = template.replace(" [X]", "<extra_id_1>")
            template = template.replace("[Y]", "<extra_id_0>")
        return template

    def encode_y(self, template, s, o) -> str:
        x_idx = template.find("[X]")
        y_idx = template.find("[Y]")
        if x_idx < y_idx:
            return f"<extra_id_0>{s}<extra_id_1>{o}<extra_id_2>"
        else:
            return f"<extra_id_0>{o}<extra_id_1>{s}<extra_id_2>"

    def generate_entity_span(self, data, generator, tokenizer, target_labels, mode, task_type, use_label_constraint):
        eos_token_id = tokenizer.convert_tokens_to_ids("<extra_id_2>")
        entire_vocab = set(tokenizer.vocab.values()) - set(tokenizer.convert_tokens_to_ids(["<extra_id_0>", "<extra_id_1>", "<extra_id_2>"]))
        beam_size = 4

        if mode in ["auto_templ_single", "auto_templ_multi"]:
            test_auto_templs = self.load_test_auto_templ()

        if use_label_constraint is True:
            rv_model = SentenceTransformer('all-MiniLM-L6-v2')
            rv_model = rv_model.to(device)
            rv_model = rv_model.eval()

        test_data = []
        for i in tqdm(range(0, len(data.sents))):
            batch = {"input_ids": [], "attention_mask": [], "labels": [], "input_text": [],
                        "outputs": [], "relations": [], "loss_list": [], "final_input_text": [], "final_relations": [],
                        "templates": [], "final_templates": []}
            example = data.sents[i]
            context = example.text

            if task_type == "EE": # for entity extraction, gold label (relation) is given.
                final_target_labels = [example.label]
            else:
                if use_label_constraint is True:
                    context_emb = rv_model.encode(context)
                    rel2dist_score = {}
                    for k in target_labels:
                        _rv_input_text = f"{k}. {self.relname2desc[k]}"
                        rel_emb = rv_model.encode(_rv_input_text)
                        rel2dist_score[k] = distance.cosine(context_emb, rel_emb)
                    rel_scores =  {k: v for k, v in sorted(rel2dist_score.items(), key=lambda item: item[1])}
                    final_target_labels = []
                    for jj, (k, v) in enumerate(rel_scores.items(), 1):
                        if v < self.label_constraint_th or (jj==1): # need at least 1
                            final_target_labels.append(k)
                else:
                    final_target_labels = target_labels
                
            for k in final_target_labels:
                if mode in ["auto_templ_single", "auto_templ_multi"]:
                    input_text = []
                    for nt in range(min(self.top_n_templ, len(test_auto_templs[k]))):
                        _templ = test_auto_templs[k][nt]
                        batch["templates"].append(_templ)
                        batch["relations"].append(k)
                        input_text.append(f"{context}</s>{self.get_templated_input(_templ)}</s>")
                else:
                    _templ = self.relname2template[k]
                    batch["relations"].append(k)
                    batch["templates"].append(_templ)
                    input_text = [f"{context}</s>{self.get_templated_input(_templ)}</s>"]

                    if task_type == "RC": # only for relation classification. two entities are given.
                        assert len(example.triplets) == 1
                        s, r, o = example.triplets[0].as_tuple()
                        out_w_gold_ent = self.encode_y(template=_templ, s=s, o=o)
                        batch["outputs"].append(out_w_gold_ent)

                batch["input_text"].extend(input_text)

            if task_type in ["TE", "EE"]: # For Triplet/Entity extraction, we need to generate entities
                _, out = generator.run(batch["input_text"], # len(out) is len(input_text) * num_retrun
                                    do_sample=False, num_beams=beam_size, top_k=None,
                                    num_return=beam_size, eos_token_id=eos_token_id, save_scores=False,
                                    bad_words_ids=[[x] for x in (entire_vocab - set(tokenizer(context).input_ids))])
                for it, j in enumerate(range(0, len(out), beam_size)):
                    valid_out = []
                    for o in out[j:j+beam_size]:
                        pred = o.replace("<extra_id_0>", "").replace("<extra_id_2>", "").strip()
                        tokens = pred.split("<extra_id_1>")
                        if len(tokens) != 2:
                            continue
                        first, second = [x.strip() for x in tokens]
                        if (context.find(first) != -1) and (context.find(second) != -1) and (first != second):
                            if o not in valid_out:
                                valid_out.append(o)
                    if len(valid_out) == 0:
                        valid_out.append(out[j])

                    for m_out in valid_out:
                        batch["final_input_text"].append(batch["input_text"][it])
                        batch["final_relations"].append(batch["relations"][it])
                        if mode in ["auto_templ_single", "auto_templ_multi"]:
                            batch["final_templates"].append(batch["templates"][it])
                        batch["outputs"].append(m_out)
            else: # for RC
                batch["final_input_text"] = batch["input_text"]
                batch["final_relations"] = batch["relations"]

            # tokenization
            temp = tokenizer(batch["final_input_text"], padding=True, return_tensors="pt")
            batch["input_ids"] = temp.input_ids
            batch["attention_mask"] = temp.attention_mask
            batch["labels"] = tokenizer(batch["outputs"], padding=True, return_tensors="pt").input_ids
            test_data.append(batch)

        assert len(data.sents) == len(test_data)
        return test_data

    def load_test_auto_templ(self):
        raise NotImplementedError("Use this class: AutoTemplExtractor")

    def predict(self, 
                data_dir:str, path_in: str, path_out: str, 
                mode: str = "single", use_label_constraint: bool = True, 
                load_dir: str = None, target_labels = None, task_type: str = "TE"):
        # load test data         
        data = Dataset.load(path_in)
        # load a model
        model = self.get_model(load_dir)
        gen = model.load_generator(device=device)
        tokenizer = gen.tokenizer
        print("\n\t\t\t |||||| loaded model: ", model, "\n\n")

        # 1) generate entity spans
        test_data = self.generate_entity_span(data, gen, tokenizer,
                                                target_labels=target_labels,
                                                mode=mode, task_type=task_type,
                                                use_label_constraint=use_label_constraint)
    
        # 2) score
        model = AutoModelForSeq2SeqLM.from_pretrained(load_dir)
        model = model.to(device)
        model.eval()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        for i in tqdm(range(0, len(test_data)), desc="scoring"):
            batch = test_data[i]
            # Use this when you have enough memory
            # outputs = model(input_ids=batch["input_ids"].to(device),
            #                 attention_mask=batch["attention_mask"].to(device),
            #                 labels=batch["labels"].to(device))
            # loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            # _labels = batch["labels"][:]
            # _labels[_labels == tokenizer.pad_token_id] = -100
            # masked_lm_loss = loss_fct(outputs.logits.view(-1, model.config.vocab_size), _labels.view(-1).to(device))
            # masked_lm_loss = masked_lm_loss.view(_labels.shape).sum(dim=1).tolist()
            # batch["loss_list"] = masked_lm_loss

            # To reduce GPU usage.
            temp_batch_size = 5
            temp_mlm_loss = []
            for jj in range(0, len(batch["input_ids"]), temp_batch_size):
                end_idx = len(batch["input_ids"]) if jj+temp_batch_size > len(batch["input_ids"]) else jj + temp_batch_size
                    
                outputs = model(input_ids=batch["input_ids"][jj:end_idx].to(device),
                                attention_mask=batch["attention_mask"][jj:end_idx].to(device),
                                labels=batch["labels"][jj:end_idx].to(device))  
                _labels = batch["labels"][jj:end_idx]
                _labels[_labels == tokenizer.pad_token_id] = -100
                _lm_loss = loss_fct(outputs.logits.view(-1, model.config.vocab_size), _labels.view(-1).to(device))
                _lm_loss = _lm_loss.view(_labels.shape).sum(dim=1).tolist()
                temp_mlm_loss.extend(_lm_loss)
            batch["loss_list"] = temp_mlm_loss
        # Done scoring

        # 3) Decode to triplet and export .csv file for debugging
        predictions = []
        export_data = { # for debugging, to export all results to csv.
            "test_id": [], "input_text": [],
            "gold_head": [], "gold_tail": [], "gold_rel": [],
            "pred_head": [], "pred_tail": [], "pred_rel": [], "output_text": [],
            "score": [], "T/F": [],
        }
        assert len(data.sents) == len(test_data)
        for i in range(0, len(test_data)):            
            loss_list = {}
            preds = []
            gold_label = data.sents[i].label
            context = data.sents[i].text

            batch = test_data[i]
            for jj in range(len(batch["final_input_text"])):
                cur_label = batch["final_relations"][jj]
                _loss = batch["loss_list"][jj]
                loss_list[jj] = _loss
                try: # parse model's output into a triplet form
                    pred = batch["outputs"][jj].replace("<extra_id_0>", "").replace("<extra_id_2>", "").strip()
                    first, second = [x.strip() for x in pred.split("<extra_id_1>")]
                    if mode in ["auto_templ_single", "auto_templ_multi"]:
                        template = batch["final_templates"][jj]
                    else:
                        template = self.relname2template[cur_label]
                    x_idx = template.find("[X]")
                    y_idx = template.find("[Y]")
                    assert x_idx != -1 and y_idx != -1
                    if x_idx < y_idx:
                        head_entity, tail_entity = first, second
                    else:
                        head_entity, tail_entity = second, first
                    new_sent = RelationSentence.from_spans(context, head=head_entity, tail=tail_entity, label=cur_label)
                    new_sent.head_text = head_entity
                    new_sent.tail_text = tail_entity
                    new_sent.raw = batch["outputs"][jj]
                except: # parsing error
                    new_sent = RelationSentence(tokens=context.split(),
                                                head=[], tail=[], raw=batch["outputs"][jj], label=cur_label)
                new_sent.score = _loss
                preds.append(new_sent)

            xx =  {k: v for k, v in sorted(loss_list.items(), key=lambda item: item[1])}
            final_preds = []
            for j, (k, _loss) in enumerate(xx.items()):
                final_preds.append(preds[k])
                export_data["test_id"].append(f"{i+1}_rank{j+1}")
                export_data["input_text"].append(batch["final_input_text"][k])
                export_data["gold_rel"].append(gold_label)
                export_data["gold_head"].append(data.sents[i].head_entity)
                export_data["gold_tail"].append(data.sents[i].tail_entity)
                export_data["pred_rel"].append(preds[k].label)
                export_data["pred_head"].append(preds[k].head_text)
                export_data["pred_tail"].append(preds[k].tail_text)
                export_data["score"].append(round(_loss, 4))
                export_data["output_text"].append(preds[k].raw)
                if (gold_label == preds[k].label) and (data.sents[i].head_entity == preds[k].head_text) and (data.sents[i].tail_entity == preds[k].tail_text):
                    export_data["T/F"].append(True)
                else:
                    export_data["T/F"].append(False)

            if mode in ["single", "auto_templ_single"]:
                predictions.append(Sentence(triplets=[final_preds[0]]))
            else:
                __final_preds = []
                final_preds_set = set()
                for x in final_preds:
                    triplet_str = f"{x.head_text}_{x.label}_{x.tail_text}"
                    if (x.score < self.threshold) and (triplet_str not in final_preds_set):
                        __final_preds.append(x)
                        final_preds_set.add(triplet_str)
                predictions.append(Sentence(triplets=__final_preds))

        df = pd.DataFrame(data=export_data) 
        csv_out = path_out.split(".")[0] + ".csv"
        df.to_csv(csv_out)
        Dataset(sents=predictions).save(path_out)

    @staticmethod
    def score(path_pred: str, path_gold: str) -> dict:
        pred = Dataset.load(path_pred)
        gold = Dataset.load(path_gold)
        assert len(pred.sents) == len(gold.sents)
        
        num_pred, num_gold = 0, 0
        num_correct = 0
        num_correct_rc = 0
        num_correct_ee = 0 # in this case entity order is ignored 
        num_pred_ee, num_gold_ee = 0, 0
        num_correct_eht = 0 # in this case entity order is considered 

        # performance by label
        print("[Test] unseen relation set: ", gold.get_labels())
        results_by_label = {t: dict(tp=0, RC_tp=0, RC_p=0, RC_r=0, n_pred=0, n_true=0, precision=0, recall=0) for t in gold.get_labels()}

        for i in range(len(gold.sents)):
            num_pred += len(pred.sents[i].triplets)
            num_gold += len(gold.sents[i].triplets)
            num_pred_ee += (2*len(pred.sents[i].triplets))
            num_gold_ee += (2*len(gold.sents[i].triplets))
            for g in gold.sents[i].triplets:
                results_by_label[g.label]["n_true"] += 1
            for p in pred.sents[i].triplets:
                results_by_label[p.label]["n_pred"] += 1
                for g in gold.sents[i].triplets:
                    if (p.head, p.tail, p.label) == (g.head, g.tail, g.label):
                        num_correct += 1
                        results_by_label[g.label]["tp"] += 1 
                    if p.label == g.label:
                        num_correct_rc += 1
                        results_by_label[g.label]["RC_tp"] += 1 
                    if p.head == g.head or p.head == g.tail:
                        num_correct_ee += 1
                    if p.tail == g.head or p.tail == g.tail:
                        num_correct_ee += 1
                    if p.head == g.head and p.tail == g.tail:
                        num_correct_eht += 1
                    

        precision = safe_divide(num_correct, num_pred)
        recall = safe_divide(num_correct, num_gold)

        precision_rc = safe_divide(num_correct_rc, num_pred)
        recall_rc = safe_divide(num_correct_rc, num_gold)

        precision_eht = safe_divide(num_correct_eht, num_pred)
        recall_eht = safe_divide(num_correct_eht, num_gold)

        precision_ee = safe_divide(num_correct_ee, num_pred_ee)
        recall_ee = safe_divide(num_correct_ee, num_gold_ee)

        for k, v in results_by_label.items():
            v["precision"] = safe_divide(v["tp"], v["n_pred"])
            v["recall"] = safe_divide(v["tp"], v["n_true"])

            v["RC_p"] = safe_divide(v["RC_tp"], v["n_pred"])
            v["RC_r"] = safe_divide(v["RC_tp"], v["n_true"])

        macro_p = sum([v["RC_p"] for k, v in results_by_label.items()]) / len(results_by_label)
        macro_r = sum([v["RC_r"] for k, v in results_by_label.items()]) / len(results_by_label)

        info = dict(
            path_pred=path_pred,
            path_gold=path_gold,
            precision=precision,
            recall=recall,
            score=safe_divide(2 * precision * recall, precision + recall),
            RC_micro_p=precision_rc,
            RC_micro_r=recall_rc,
            RC_micro_f1=safe_divide(2 * precision_rc * recall_rc, precision_rc + recall_rc),
            RC_Macro_p=macro_p,
            RC_Macro_r=macro_r,
            RC_Macro_f1=safe_divide(2 * macro_p * macro_r, (macro_p + macro_r)),
            precision_eht=precision_eht,
            recall_eht=recall_eht,
            score_eht=safe_divide(2 * precision_eht * recall_eht, precision_eht + recall_eht),
            precision_ee=precision_ee,
            recall_ee=recall_ee,
            score_ee=safe_divide(2 * precision_ee * recall_ee, precision_ee + recall_ee),
        )
        info.update(results_by_label=results_by_label)
        return info


if __name__ == "__main__":
    Fire()
