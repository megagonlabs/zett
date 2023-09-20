import json
import os
import shutil
from pathlib import Path
from typing import List, Tuple

from fire import Fire
from pydantic import BaseModel
from random import choices

template_map = {}
template_desc_map = {}
labelname2rid = {}
laebl_cls_id = {}


class DynamicModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True


class RelationSentence(BaseModel):
    tokens: List[str]
    head: List[int]
    tail: List[int]
    label: str
    head_id: str = ""
    tail_id: str = ""
    label_id: str = ""
    error: str = ""
    raw: str = ""
    score: float = 0.0
    head_text: str = ""
    tail_text: str = ""
    template: str = ""

    def as_tuple(self) -> Tuple[str, str, str]:
        head = " ".join([self.tokens[i] for i in self.head])
        tail = " ".join([self.tokens[i] for i in self.tail])
        return head, self.label, tail

    def as_line(self) -> str:
        return self.json() + "\n"

    def is_valid(self) -> bool:
        for x in [self.tokens, self.head, self.tail, self.label]:
            if len(x) == 0:
                return False
        for x in [self.head, self.tail]:
            if -1 in x:
                return False
        return True

    def get_template(self, label=None) -> str:
        if label is not None:
            self.template = template_map[labelname2rid[label]]
        elif len(self.template) == 0:
            self.template = template_map[labelname2rid[self.label]]
        return self.template

    def get_templated_summary(self, label=None) -> str:
        if len(self.head_text) == 0:
            self.head_text = " ".join([self.tokens[i] for i in self.head])
        if len(self.tail_text) == 0:
            self.tail_text = " ".join([self.tokens[i] for i in self.tail])
        template = self.get_template(label)
        x_idx = template.find("[X]")
        y_idx = template.find("[Y]")
        if x_idx < y_idx:
            template = template.replace("[X]", "<extra_id_0>")
            template = template.replace(" [Y]", "<extra_id_1>")
        else:
            template = template.replace(" [X]", "<extra_id_1>")
            template = template.replace("[Y]", "<extra_id_0>")
        return template
    
    def get_desc(self, label=None) -> str:
        if label is not None:
            return template_desc_map[labelname2rid[label]]
        return template_desc_map[labelname2rid[self.label]]

    def get_random_templated_summary(self, true_labels, n_target=5) -> List[str]:
        pool = list(set(labelname2rid.keys()) - set(true_labels))
        random_labels = choices(pool, k=n_target)
        return [self.get_templated_summary(label=x) for x in random_labels]

    def get_label_cls_id(self, label=None) -> int:
        if label is None:
            return laebl_cls_id[self.label]
        else:
            return laebl_cls_id[label]

    def get_summary(self, label=None) -> str:
        template = self.get_template(label)
        x_idx = template.find("[X]")
        y_idx = template.find("[Y]")
        assert x_idx != -1 and y_idx != -1
        s, r, o = self.as_tuple()

        if x_idx < y_idx:
            return f"<extra_id_0>{s}<extra_id_1>{o}<extra_id_2>"
        else:
            return f"<extra_id_0>{o}<extra_id_1>{s}<extra_id_2>"

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    @classmethod
    def from_spans(cls, text: str, head: str, tail: str, label: str, strict=True):
        tokens = text.split()
        sent = cls(
            tokens=tokens,
            head=head,
            tail=tail,
            label=label,
            label_id=labelname2rid[label],
        )
        if strict:
            assert sent.is_valid(), (head, label, tail, text)
        return sent

    def as_marked_text(self) -> str:
        tokens = list(self.tokens)
        for i, template in [
            (self.head[0], "[H {}"),
            (self.head[-1], "{} ]"),
            (self.tail[0], "[T {}"),
            (self.tail[-1], "{} ]"),
        ]:
            tokens[i] = template.format(tokens[i])
        return " ".join(tokens)


def delete_checkpoints(
    folder: str = ".", pattern="**/checkpoint*", delete: bool = True
):
    for p in Path(folder).glob(pattern):
        if (p.parent / "config.json").exists():
            print(p)
            if delete:
                if p.is_dir():
                    shutil.rmtree(p)
                elif p.is_file():
                    os.remove(p)
                else:
                    raise ValueError("Unknown Type")


if __name__ == "__main__":
    Fire()
