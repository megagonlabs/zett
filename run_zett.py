from pathlib import Path
import json
import os
import fire
from typing import List
import numpy as np

from wrapper import Extractor, Sentence, Dataset


def train(data_name: List[str] = ["fewrel", "wiki"],
            n_unseen_rel: List[int] = [5, 10, 15],
            rd_fold: List[int] = [0, 1, 2, 3, 4],
            model_name: str = "baseline",
            templ_file: str = "templates/templates.tsv"):
    for dn in data_name:
        for num_unseen_labels in n_unseen_rel:
            for random_seed in rd_fold:
                data_dir = f"outputs/data/{dn}/unseen_{num_unseen_labels}_seed_{random_seed}"
                path_dev = data_dir + "/dev.jsonl"
                path_train = data_dir + "/train.jsonl"
                save_dir = f"outputs/wrapper/{dn}/unseen_{num_unseen_labels}_seed_{random_seed}/{model_name}"
                print(dict(data_dir=data_dir, save_dir=save_dir))

                extractor = Extractor(
                    load_dir="t5-base",
                    save_dir=str(Path(save_dir) / "extractor"),
                    model_kwargs=dict(
                        epochs_finetune=3,  # num_train_epochs
                        batch_size=8,       # per_device_train_batch_size
                        grad_accumulation=8, # gradient_accumulation_steps
                        lr_finetune=3e-5,
                        random_seed=42,),
                    templ_filepath=templ_file
                )

                extractor.fit(path_train, path_dev)
                
                test_data=Dataset.load(path_dev)
                # At training time, evaluate on the dev set only single cases
                test_data.sents = [s for s in test_data.sents if len(s.triplets) == 1]
                test_label = test_data.get_labels()
                
                path_pred = str(Path(save_dir) / "pred.jsonl")

                extractor.predict(
                    data_dir=data_dir, path_in=path_dev, path_out=path_pred,
                    load_dir=str(Path(save_dir) / "extractor" / "model"),
                    target_labels=test_label)
                
                results = extractor.score(path_pred, path_dev)
                print(json.dumps(results, indent=2))
                with open(Path(save_dir) / "results.json", "w") as f:
                    json.dump(results, f, indent=2)
                

def test(data_name: List[str] = ["fewrel", "wiki"],
            n_unseen_rel: List[int] = [5, 10, 15],
            rd_fold: List[int] = [0, 1, 2, 3, 4],
            model_name: str = "baseline",
            eval_mode: str = "single",
            task_type: str = "TE", use_label_constraint=True, 
            templ_file="templates/templates.tsv"):
    final_results = {}
    for dn in data_name:
        final_results[dn] = {}
        for num_unseen_labels in n_unseen_rel:
            final_results[dn][num_unseen_labels] = {}
            for random_seed in rd_fold:
                data_dir = f"outputs/data/{dn}/unseen_{num_unseen_labels}_seed_{random_seed}"
                save_dir = f"outputs/wrapper/{dn}/unseen_{num_unseen_labels}_seed_{random_seed}/{model_name}"
                path_test = data_dir + "/test.jsonl"
                print(dict(data_dir=data_dir, save_dir=save_dir))

                path_model = save_dir + "/extractor"
                path_results = str(Path(path_model) / f"results_{task_type}_{eval_mode}.json")
                if not os.path.exists(path_results):
                    test_data = Dataset.load(path_test)

                    load_dir = path_model + "/model"
                    model = Extractor(load_dir=load_dir,
                                    save_dir=path_model,
                                    templ_filepath=templ_file)

                    if task_type == "RC": # Relation Classification
                        rc_sents = []
                        for s in test_data.sents: 
                            for t in s.triplets: # for RC, all triplets are targets
                                rc_sents.append(Sentence(triplets=[t]))
                        test_data = Dataset(sents=rc_sents)
                    elif task_type == "EE": # Entity Extraction
                        test_data.sents = [s for s in test_data.sents if len(s.triplets) == 1]
                    # TE: Triplet Extraction
                    elif task_type == "TE" and (eval_mode == "single" or eval_mode == "auto_templ_single"):
                        test_data.sents = [s for s in test_data.sents if len(s.triplets) == 1]
                    elif task_type == "TE" and eval_mode == "multi":
                        test_data.sents = [s for s in test_data.sents if len(s.triplets) > 1]
                    else:
                        raise ValueError(f"mode must be single or multi")

                    path_in = str(Path(path_model) / f"pred_in_{task_type}_{eval_mode}.jsonl")
                    test_data.save(path_in)
                    path_out = str(Path(path_model) / f"pred_out_{task_type}_{eval_mode}.jsonl")
                    test_label = test_data.get_labels()
                    
                    model.predict(
                        data_dir=data_dir, path_in=path_in, path_out=path_out,
                        load_dir=load_dir, mode=eval_mode,
                        use_label_constraint=use_label_constraint,
                        target_labels=test_label, task_type=task_type)

                    results = model.score(path_pred=path_out, path_gold=path_in)
                    results.update(mode=eval_mode, path_results=path_results)
                    with open(path_results, "w") as fw:
                        json.dump(results, fw, indent=2)
                
                results = json.load(open(path_results))
                assert results["mode"] == eval_mode
                final_results[dn][num_unseen_labels][random_seed] = results

    print(json.dumps(final_results, indent=2))
    for dn in final_results:
        for n_label in final_results[dn]:
            print(task_type, dn, ", num_unseen_labels: ", n_label)
            for rd in final_results[dn][n_label]:
                if task_type == "TE":
                    print(f'{final_results[dn][n_label][rd]["precision"]} {final_results[dn][n_label][rd]["recall"]} {final_results[dn][n_label][rd]["score"]}')
                elif task_type == "RC":
                    print(f'{final_results[dn][n_label][rd]["RC_Macro_p"]} {final_results[dn][n_label][rd]["RC_Macro_r"]} {final_results[dn][n_label][rd]["RC_Macro_f1"]}')
                else:
                    print(f'{final_results[dn][n_label][rd]["score_ee"]} {final_results[dn][n_label][rd]["score_eht"]}')


def param_tune(data_name: List[str] = ["fewrel", "wiki"],
            n_unseen_rel: List[int] = [5, 10, 15],
            rd_fold: List[int] = [0, 1, 2, 3, 4],
            model_name: str = "baseline",
            eval_mode: str = "single",
            task_type: str = "TE"):
    final_results = {}
    for dn in data_name:
        final_results[dn] = {}
        for num_unseen_labels in n_unseen_rel:
            final_results[dn][num_unseen_labels] = {}
            for random_seed in rd_fold:
                best_th, best_score = -1, -1
    
                final_results[dn][num_unseen_labels][random_seed] = {}
                data_dir = f"outputs/data/{dn}/unseen_{num_unseen_labels}_seed_{random_seed}"
                save_dir = f"outputs/wrapper/{dn}/unseen_{num_unseen_labels}_seed_{random_seed}/{model_name}"
                path_test = data_dir + "/dev.jsonl"
                print(dict(data_dir=data_dir, save_dir=save_dir))

                path_model = save_dir + "/extractor"
                path_results = str(Path(path_model) / f"results_{task_type}_{eval_mode}.json")
                
                test_data = Dataset.load(path_test)

                load_dir = path_model + "/model"
                model = Extractor(load_dir=load_dir, save_dir=path_model)

                if task_type == "TE" and (eval_mode == "single" or eval_mode == "auto_templ_single"):
                    test_data.sents = [s for s in test_data.sents if len(s.triplets) == 1]
                elif task_type == "TE" and eval_mode == "multi":
                    test_data.sents = [s for s in test_data.sents if len(s.triplets) > 1]
                else:
                    raise ValueError(f"mode must be single or multi")

                path_in = str(Path(path_model) / f"pred_in_{task_type}_{eval_mode}.jsonl")
                test_data.save(path_in)
                path_out = str(Path(path_model) / f"pred_out_{task_type}_{eval_mode}.jsonl")
                test_label = test_data.get_labels()
                    
                for th in np.arange(2.0, 3.7, 0.1):
                # for th in np.arange(0.85, 0.92, 0.01):
                    model.threshold = th
                    #model.label_constraint_th = th
                    model.predict(
                        data_dir=data_dir, path_in=path_in, path_out=path_out,
                        load_dir=load_dir, mode=eval_mode,
                        use_label_constraint=True,
                        target_labels=test_label, task_type=task_type)

                    results = model.score(path_pred=path_out, path_gold=path_in)
                    results.update(mode="param_tune", threshold=th)
                    final_results[dn][num_unseen_labels][random_seed][str(th)] = results
                    if results["score"] > best_score:
                        best_score = results["score"]
                        best_th = th

                # evaluate with the best performing threshold
                print(f"Best score: {best_score}, Best threshold: {best_th}")

                path_test = data_dir + "/test.jsonl"
                test_data = Dataset.load(path_test)

                if task_type == "TE" and (eval_mode == "single" or eval_mode == "auto_templ_single"):
                    test_data.sents = [s for s in test_data.sents if len(s.triplets) == 1]
                elif task_type == "TE" and eval_mode == "multi":
                    test_data.sents = [s for s in test_data.sents if len(s.triplets) > 1]
                else:
                    raise ValueError(f"mode must be single or multi")

                path_in = str(Path(path_model) / f"pred_in_{task_type}_{eval_mode}.jsonl")
                test_data.save(path_in)
                path_out = str(Path(path_model) / f"pred_out_{task_type}_{eval_mode}.jsonl")
                test_label = test_data.get_labels()

                model.threshold = best_th
                model.predict(
                        data_dir=data_dir, path_in=path_in, path_out=path_out,
                        load_dir=load_dir, mode=eval_mode,
                        use_label_constraint=True,
                        target_labels=test_label, task_type=task_type)

                results = model.score(path_pred=path_out, path_gold=path_in)
                results.update(mode=eval_mode, threshold=best_th)
                final_results[dn][num_unseen_labels][random_seed]["final"] = results

    for dn in final_results:
        for n_label in final_results[dn]:
            for rd in final_results[dn][n_label]:
                print(task_type, dn, ", num_unseen_labels: ", n_label, " data_fold: ", rd)
                for th in final_results[dn][n_label][rd]:
                    print(f'{th}: {final_results[dn][n_label][rd][th]["precision"]} {final_results[dn][n_label][rd][th]["recall"]} {final_results[dn][n_label][rd][th]["score"]}')


if __name__ == "__main__":
    fire.Fire()
