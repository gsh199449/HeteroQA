
# How to get the dataset
Signed the following copyright announcement with your name and organization. Then complete the form online (https://forms.gle/bcLsVzUnxfahA7T96) and **mail** to shengao#pku.edu.cn ('#'->'@'), we will send you the corpus by e-mail when approved.

# Copyright
The original copyright of all the data belongs to the source owner.
The copyright of annotation belongs to our group, and they are free to the public.
The dataset is only for research purposes. Without permission, it may not be used for any commercial purposes and distributed to others.

# Data Sample
Each line in the JSON file represents one data sample, which contains the dialog history and the ground truth sticker.

|  Json Key Name  | Description                                |
|:---------------:|--------------------------------------------|
| content |  Question (str)  |
| summary |  Answer (str)  |
| retrieval |  (Object)  |
| retrieval.search_qa |  (List)  |
| retrieval.search_qa.question |  similar question (str)  |
| retrieval.search_qa.answer |  similar answer (str)  |
| retrieval.search_qa.score |  similarity for similar qa (float)  |
| retrieval.search_xueqiu |  (List)  |
| retrieval.search_xueqiu.content |  sentences in the related article (List of str)  |
| retrieval.search_xueqiu.score |  similarity for the related article (float)  |
| retrieval.search_xueqiu.title |  title for the related article (str)  |

A random sampled data is shown below:

```json5
{
    "content": "how many square decimeters",
    "summary": "There are 0.0001 square decimeters in a square millimeter.",
    "retrieval": {
        "search_qa": [
            {
                "answer": "10.76 square feet",
                "question": "how many square feet to square meter",
                "score": 13.649996
            },
            {
                "answer": "9 square feet.",
                "question": "how many square feet in a square yard",
                "score": 13.001198
            },
        ],
        "search_xueqiu": [
            {
                "content": [
                    "Online calculators to convert square millimeters to square decimeters (mm2 to dm2) and square decimeters to square millimeters (dm2 to mm2) with formulas, examples, and tables. Our conversions provide a quick and easy way to convert between Area units."
                ],
                "score": 0,
                "title": ""
            },
            {
                "content": [
                    "SQUARE METER TO SQUARE DECIMETER (m2 TO dm2) FORMULA . To convert between Square Meter and Square Decimeter you have to do the following: First divide 1 / 0.01 = 100. Then multiply the amount of Square Meter you want to convert to Square Decimeter, use the chart below to guide you. SQUARE METER TO SQUARE DECIMETER (m2 TO dm2) CHART"
                ],
                "score": 0,
                "title": ""
            },
        ]
    }
}
```

# Training

Install python 3.7 and torch (torch-1.5.0+cu101-cp37-cp37m-linux_x86_64).
Other packages are listed in the `requirements.txt`.

Modify the [L458](https://github.com/gsh199449/HeteroQA/blob/master/run_mybart.py#L458) in `run_mybart.py` to define the dataset name (`DATASET_NAME_YOU_DEFINED`). Then run the command below to make a dataset file:
```bash
python3 run_mybart.py --do_train --do_eval --model_name_or_path PATH_OF_bart-base --train_file msm_plus/train.json --validation_file msm_plus/validation.json --test_file msm_plus/validation.json --save_steps 500 --save_total_limit 20 --output_dir das --exp_name msm-make-data --max_source_length 200 --max_target_length 100
```

Training command
```bash
python3 run_mybart.py --model_name_or_path PATH_OF_bart-base --save_dataset_path DATASET_NAME_YOU_DEFINED --exp_name msm-lab --two_hidden_merge_method add --do_train --do_eval --eval_steps 3000 --evaluation_strategy steps --predict_with_generate True --output_dir model/ --save_steps 3000 --save_total_limit 3 --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --dataloader_num_workers 8 --num_train_epochs 10 --add_bidirectional_edge True --hetero_graph True --hetero_graph_model hgt --magic_hgt True --add_graph_loss 0.01
```

# Evaluation

Using `metrics/test_file.py` to evaluate the decoded file. First, change the path in [`metrics/test_file.py` L106](https://github.com/gsh199449/HeteroQA/blob/master/metrics/test_file.py#L106), then run the script.

Generated Answers of test dataset:

<!-- [AntQA](https://drive.google.com/file/d/185I3YrxsLjTGAk7ABE9ZFaP7cBf4wm_D/view?usp=sharing) -->

[MSM-Plus](https://drive.google.com/file/d/1VCsiiao6lrVO-e-3wiE04OzcS56-Ii2U/view?usp=sharing)