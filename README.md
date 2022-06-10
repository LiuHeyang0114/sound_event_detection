## Sound Event Detection Project

### How To Run?

1. 提取特征：
```bash
cd data;
sbatch prepare_data.sh /dssg/home/acct-stu/stu464/data/domestic_sound_events
cd ..;
```
为了避免训练数据占据仓库空间，未包含完整的训练数据，在运行前需要完成特征提取，但只需要运行一次
2. 修改脚本run.sh：以CRNN+enlarge(2)+SpecAugment为例
```bash
python run.py train_evaluate configs/baseline_crnn_specaug/baseline_enlarge_2.yaml data/eval/feature.csv data/eval/label.csv
python run.py train_evaluate configs/baseline_crnn_specaug/baseline_enlarge_2.yaml data/eval/feature.csv data/eval/label.csv
python run.py train_evaluate configs/baseline_crnn_specaug/baseline_enlarge_2.yaml data/eval/feature.csv data/eval/label.csv
python run.py train_evaluate configs/baseline_crnn_specaug/baseline_enlarge_2.yaml data/eval/feature.csv data/eval/label.csv
python run.py train_evaluate configs/baseline_crnn_specaug/baseline_enlarge_2.yaml data/eval/feature.csv data/eval/label.csv
python run.py train_evaluate configs/baseline_crnn_specaug/baseline_enlarge_2.yaml data/eval/feature.csv data/eval/label.csv
python run.py train_evaluate configs/baseline_crnn_specaug/baseline_enlarge_2.yaml data/eval/feature.csv data/eval/label.csv
python run.py train_evaluate configs/baseline_crnn_specaug/baseline_enlarge_2.yaml data/eval/feature.csv data/eval/label.csv
python run.py train_evaluate configs/baseline_crnn_specaug/baseline_enlarge_2.yaml data/eval/feature.csv data/eval/label.csv
python run.py train_evaluate configs/baseline_crnn_specaug/baseline_enlarge_2.yaml data/eval/feature.csv data/eval/label.csv
```
由于训练和测试具有一定的偶然性和随机性，我们采取十次实验并以评测指标的最大值（Maxima）、最小值（Minima）、中间八次结果的平均值（Median Average）进行展示。

你也可以自定义训练参数，其中支持的数据增强为：
```bash
augment:
    mode:
    - spec_aug                  #SpecAugment
    - time_shifting             #Time Shifting (utterance) : usually harmful
    - time_shifting_batch       #Time Shifting (batch)
    - mixup                     #mixup
```

3. 训练、测试:
```bash
srun -N 1 -n 1 -p a100 --gres=gpu:1 --pty /bin/bash
./run.sh
```

注: evaluate.py 用于计算指标，预测结果 `prediction.csv` 写成这样的形式 (分隔符为 `\t`):
```
filename        event_label     onset   offset
Y09RRavdW3C0_30.000_40.000.wav  Speech  0.000   1.000
YIZ_zfkNcxRQ_61.000_71.000.wav  Blender 8.000   9.000
......
```
调用方法：
```bash
python evaluate.py --prediction prediction.csv \
                   --label data/eval/label.csv \
                   --output result.txt
#prediction.csv应当修改为训练输出的正确路径
```

### Final Result (Useful methods only)

Evaluated by Event-based F1-score

1.Model Structure Improvement

|structure|Maxima|Minima|Median Average|
|---|---|---|---|
CRNN | 0.101612 | 0.082934 | 0.092312 | 
CRNN+enlarge(2) | 0.148058 | 0.140121 | 0.144807 |
CRNN+enlarge(4) | 0.150784 | 0.129443 | 0.141605 | 
CRNN + Res | 0.135006 | 0.097071 | 0.115494 | 
CRNN + Res + enlarge(2) | 0.152977 | 0.127373 | 0.138823 | 
CRNN + Res + enlarge(4) | 0.174622 | 0.151641 | 0.159505 | 

2.Data Augmentation

|structure|Maxima|Minima|Median Average|
|---|---|---|---|
CRNN+enlarge(2) | 0.148058 | 0.140121 | 0.144807 |
CRNN+enlarge(2) + SpecAugment | 0.092802 | 0.077018| 0.081874 |
CRNN+enlarge(2) + time shifting(batch) | 0.151220 | 0.142876| 0.146453 |
CRNN+enlarge(2) + Mixup | 0.148160 | 0.126997| 0.129065 |
CRNN + Res + enlarge(4) | 0.174622 | 0.151641 | 0.159505 | 
CRNN + Res + enlarge(4) + SpecAugment| 0.190278 | 0.152106 | 0.165891 |
CRNN + Res + enlarge(4) + time shifting(batch)| 0.169588 | 0.141582 | 0.157620 |
CRNN + Res + enlarge(4) + Mixup| 0.176428 | 0.152736 | 0.160516 | 