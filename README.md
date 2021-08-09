# MentorNet
pytorch version

Related paper:
**<a href="https://arxiv.org/abs/1712.05055">MentorNet: Learning Data-Driven Curriculum for Very Deep Neural Networks on Corrupted Labels
</a>**
<br>
Lu Jiang, Zhengyuan Zhou, Thomas Leung, Li-Jia Li, Li Fei-Fei
<br>
Presented at [ICML 2018](https://icml.cc/Conferences/2018)

Related code:
[MentorNet(google)](https://github.com/google/mentornet)


## usage (how to train mentornet (NOT with StudentNet))

- first you need to train your student model on a noisy dataset which you have a corresponding clean version.
- store the loss, epoch and label in a csv file. the format is as below:
  ```
  'id' 'epoch' 'noisy label' 'clean label' 'loss on the noisy label'
  ...
  ```
  there is a sample csv file `fake_data.csv`
- if you want to preprocess the csv file, use these command:
  ```
  python main.py --process_data=true --raw_csv="\path" --data_path="save\path\to"
  ```

  if you want to train mentornet on the dataset, use these:
  ```
  python main.py --process_data=false --processed_path="\path\to\blah_percentile_40" --epoch=10 --device="cpu" --batch_size=32 --show_progress_bar=false
  ```


UPDATE:
- 8.2.2020: add mentornet_nn class, it works but I am not sure if it is correct.
- 8.3.2020: add MentorNet class for training MentorNet_nn with StudentNet, it can run successfully on cuda. Same, I am not sure if it is correct.
- 8.4.2020: add dataset, dataloader and data_generator. the origin tf version didn't use much 'tf' in this part, so I just copy that here.
- 8.7.2020: add MentorNet trainer class, HAVEN'T TESTED.
- 8.8.2020: MentorNet_nn can be trained using trainer in train.py. The training loss is decreasing, so I guess it works to some extend.
