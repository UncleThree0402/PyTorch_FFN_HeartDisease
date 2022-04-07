# Pytorch FFN Heart Disease Prediction

## Data

Data is from [UCL Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease)

### Data Before Normalized
![]

### Data After Normalized
![box_plot_after_normalize](https://github.com/UncleThree0402/PyTorch_FFN_HeartDisease/blob/master/Photo/box_plot_after_normalize.png)

### Count of label
To check is dataset balanced

![count_labels](https://github.com/UncleThree0402/PyTorch_FFN_HeartDisease/blob/master/Photo/count_labels.png)

## Model

### Net
```python
class HeartNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.input = nn.Linear(13, 20)

        self.ll1 = nn.Linear(20, 30)

        self.d1 = nn.Dropout(0.2)

        self.ll2 = nn.Linear(30, 30)

        self.d2 = nn.Dropout(0.2)

        self.ll3 = nn.Linear(30, 13)

        self.d3 = nn.Dropout(0.2)

        self.output = nn.Linear(13, 1)

    def forward(self, x):
        x = self.input(x)
        x = nn.LeakyReLU()(x)
        x = self.d1(x)
        x = self.ll1(x)
        x = nn.LeakyReLU()(x)
        x = self.d2(x)
        x = self.ll2(x)
        x = nn.LeakyReLU()(x)
        x = self.d3(x)
        x = self.ll3(x)
        x = nn.LeakyReLU()(x)
        x = self.output(x)
        return x
```

### Loss Function
```python
loss_fn = nn.BCEWithLogitsLoss()
```

### Optimizer
```python
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
```
>lr = 0.0001, weight_decay = 0.015

### lr_scheduler
```python
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
```
>step_size = 10, gamma = 0.9

## Train

### Loss
![losses](https://github.com/UncleThree0402/PyTorch_FFN_HeartDisease/blob/master/Photo/losses.png)

### Accuracy
![accuracies](https://github.com/UncleThree0402/PyTorch_FFN_HeartDisease/blob/master/Photo/accuracies.png)

### Learning Rate
![lr_rate](https://github.com/UncleThree0402/PyTorch_FFN_HeartDisease/blob/master/Photo/lr_rate.png)

## Performance
![performance](https://github.com/UncleThree0402/PyTorch_FFN_HeartDisease/blob/master/Photo/performance.png)

### Train
```bash
               precision    recall  f1-score   support

          No       0.83      0.93      0.87       124
         Yes       0.91      0.79      0.84       113

    accuracy                           0.86       237
   macro avg       0.87      0.86      0.86       237
weighted avg       0.87      0.86      0.86       237
```

### Valid
```bash
               precision    recall  f1-score   support

          No       0.94      0.89      0.91        18
         Yes       0.85      0.92      0.88        12

    accuracy                           0.90        30
   macro avg       0.89      0.90      0.90        30
weighted avg       0.90      0.90      0.90        30
```

### Test
```bash
               precision    recall  f1-score   support

          No       0.89      0.94      0.92        18
         Yes       0.91      0.83      0.87        12

    accuracy                           0.90        30
   macro avg       0.90      0.89      0.89        30
weighted avg       0.90      0.90      0.90        30
```

### Confusion matrix

#### Train
![train_conf](https://github.com/UncleThree0402/PyTorch_FFN_HeartDisease/blob/master/Photo/train_cm.png)

#### Valid
![valid_cm](https://github.com/UncleThree0402/PyTorch_FFN_HeartDisease/blob/master/Photo/valid_cm.png)

#### Test
![test_cm](https://github.com/UncleThree0402/PyTorch_FFN_HeartDisease/blob/master/Photo/test_cm.png)

