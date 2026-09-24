## Part 1

The MLP has 784 inputs, hidden layers of 256 and 128 neurons, and 10 outputs, for 235,146 parameters. I used cross-entropy on the logits, then trained for 15 epochs with Adam (`lr=1e-3`, batch size 64, no weight decay).

The last epoch gave 94.15% train, 89.06% validation and 88.55% test accuracy. The baseline gap plot ends at 5.10 points. Validation loss rose from 0.317 at epoch 7 to 0.373 at epoch 15 while train loss fell, so it overfits.

## Experiment 1

| Hidden layers | Train | Validation | Gap | Test |
|---|---:|---:|---:|---:|
| [4] | 81.94% | 81.67% | 0.28 | 80.79% |
| [64] | 92.26% | 88.28% | 3.99 | 87.99% |
| [256, 128] | 94.15% | 89.06% | 5.10 | 88.55% |
| [512, 256] | 94.61% | 89.41% | 5.20 | 88.58% |

Gaps are in percentage points. The [4] model underfits even with its small gap. The gap plot shows larger gaps as model size increases. In the validation accuracy plot, [512, 256] gains only 0.35 point over [256, 128]. I kept [256, 128] for the next tests because the larger model has more than twice as many parameters.

## Experiment 2A

With SGD, `lr=1e-5` gave 16.93% validation and 17.57% test accuracy. `1e-3` gave 82.03% and 80.64%; `0.1` gave 89.40% and 88.89%. The learning rate plot shows almost no progress at `1e-5`. `1e-3` learns much more slowly than `0.1`.

## Experiment 2B

The best tested SGD rate was `0.1` (89.40% validation, epoch 15). For Adam it was `1e-3` (89.38%, epoch 12). In the optimiser plot, Adam reached 88% at epoch 3, SGD at epoch 5. At `lr=1e-3`, Adam ended at 89.06% and SGD at 82.03%. Adam adapts its updates per parameter, but `lr=0.1` still made it collapse to about 10%.

## Experiment 3

| Setting | Train | Validation | Gap | Test | Best epoch |
|---|---:|---:|---:|---:|---:|
| None | 95.18% | 89.27% | 5.91 | 88.55% | 12 |
| Dropout 0.3 | 90.61% | 88.97% | 1.64 | 88.58% | 17 |
| Weight decay 1e-3 | 90.72% | 88.73% | 1.99 | 88.25% | 16 |
| Dropout with batch norm | 91.51% | 89.38% | 2.12 | 89.31% | 14 |

Gaps are in percentage points. Dropout switches off activations during training; weight decay limits large weights. In the regularisation gap plot, dropout has the smallest gap. Dropout with batch norm has the best final validation accuracy and peaks at 89.64%. Dropout and weight decay peak later than the run without regularisation. They fit the training set less, but validation accuracy changes little.
