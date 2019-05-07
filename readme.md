## 比赛结果验证总结

官方拖了半个月的C榜测试集不公布，改成新人赛在线验证
（包含比赛A/B/C阶段的所有训练输入数据集）：

[【新人赛】城市计算AI挑战赛](https://tianchi.aliyun.com/competition/entrance/231712/rankingList)

##### 验证结果如下：

在线训练模型复现的结果，A榜和B的结果都比线下训练的差了一些，
如果想要好一点的结果，则将训练epoch调大，
通过tensorboard观察验证集的曲线决定什么时候停止。
另外，验证了全数据训练的mode_xxxx_tol模型（比赛时不够提交次数验证），
存在过拟合现象，不宜使用。

预测出来的结果乘个衰减（DECAY）能明显的降低MAE，尤其是C榜，所有的尝试结果如下：

##### testA

所有模型融合的结果：

| DECAY | 1 |
| :--: | :--: |
| MAE | 12.28 |

丢弃5个mode_xxxx_tol模型的结果：

| DECAY | 1 | 0.98 | 0.95 |
| :--: | :--: | :--: | :--: |
| MAE | 12.05 | __11.88__ | 12.12|

##### testB

所有模型融合的结果：

| DECAY | 1 |
| :--: | :--: |
| MAE | 11.80 |

丢弃5个mode_xxxx_tol模型的结果：

| DECAY | 1 |
| :--: | :--: |
| MAE | 11.89|

##### testC

所有模型融合的结果：

| DECAY | 1 |
| :--: | :--: |
| MAE | 17.13 |

丢弃5个mode_xxxx_tol模型的结果：

| DECAY | 1 | 0.95 | 0.90 | 0.85 |
| :--: | :--: | :--: | :--: | :--: |
| MAE | 16.51 | 13.85 | 12.26 | __12.00__ |

no_tol: 1->16.51 0.95->13.85 0.90->12.26 0.85->12.00

至于衰减取值更细致些会不会有更好的结果就懒得试了，
因为这个衰减的值是无法从1到25号的训练数据中学习到的，不是一样的变化模式，
而要通过主观臆测春运带来的影响多大来确定，或者结合往年春运的一些外部数据来猜测，
没啥意思。

注：此模型的做法是只用给的2019-01-01 -> 2019-01-25的数据训练模型，
目标是给任意一天的数据作为输入，然后预测第二天的人流量
（以为比赛可能会用2到4月的某一天作为测试数据，或者是训练数据日期之前的某一天，
或者某一个节假日，所以开始还有个holiday字段，把1月1号元旦也用来训练了，
到了B/C榜之后，才知道自己想多了）。
由于只有不到一个月的训练数据，也不包含众多节假日，所以应该没有很好泛化能力。
此类模型比较适合有较长时间的数据用于训练的场景。

## Steps recommended

bash train.sh A/B/C  # train only once, ABC are the same

bash test_no_tol.sh A/B/C

## Train

bash train.sh B  # for test B, models will be saved in ../train/models/

bash train.sh C  # for test C, but it is no need to train it again


## Predict

bash test.sh B  # for test B

bash test.sh C  # for test C


## References
Lv, Y., Duan, Y., Kang, W., Li, Z., & Wang, F. Y. (2015). 
Traffic Flow Prediction with Big Data: A Deep Learning Approach. 
IEEE Transactions on Intelligent Transportation Systems, 16(2), 
865–873. https://doi.org/10.1109/TITS.2014.2345663

Liu, L., & Chen, R. (2017). 
A novel passenger flow prediction model using deep learning methods. 
Transportation Research Part C, 84, 
74–91. https://doi.org/10.1016/j.trc.2017.08.001


## Feature
block: range(0, 60 / 10 * 24)

stationID: range(0, 81)

status: range(0, 2)

payType: range(0, 4)

week: range(0, 7)

holiday: range(0, 2)

flow: number_in or number_out

yesterday: the flow of yesterday, has many differences to daily meanings

lastWeek: just the flow of (current day - 7)

averWeek: the mean of days before current day with the same week feature value

averWeekOutIn: similar to averWeek, but ignore pay type

yesterdayNormal: in daily meanings

Can see the rules of generating historical features in generateHistoricData


## Feature Selection

The way to do feature selection is directly eliminate a feature, 
and then train a new model, to see if the result is getting better or worse. 

1. pay_type: try to eliminate this feature, directly predicts the flow ignore
   pay type, get a worse result.
   
2. averWeekOutIn: try to not use this feature, get a better result.

3. lastWeek, averWeek: the results in training and validation set are very well,
   but the result in test A will be over-fitting.
   
4. yesterdayNormal: because the flows between workday and weekend are very different,
   so use the yesterday's data in daily meanings is not suitable.
   
5. holiday: only one day in the training data only is holiday, 
   and it has obvious different pattern. 
   So the holiday is eliminated finally.


## Validation Set

The course of change of the training set and the verification set is as follows: 

Firstly, just selects the last two days as validation set by feeling.

Secondly, the test set is to predict the flow of one day, 
so selects the last day as validation set. 
(it is no significant difference to select the validation set 
that has the same day of the week with test set)

Finally, to let the training set be balance, the number each day 
of the week is set to three, and due to the significant difference of holiday,
the first day is eliminated, and the second day do not have normal yesterday,
is eliminated too. Then the validation set is also the last two days.


## Models

In the first time, I try to use SAE showed in the References, but it is 
hard to train and spend much more time in training, and the time of 
this competition is short, so I finally choose fully connected 
networks, which I thought can have similarity performance to SAE.

The construction of features in the References is not suitable to this 
competition, due to the data sets has very large difference. 
The training data set of this competitions has very strong relation 
between weeks, but the test data don't have such a good pattern, so it 
will lead to over-fitting. 

Problems: there are many mistakes made by myself, many models' 
parameters are wrongly set, such as not set drop out, wrong n_outputs, 
wrong features to train, and wrong validate set. But due to the need of 
recurring the result of test B, these wrong settings are being kept.

All the models used in test B are base on the fully connected network 
with different parameters. Five models is selected by the mae of test A, 
the other five models should have the same parameters with the before 
five models (they are not exactly the same because of my mistakes), 
but they are trained by all training data (including validation data),
The differences of each model will be list below:

#### model_1228
training data: 2019-01-03 -> 2019-01-23

validate data: 2019-01-24 -> 2019-01-25

weights = [1, 0, 0]  # the weight of three different loss functions

lr = 0.0001  # learning rate

constants.FEAT_LEN = [144, 81, 2, 4, 7, 1]  # the length of each feature being one-hot encoding

constants.HEADER_FEAT = ['block', 'stationID', 'status', 'payType', 'weekDay', 'flow'] # the features used in this model

constants.HEADER_HIS = ['yesterday']  # the features generate by historical data

n_inputs = sum(constants.FEAT_LEN)  # the real input of nn is n_inputs * n_outputs 

n_outputs = 3  # the output of nn

n_mlp = [400, 400, 400]  # the size of each hidden layer

##### model_1228_tol
training data: 2019-01-03 -> 2019-01-25

validate data: not important, training stopped according to the training epoch of model_1228

weights = [1, 0, 0]

lr = 0.0001

constants.FEAT_LEN = [144, 81, 2, 4, 7, 1]

constants.HEADER_FEAT = ['block', 'stationID', 'status', 'payType', 'weekDay', 'flow']

constants.HEADER_HIS = ['yesterday']

n_inputs = sum(constants.FEAT_LEN)

n_outputs = 4

n_mlp = [400, 400, 400]

##### model_1250
training data: 2019-01-03 -> 2019-01-23

validate data: 2019-01-24 -> 2019-01-25

weights = [10000, 100, 1]

lr = 0.0001

constants.FEAT_LEN = [144, 81, 2, 4, 7, 1]

constants.HEADER_FEAT = ['block', 'stationID', 'status', 'payType', 'weekDay', 'flow']

constants.HEADER_HIS = ['yesterday']

n_inputs = sum(constants.FEAT_LEN)

n_outputs = 3

n_mlp = [400, 400, 400]

##### model_1250_tol
training data: 2019-01-03 -> 2019-01-25

validate data: not important, training stopped according to the training epoch of model_1250

weights = [10000, 100, 1]

lr = 0.0001

constants.FEAT_LEN = [144, 81, 2, 4, 7, 1]

constants.HEADER_FEAT = ['block', 'stationID', 'status', 'payType', 'weekDay', 'flow']

constants.HEADER_HIS = ['yesterday']

n_inputs = sum(constants.FEAT_LEN)

n_outputs = 4

n_mlp = [400, 400, 400]


##### model_1266
training data: 2019-01-03 -> 2019-01-23

validate data: 2019-01-24 -> 2019-01-25

weights = [1, 0, 0]

lr = 0.0001

constants.FEAT_LEN = [144, 81, 2, 4, 7, 1]

constants.HEADER_FEAT = ['block', 'stationID', 'status', 'payType', 'weekDay', 'flow']

constants.HEADER_HIS = ['yesterday']

n_inputs = sum(constants.FEAT_LEN)

n_outputs = 4

n_mlp = [500, 500]

##### model_1266_tol
training data: 2019-01-03 -> 2019-01-25

validate data: not important, training stopped according to the training epoch of model_1266


weights = [1, 0, 0]

lr = 0.0001

constants.FEAT_LEN = [144, 81, 2, 4, 7, 1]

constants.HEADER_FEAT = ['block', 'stationID', 'status', 'payType', 'weekDay', 'flow']

constants.HEADER_HIS = ['yesterday']

n_inputs = sum(constants.FEAT_LEN)

n_outputs = 4

n_mlp = [500, 500]

##### model_1294
training data: 2019-01-01 -> 2019-01-24

validate data: 2019-01-25

weights = [1, 0, 0]

lr = 0.001
constants.FEAT_LEN = [144, 81, 2, 4, 7, 2, 1, 1, 1]

constants.HEADER_FEAT = ['block', 'stationID', 'status', 'payType', 'weekDay',  'holiday', 'flow']

constants.HEADER_HIS = ['yesterdayNormal', 'lastWeek', 'averWeek']

n_inputs = sum(constants.FEAT_LEN)

n_outputs = 4

n_mlp = [400, 400, 400, 400, 400, 400]

##### model_1294_tol
training data: 2019-01-03 -> 2019-01-25

validate data: not important, training stopped according to the training epoch of model_1294

weights = [1, 0, 0]

lr = 0.001

constants.FEAT_LEN = [144, 81, 2, 4, 7, 2, 1, 1, 1]

constants.HEADER_FEAT = ['block', 'stationID', 'status', 'payType', 'weekDay',  'holiday', 'flow']

constants.HEADER_HIS = ['yesterdayNormal', 'lastWeek', 'averWeek']

n_inputs = sum(constants.FEAT_LEN)

n_outputs = 4

n_mlp = [400, 400, 400, 400, 400, 400]

##### model_1305
training data: 2019-01-01 -> 2019-01-22

validate data: 2019-01-23 -> 2019-01-25

weights = [1, 0, 0]

lr = 0.0001

constants.FEAT_LEN = [144, 81, 2, 4, 7, 2, 1, 1, 1]

constants.HEADER_FEAT = ['block', 'stationID', 'status', 'payType', 'weekDay',  'holiday', 'flow']

constants.HEADER_HIS = ['yesterdayNormal', 'lastWeek', 'averWeek']

n_inputs = sum(constants.FEAT_LEN)

n_outputs = 4

n_mlp = [400, 400, 400]

##### model_1305_tol
training data: 2019-01-03 -> 2019-01-25

validate data: not important, training stopped according to the training epoch of model_1305

weights = [1, 0, 0]

lr = 0.0001

constants.FEAT_LEN = [144, 81, 2, 4, 7, 2, 1, 1, 1]

constants.HEADER_FEAT = ['block', 'stationID', 'status', 'payType', 'weekDay',  'holiday', 'flow']

constants.HEADER_HIS = ['yesterdayNormal', 'lastWeek', 'averWeek']

n_inputs = sum(constants.FEAT_LEN)

n_outputs = 4

n_mlp = [400, 400, 400]

more detials: look at the source code of one of the models, 
              the other nine are redundant.

##### Blend
get the mean of results predicted by above models


## 比赛存在的问题

<img src="pictures/比赛规则.png" width = "800" height = "400" alt="比赛规则" align=center />

1. <font color=red>临时改规则要求部署训练代码</font>，导致时间紧缺，工作量暴增，
   而官方又不给合理答复，只能通宵改代码，还没时间调试。

2. <font color=red>结果不透明</font>，刚开始只有晋级的会收到邮件，其他人都不知道
   情况如何，问官方给不给C榜结果，回复说不给。
   后面把截图发群里，又改说在讨论（晋级的队伍都决定了，居然还在讨论C榜结果......）。
   后面过了一天，才给了一张B&C榜结果的图片（此处可自行想象）......

3. 4月5号C榜排名揭晓，但<font color=red>C榜数据至今没有按规则所说公布出来</font>。
   无论在官方群询问还是私聊官方工作人员什么时候公布C榜数据，都不给答复。
   因此，至今不知为何大半队伍结果的目标是星辰大海，无法解决模型可能存在的问题。
   
    更新：现在是不公布数据，改成新人赛了，解决了点问题。
   
总的来说，此次参加比赛，是本渣渣第一次真正意义参加比赛，
给自己带来了很多的锻炼，也认识了些大佬，但也遇到了解到了些自己不希望会存在的事情：
不按规则办事，结果不透明，工作人员基本不及时回答解决参赛人员关注的问题，
最后为最终的比赛结果。此次参加比赛只能说体验很差，对天池也感到很失望。
 
<img src="pictures/比赛结果.png" width = "800" height = "300" alt="比赛结果" align=center />
