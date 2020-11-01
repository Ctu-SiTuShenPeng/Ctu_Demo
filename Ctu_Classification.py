import paddle.fluid as fluid
import paddle, os, cv2, time

def image_reader(self):
    def reader():
        for line in self.DataList:
            img_path, label = line.split(' ')
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            img = img.transpose((2, 0, 1))
            yield img, int(label)
    return reader

DataDir='./DataSet/'
className=['']
dataList=[]
for each in range(0,len(className)):
    for img_path in os.listdir(DataDir+className[each]):
        dataList.append(DataDir+className[each] + '/' + img_path + " %d" % each)

place = fluid.CPUPlace()
image = fluid.layers.data(name='image', shape=[3, 224, 224],  dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
conv1 = fluid.layers.conv2d(input=image, num_filters=32, filter_size=3, stride=1, padding="SAME", act="relu",param_attr=fluid.param_attr.ParamAttr(initializer=paddle.fluid.initializer.MSRA()))
conv2 = fluid.layers.conv2d(input=conv1, num_filters=32, filter_size=3, stride=1, padding="SAME", act="relu",param_attr=fluid.param_attr.ParamAttr(initializer=paddle.fluid.initializer.MSRA()))
pool1 = fluid.layers.pool2d(conv2, pool_size=[2, 2], pool_type="max", pool_stride=[2, 2], pool_padding='VALID')
conv3 = fluid.layers.conv2d(input=pool1, num_filters=64, filter_size=3, stride=1, padding="SAME", act="relu",param_attr=fluid.param_attr.ParamAttr(initializer=paddle.fluid.initializer.MSRA()))
conv4 = fluid.layers.conv2d(input=conv3, num_filters=64, filter_size=3, stride=1, padding="SAME", act="relu",param_attr=fluid.param_attr.ParamAttr(initializer=paddle.fluid.initializer.MSRA()))
pool2 = fluid.layers.pool2d(conv4, pool_size=[2, 2], pool_type="max", pool_stride=[2, 2], pool_padding='VALID')
conv3 = fluid.layers.conv2d(input=pool2, num_filters=128, filter_size=3, stride=1, padding="SAME", act="relu")
conv4 = fluid.layers.conv2d(input=conv3, num_filters=128, filter_size=3, stride=1, padding="SAME", act="relu")
pool2 = fluid.layers.pool2d(conv4, pool_size=[2, 2], pool_type="max", pool_stride=[2, 2], pool_padding='VALID')
flatten1 = paddle.fluid.layers.flatten(pool2)
fc1 = fluid.layers.fc(input=flatten1, size=64, act="relu")
drop1 = fluid.layers.dropout(x=fc1, dropout_prob=0.5)
model = fluid.layers.fc(input=drop1, size=len(className), act="softmax")

avg_cost = fluid.layers.mean(fluid.layers.cross_entropy(input=model, label=label))
acc = fluid.layers.accuracy(input=model, label=label)
fluid.optimizer.AdamOptimizer(learning_rate=0.001, regularization=fluid.regularizer.L2DecayRegularizer(1e-4)).minimize(avg_cost)

train_reader = paddle.batch(reader=image_reader(), batch_size=4)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

for pass_id in range(500):
    for batch_id, data in enumerate(train_reader()):
        train_cost, train_acc = exe.run(program=fluid.default_main_program(), feed=feeder.feed(data), fetch_list=[avg_cost, acc])
        print('迭代[%d:%d]-> 训练Loss:%0.5f, 准确率:%0.5f' % (pass_id + 1, batch_id + 1, train_cost[0], train_acc[0]))
    fluid.io.save_inference_model("./ModelData", feeded_var_names=[image.name], target_vars=[model], executor=exe)
