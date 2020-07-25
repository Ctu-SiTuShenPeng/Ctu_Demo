import paddle.fluid as fluid
import numpy as np
import paddle, os, shutil, cv2, time

class Ctu_Classification:
    def CreateDataList(self,data_root_path,class_file):
        self.ClassIndex = os.listdir(data_root_path)
        for each_Num in range(len(self.ClassIndex)):
            for img_path in os.listdir(os.path.join(data_root_path,self.ClassIndex[each_Num])):
                self.DataList.append(os.path.join(data_root_path,self.ClassIndex[each_Num],img_path) + ",%d" % each_Num)
        with open(class_file,'w') as file:
            for each in self.ClassIndex:
                file.write(each + "\n")

    def image_reader(self):
        np.random.shuffle(self.DataList)
        def reader():
            for line in self.DataList:
                img_path, label = line.split(',')
                img = cv2.imread(img_path)
                img = cv2.resize(img, (224, 224))
                img = img.transpose((2, 0, 1))
                yield img, int(label)
        return reader

    def __init__(self, UseGpu=-1):
        if UseGpu != -1:
            self.place = fluid.CUDAPlace(UseGpu)
        else:
            self.place = fluid.CPUPlace()
        self.DataList = []
        self.ClassIndex=[]
        self.model = None

    def InitModel(self, ImageDir,class_json="./ClassIndex.json",lr=0.001):
        def MySeftNet(input,classnum):
            conv1 = fluid.layers.conv2d(input=input, num_filters=32, filter_size=3, stride=1, padding="SAME", act="relu",param_attr=fluid.param_attr.ParamAttr(initializer=paddle.fluid.initializer.MSRA()))
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
            out = fluid.layers.fc(input=drop1, size=classnum, act="softmax")
            return out
        self.CreateDataList(ImageDir, class_json)
        self.image = fluid.layers.data(name='image', shape=[3, 224, 224],  dtype='float32')
        self.label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        self.model = MySeftNet(self.image, len(self.ClassIndex))
        avg_cost = fluid.layers.mean(fluid.layers.cross_entropy(input=self.model, label=self.label))
        acc = fluid.layers.accuracy(input=self.model, label=self.label)
        fluid.optimizer.AdamOptimizer(learning_rate=lr, regularization=fluid.regularizer.L2DecayRegularizer(1e-4)).minimize(avg_cost)
        self.train_fetch_list = [avg_cost, acc]

    def train(self, TrainNum=500,  batch_size=8,SaveModel="./BestModel"):
        train_reader = paddle.batch(reader=self.image_reader(), batch_size=batch_size)
        test_reader = paddle.batch(reader=self.image_reader(), batch_size=batch_size)
        feeder = fluid.DataFeeder(place=self.place, feed_list=[self.image, self.label])
        exe = fluid.Executor(self.place)
        exe.run(fluid.default_startup_program())
        for pass_id in range(TrainNum):
            for batch_id, data in enumerate(train_reader()):
                start_time = time.time()
                train_cost, train_acc = exe.run(program=fluid.default_main_program(), feed=feeder.feed(data), fetch_list=self.train_fetch_list)
                print('迭代[%d/%d:%d]-> 训练Loss:%0.5f, 准确率:%0.5f -> 耗时:%0.5f 秒' % (pass_id+1, TrainNum, batch_id+1, train_cost[0], train_acc[0],time.time()-start_time))
            start_time = time.time()
            test_accs = []
            test_costs = []
            for batch_id, data in enumerate(test_reader()):
                test_cost, test_acc = exe.run(program=fluid.default_main_program().clone(for_test=True), feed=feeder.feed(data), fetch_list=self.train_fetch_list)
                test_accs.append(test_acc[0])
                test_costs.append(test_cost[0])
            test_cost = (sum(test_costs) / len(test_costs))
            test_acc = (sum(test_accs) / len(test_accs))
            print('测试:%d, Loss:%0.5f, Acc:%0.5f -> time:%0.5f 秒' % (pass_id+1, test_cost, test_acc,time.time()-start_time))

            shutil.rmtree(SaveModel, ignore_errors=True)
            fluid.io.save_inference_model(SaveModel, feeded_var_names=[self.image.name], target_vars=[self.model], executor=exe)

    def load_Model(self,ModelFile,ClassJson):
        self.exe = fluid.Executor(self.place)
        [self.infer_program, self.feeded_var_names, self.target_var] = fluid.io.load_inference_model(dirname=ModelFile,  executor=self.exe)
        file = open(ClassJson, 'r', encoding='utf-8')
        self.ClassIndex = [each.strip() for each in file.readlines()]
        file.close()

    def predict(self, img):
        start_time = time.time()
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        result = self.exe.run(program=self.infer_program, feed={self.feeded_var_names[0]: img}, fetch_list=self.target_var)
        result_value={
            "title":self.ClassIndex[np.argsort(result)[0][0][-1]],
            "Confidence":result[0][0][np.argsort(result)[0][0][-1]],
            "time":(time.time()-start_time)*1000
        }
        return result_value

if __name__ == '__main__':
    ctu = Ctu_Classification(UseGpu=0)
    ctu.InitModel(ImageDir="./DataSet/Image", class_json="./classes1.txt")
    ctu.train(TrainNum=200, batch_size=8, SaveModel="./BestModel")
    del ctu

    # ctu1 = Ctu_Classification(UseGpu=0)
    # ctu1.load_Model("./BestModel1", "./classes1.txt")
    # cv2.namedWindow("origin", 0)
    # cv2.resizeWindow("origin", 640, 480)
    #
    # for root, dirs, files in os.walk("../DataSet_Classification/Image1"):
    #     for f in files:
    #         img_cv = cv2.imread(os.path.join(root, f))
    #         res = ctu1.predict(img_cv)
    #         print(res)
    #         cv2.imshow("origin", img_cv)
    #         cv2.waitKey()
