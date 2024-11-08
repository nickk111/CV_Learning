# YOLOv11 | 一文带你深入理解ultralytics最新作品yolov11的创新 | 训练、推理、验证、导出 （附网络结构图）

**目录**

**一、本文介绍**

**二、YOLOv11和YOLOv8对比**

**三、YOLOv11的网络结构解析**

**四、YOLOv11下载、环境安装、数据集获取**

**五、模型训练**

**5.1 训练的三种方式**

**5.1.1 方式一**

**5.1.2 方式二**

**5.1.3 方式三 （推荐，避免keyError错误.）**

**六、模型验证/测试**

**七、模型推理**

**八、模型输出**

* * *

## 一、本文介绍

ultralytics发布了最新的作品YOLOv11，这一次YOLOv11的变化相对于ultralytics公司的上一代作品YOLOv8变化不是很大的（YOLOv9、YOLOv10均不是ultralytics公司作品），其中改变的位置涉及到C2f变为C3K2，在SPPF后面加了一层类似于注意力机制的C2PSA，还有一个变化大家从yaml文件是看不出来的就是它的检测头内部替换了两个DWConv，以及模型的深度和宽度参数进行了大幅度调整，但是在损失函数方面就没有变化还是采用的CIoU作为边界框回归损失，下面带大家深入理解一下ultralytics最新作品YOLOv11的创新点。

**下图为最近的YOLO系列发布时间线！**

![](https://i-blog.csdnimg.cn/direct/af970d3eee954b988a2252ae7d822692.png)

* * *

## 二、YOLOv11和YOLOv8对比

在YOLOYOLOv5，YOLOv8，和YOLOv11是ultralytics公司作品（ultralytics出品必属精品），下面用一张图片从yaml文件来带大家对比一下YOLOv8和YOLOv11的区别，配置文件变得内容比较少大家可以看一卡，左侧为YOLOv8右侧为YOLOv11，不同的点我用黑线标注了出来。

![](https://i-blog.csdnimg.cn/direct/77ed65ad1abb49febddd085365199d72.png)

* * *

## 三、YOLOv11的网络结构解析 

下面的图片为YOLOv11的网络结构图。

![](https://i-blog.csdnimg.cn/direct/1b8c2f927496412ea0fb8e56c4c4825d.png)

**其中主要创新点可以总结如下- > **

* * *

1\. 提出C3k2机制，其中C3k2有参数为c3k，其中在网络的浅层c3k设置为False（下图中可以看到c3k2第二个参数被设置为False，就是对应的c3k参数）。

![](https://i-blog.csdnimg.cn/direct/32a1df28bcc2439bb438c3079eb08cb4.png)

此时所谓的C3k2就相当于YOLOv8中的C2f，其网络结构为一致的，其中的C3k机制的网络结构图如下图所示 **（为什么叫C3k2，我个人理解是因为C3k的调用时C3k其中的参数N固定设置为2的原因，个人理解不一定对** ）。 

![](https://i-blog.csdnimg.cn/direct/51e67140fed44ba9bfce8d0f6a0658e5.png)

* * *

2\. 第二个创新点是提出C2PSA机制，这是一个C2（C2f的前身）机制内部嵌入了一个多头注意力机制，在这个过程中我还发现作者尝试了C2fPSA机制但是估计效果不如C2PSA，有的时候机制有没有效果理论上真的很难解释通，下图为C2PSA机制的原理图，仔细观察把Attention哪里去掉则C2PSA机制就变为了C2所以我上面说C2PSA就是C2里面嵌入了一个PSA机制。

![](https://i-blog.csdnimg.cn/direct/78df93ba25404dbba0fb7ca0ee0ab2ae.png)

* * *

3\. 第三个创新点可以说是原先的解耦头中的分类检测头增加了两个DWConv，具体的对比大家可以看下面两个图下面的是YOLOv11的解耦头，上面的是YOLOv8的解耦头.

![](https://i-blog.csdnimg.cn/direct/d0e8569fe33c453d816165e73f74826e.png)

我们上面看到了在分类检测头中YOLOv11插入了两个DWConv这样的做法可以大幅度减少参数量和计算量（原先两个普通的Conv大家要注意到卷积和是由3变为了1的，这是形成了两个深度可分离Conv），大家可能不太理解为什么加入了两个DWConv还能够减少计算量，以及什么是深度可分离Conv，下面我来解释一下。

> **`DWConv` 代表 Depthwise Convolution（深度卷积）**，是一种在卷积神经网络中常用的高效卷积操作。它主要用于减少计算复杂度和参数量，尤其在移动端或轻量化网络（如 MobileNet）中十分常见。
> 
> **1\. 标准卷积的计算过程**
> 
> 在标准卷积操作中，对于一个输入张量（通常是一个多通道的特征图），卷积核的尺寸是 `(h, w, C_in)`，其中 `h` 和 `w` 是卷积核的空间尺寸，`C_in` 是输入通道的数量。而卷积核与输入张量做的是完整的卷积运算，每个输出通道都与所有输入通道相连并参与卷积操作，导致计算量比较大。
> 
> 标准卷积的计算过程是这样的：
> 
>   * 每个输出通道是所有输入通道的组合（加权求和），卷积核在每个位置都会计算与所有输入通道的点积。
>   * 假设有 `C_in` 个输入通道和 `C_out` 个输出通道，那么卷积核的总参数量是 `C_in * C_out * h * w`。
> 

> 
> 2\. **Depthwise Convolution（DWConv）**
> 
> 与标准卷积不同， **深度卷积** 将输入的每个通道单独处理，即 **每个通道都有自己的卷积核进行卷积** ，不与其他通道进行交互。它可以被看作是标准卷积的一部分，专注于空间维度上的卷积运算。
> 
> **深度卷积的计算过程：**
> 
>   * 假设输入张量有 `C_in` 个通道，每个通道会使用一个 `h × w` 的卷积核进行卷积操作。这个过程称为“深度卷积”，因为每个通道独立进行卷积运算。
>   * 输出的通道数与输入通道数一致，每个输出通道只和对应的输入通道进行卷积，没有跨通道的组合。
>   * 参数量和计算量相比标准卷积大大减少，卷积核的参数量是 `C_in * h * w`。
> 

> 
> **深度卷积的优点：**
> 
>   1. **计算效率高** ：相对于标准卷积，深度卷积显著减少了计算量。它只处理空间维度上的卷积，不再处理通道间的卷积。
>   2.  **参数量减少** ：由于每个卷积核只对单个通道进行卷积，参数量大幅减少。例如，标准卷积的参数量为 `C_in * C_out * h * w`，而深度卷积的参数量为 `C_in * h * w`。
>   3.  **结合点卷积可提升效果** ：为了弥补深度卷积缺乏跨通道信息整合的问题，通常深度卷积后会配合 `1x1` 的点卷积（Pointwise Convolution）使用，通过 `1x1` 的卷积核整合跨通道的信息。这种组合被称为 **深度可分离卷积** （Depthwise Separable Convolution） | **这也是我们本文YOLOv11中的做法** 。
> 

> 
> 3\. **深度卷积与标准卷积的区别**
> 
> 操作类型| 卷积核大小| 输入通道数| 输出通道数| 参数量  
> ---|---|---|---|---  
> 标准卷积| `h × w`| `C_in`| `C_out`| `C_in * C_out * h * w`  
> 深度卷积（DWConv）| `h × w`| `C_in`| `C_in`| `C_in * h * w`  
>   
> 可以看出，深度卷积在相同的卷积核大小下，参数量减少了约 `C_out` 倍 （细心的人可以发现用最新版本的ultralytics仓库运行YOLOv8参数量相比于之前的YOLOv8以及大幅度减少了这就是因为检测头改了的原因但是名字还是Detect，所以如果你想继续用YOLOv8发表论文做实验那么不要更新最近的ultralytics仓库）。
> 
> **4\. 深度可分离卷积 (Depthwise Separable Convolution)**
> 
> 深度卷积常与 `1x1` 的点卷积配合使用，这称为深度可分离卷积。其过程如下：
> 
>   1. 先对输入张量进行深度卷积，对每个通道独立进行空间卷积。
>   2. 然后通过 `1x1` 点卷积，对通道维度进行混合，整合不同通道的信息。
> 

> 
> 这样既可以保证计算量的减少，又可以保持跨通道的信息流动。
> 
> 5\. **总结**
> 
> `DWConv` 是一种高效的卷积方式，通过单独处理每个通道来减少计算量，结合 `1x1` 的点卷积，形成深度可分离卷积，可以在保持网络性能的同时极大地减少模型的计算复杂度和参数量。

**看到这里大家应该明白了为什么加入了两个DWConv还能减少参数量以及YOLOv11的检测头创新点在哪里。**

* * *

4.YOLOv11和YOLOv8还有一个不同的点就是其各个版本的模型（N - S - M- L - X）网络深度和宽度变了 ![](https://i-blog.csdnimg.cn/direct/a5fffabb154543cb9e544a24eba20aa3.png)

可以看到在深度（depth）和宽度 （width）两个地方YOLOv8和YOLOv11是基本上完全不同了，这里我理解这么做的含义就是模型网络变小了，所以需要加深一些模型的放缩倍数来弥补模型之前丧失的能力从而来达到一个平衡。

> **本章总结：** YOLOv11的改进点其实并不多更多的都是一些小的结构上的创新，相对于之前的YOLOv5到YOLOv8的创新，其实YOLOv11的创新点不算多，但是其是ultralytics公司的出品，同时ultralytics仓库的使用量是非常多的（不像YOLOv9和YOLOv10）所以在未来的很长一段时间内其实YOLO系列估计不会再更新了，YOLOv11作为最新的SOTA肯定是十分适合大家来发表论文和创新的。
> 
> **最后强调：** 本文只是对YOLOv11的创新部分进行了部分解析，其余部分其实和YOLOv8保持一致大家有需要的可以自行查阅其它资料，同时有解析不对的地方欢迎大家评论区指出和讨论。

* * *

## 四、YOLOv11下载、环境安装、数据集获取

YOLOv11的下载大家可以通过点击下面的链接进行下载->

> **官方下载地址：**[YOLOv11官方Github下载地址点击此处即可跳转.](https://github.com/ultralytics/ultralytics/tree/main?tab=readme-ov-file "YOLOv11官方Github下载地址点击此处即可跳转.")

点进去之后按照如下图所示的操作即可下载ultralytics仓库到本地.

![](https://i-blog.csdnimg.cn/direct/f893014b3997409f9c6d4916d6b3ce9b.png)

下载到本地之后大家解压缩利用自己的IDEA打开即可了，环境搭建大家可以看我另一篇文章，这里由于篇幅原因就不多介绍了，如果你自己有环境了跳过此步即可.

> **环境安装下载** ：[环境安装教程点击此处即可跳转.](https://blog.csdn.net/java1314777/article/details/142666599?sharetype=blogdetail&sharerId=142666599&sharerefer=PC&sharesource=java1314777&spm=1011.2480.3001.8118 "环境安装教程点击此处即可跳转.")

* * *

数据集获取方法可以看以下文章内容，利用roboflow获取大量数据集（1000w+数据集任你挑选）

> **数据集下载教程：**[roboflow一键导出Voc、COCO、Yolo、Csv、yolo等格式数据集教程](https://blog.csdn.net/java1314777/article/details/142666615?sharetype=blogdetail&sharerId=142666615&sharerefer=PC&sharesource=java1314777&spm=1011.2480.3001.8118 "roboflow一键导出Voc、COCO、Yolo、Csv、yolo等格式数据集教程")

* * *

## 五、模型训练

上面给大家讲完了网络的创新下面给大家讲一下YOLOv11如何进行训练预测验证等操作。

**我门打开ultralytics/cfg/default.yaml文件可以配置模型的参数，** 在其中和模型训练有关的参数及其解释如下:

| 参数名| 输入类型| 参数解释  
---|---|---|---  
0| task| str| YOLO模型的任务选择，选择你是要进行检测、分类等操作  
1| mode| str| YOLO模式的选择，选择要进行训练、推理、输出、验证等操作  
2| model| str/optional| 模型的文件，可以是官方的预训练模型，也可以是训练自己模型的yaml文件  
3| data| str/optional| 模型的地址，可以是文件的地址，也可以是配置好地址的yaml文件  
4| epochs| int| 训练的轮次，将你的数据输入到模型里进行训练的次数  
5| patience| int| 早停机制，当你的模型精度没有改进了就提前停止训练  
6| batch| int| 我们输入的数据集会分解为多个子集，一次向模型里输入多少个子集  
7| imgsz| int/list| 输入的图片的大小，可以是整数就代表图片尺寸为int*int，或者list分别代表宽和高[w，h]  
8| save| bool| 是否保存模型以及预测结果  
9| save_period| int| 在训练过程中多少次保存一次模型文件,就是生成的pt文件  
10| cache| bool| `参数cache`用于控制是否启用缓存机制。  
11| device| int/str/list/optional| GPU设备的选择：cuda device=0 or device=0,1,2,3 or device=cpu  
12| workers| int| 工作的线程，Windows系统一定要设置为0否则很可能会引起线程报错  
13| name| str/optional| 模型保存的名字，结果会保存到'project/name' 目录下  
14| exist_ok| bool| 如果模型存在的时候是否进行覆盖操作  
15| prepetrained| 

bool

| 参数pretrained用于控制是否使用预训练模型。  
16| optimizer| str| 优化器的选择choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]  
17| verbose| bool| 用于控制在执行过程中是否输出详细的信息和日志。  
18| seed| int| 随机数种子，模型中涉及到随机的时候，根据随机数种子进行生成  
19| deterministic| bool| 用于控制是否启用确定性模式，在确定性模式下，算法的执行将变得可重复，即相同的输入将产生相同的输出  
20| single_cls| bool| 是否是单标签训练  
21| rect| bool| 当 `rect` 设置为 `True` 时，表示启用矩形训练或验证。矩形训练或验证是一种数据处理技术，其中在训练或验证过程中，输入数据会被调整为具有相同宽高比的矩形形状。  
22| 

cos_lr

| bool| 控制是否使用余弦学习率调度器  
23| close_mosaic| int| 控制在最后几个 epochs 中是否禁用马赛克数据增强  
24| resume| bool| 用于从先前的训练检查点（checkpoint）中恢复模型的训练。  
25| amp| bool| 用于控制是否进行自动混合精度  
26| fraction| float| 用于指定训练数据集的一部分进行训练的比例。默认值为 1.0  
27| profile| bool| 用于控制是否在训练过程中启用 ONNX 和 TensorRT 的性能分析  
28| freeze| int/list/optinal| 用于指定在训练过程中冻结前 `n` 层或指定层索引的列表，以防止它们的权重更新。这对于迁移学习或特定层的微调很有用。  
  
* * *

### 5.1 训练的三种方式

#### 5.1.1 方式一

我们可以通过命令直接进行训练在其中指定参数，但是这样的方式，我们每个参数都要在其中打出来。命令如下:
    
    
    yolo task=detect mode=train model=yolov11n.pt data=data.yaml batch=16 epochs=100 imgsz=640 workers=0 device=0
    

> 需要注意的是如果你是Windows系统的电脑其中的Workers最好设置成0否则容易报线程的错误。

* * *

#### **5.1.2 方式二**

通过指定cfg直接进行训练，我们配置好ultralytics/cfg/default.yaml这个文件之后，可以直接执行这个文件进行训练，这样就不用在命令行输入其它的参数了。
    
    
    yolo cfg=ultralytics/cfg/default.yaml

* * *

#### **5.1.3 方式三 （推荐，避免keyError错误.）**

我们可以通过创建py文件来进行训练，这样的好处就是不用在终端上打命令，这也能省去一些工作量，我们在根目录下创建一个名字为train.py的文件，在其中输入代码
    
    
    import warnings
    warnings.filterwarnings('ignore')
    from ultralytics import YOLO
    
    if __name__ == '__main__':
        model = YOLO('yolo11.yaml')
        # 如何切换模型版本, 上面的ymal文件可以改为 yolov11s.yaml就是使用的v11s,
        # 类似某个改进的yaml文件名称为yolov11-XXX.yaml那么如果想使用其它版本就把上面的名称改为yolov11l-XXX.yaml即可（改的是上面YOLO中间的名字不是配置文件的）！
        # model.load('yolov11n.pt') # 是否加载预训练权重,科研不建议大家加载否则很难提升精度
        model.train(data=r"填写你数据集data.yaml文件的地址",
                    # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                    cache=False,
                    imgsz=640,
                    epochs=100,
                    single_cls=False,  # 是否是单类别检测
                    batch=4,
                    close_mosaic=0,
                    workers=0,
                    device='0',
                    optimizer='SGD', # using SGD 优化器 默认为auto建议大家使用固定的.
                    # resume=, # 续训的话这里填写True, yaml文件的地方改为lats.pt的地址,需要注意的是如果你设置训练200轮次模型训练了200轮次是没有办法进行续训的.
                    amp=True,  # 如果出现训练损失为Nan可以关闭amp
                    project='runs/train',
                    name='exp',
                    )
    
    

无论通过上述的哪一种方式在控制台输出如下图片的内容就代表着开始训练成功了！

![](https://i-blog.csdnimg.cn/direct/2b7e876b3ae74b9fa91410b4212a28e2.png)

* * *

## 六、模型验证/测试 

| 参数名| 类型| 参数讲解  
---|---|---|---  
1| val| bool| 用于控制是否在训练过程中进行验证/测试。  
2| split| str| 用于指定用于验证/测试的数据集划分。可以选择 'val'、'test' 或 'train' 中的一个作为验证/测试数据集  
3| save_json| bool| 用于控制是否将结果保存为 JSON 文件  
4| save_hybird| bool| 用于控制是否保存标签和附加预测结果的混合版本  
5| conf| float/optional| 用于设置检测时的目标置信度阈值  
6| iou| float| 用于设置非极大值抑制（NMS）的交并比（IoU）阈值。  
7| max_det| int| 用于设置每张图像的最大检测数。  
8| half| bool| 用于控制是否使用半精度（FP16）进行推断。  
9| dnn| bool| ，用于控制是否使用 OpenCV DNN 进行 ONNX 推断。  
10| plots| bool| 用于控制在训练/验证过程中是否保存绘图结果。  
  
验证我们划分的验证集/测试集的情况，也就是评估我们训练出来的best.pt模型好与坏

**命令行命令如下:**
    
    
    yolo task=detect mode=val model=best.pt data=data.yaml device=0
    

* * *

## 七、模型推理

我们训练好自己的模型之后，都会生成一个模型文件,保存在你设置的目录下,当我们再次想要实验该模型的效果之后就可以调用该模型进行推理了，我们也可以用官方的预训练权重来进行推理。

推理的方式和训练一样我们这里就选一种来进行举例其它的两种方式都是一样的操作只是需要改一下其中的一些参数即可:

**参数讲解**

| 参数名| 类型| 参数讲解  
---|---|---|---  
0| source| str/optinal| 用于指定图像或视频的目录  
1| show| bool| 用于控制是否在可能的情况下显示结果  
2| save_txt| bool| 用于控制是否将结果保存为 `.txt` 文件  
3| save_conf| bool| 用于控制是否在保存结果时包含置信度分数  
4| save_crop| bool| 用于控制是否将带有结果的裁剪图像保存下来  
5| show_labels| bool| 用于控制在绘图结果中是否显示目标标签  
6| show_conf| bool| 用于控制在绘图结果中是否显示目标置信度分数  
7| vid_stride| int/optional| 用于设置视频的帧率步长  
8| stream_buffer| bool| 用于控制是否缓冲所有流式帧（True）或返回最新的帧（False）  
9| line_width| int/list[int]/optional| 用于设置边界框的线宽度，如果缺失则自动设置  
10| visualize| bool| 用于控制是否可视化模型的特征  
11| augment| bool| 用于控制是否对预测源应用图像增强  
12| agnostic_nms| bool| 用于控制是否使用无关类别的非极大值抑制（NMS）  
13| classes| int/list[int]/optional| 用于按类别筛选结果  
14| retina_masks| bool| 用于控制是否使用高分辨率分割掩码  
15| boxes| bool| 用于控制是否在分割预测中显示边界框。  
  
**命令行命令如下:**
    
    
    yolo task=detect mode=predict model=best.pt source=images device=0
    

这里需要需要注意的是我们用模型进行推理的时候可以选择照片也可以选择一个视频的格式都可以。支持的视频格式有 

>   * MP4（.mp4）：这是一种常见的视频文件格式，通常具有较高的压缩率和良好的视频质量
> 
>   * AVI（.avi）：这是一种较旧但仍广泛使用的视频文件格式。它通常具有较大的文件大小
> 
>   * MOV（.mov）：这是一种常见的视频文件格式，通常与苹果设备和QuickTime播放器相关
> 
>   * MKV（.mkv）：这是一种开放的多媒体容器格式，可以容纳多个视频、音频和字幕轨道
> 
>   * FLV（.flv）：这是一种用于在线视频传输的流式视频文件格式
> 
> 

* * *

## 八、模型输出

当我们进行部署的时候可以进行文件导出，然后在进行部署。

YOLOv8支持的输出格式有如下

> 1\. ONNX（Open Neural Network Exchange）：ONNX 是一个开放的深度学习模型表示和转换的标准。它允许在不同的深度学习框架之间共享模型，并支持跨平台部署。导出为 ONNX 格式的模型可以在支持 ONNX 的推理引擎中进行部署和推理。
> 
> 2\. TensorFlow SavedModel：TensorFlow SavedModel 是 TensorFlow 框架的标准模型保存格式。它包含了模型的网络结构和参数，可以方便地在 TensorFlow 的推理环境中加载和使用。
> 
> 3\. PyTorch JIT（Just-In-Time）：PyTorch JIT 是 PyTorch 的即时编译器，可以将 PyTorch 模型导出为优化的 Torch 脚本或 Torch 脚本模型。这种格式可以在没有 PyTorch 环境的情况下进行推理，并且具有更高的性能。
> 
> 4\. Caffe Model：Caffe 是一个流行的深度学习框架，它使用自己的模型表示格式。导出为 Caffe 模型的文件可以在 Caffe 框架中进行部署和推理。
> 
> 5\. TFLite（TensorFlow Lite）：TFLite 是 TensorFlow 的移动和嵌入式设备推理框架，支持在资源受限的设备上进行高效推理。模型可以导出为 TFLite 格式，以便在移动设备或嵌入式系统中进行部署。
> 
> 6\. Core ML（Core Machine Learning）：Core ML 是苹果的机器学习框架，用于在 iOS 和 macOS 上进行推理。模型可以导出为 Core ML 格式，以便在苹果设备上进行部署。
> 
> 这些格式都提供了不同的优势和适用场景。选择合适的导出格式应该考虑到目标平台和部署环境的要求，以及所使用的深度学习框架的支持情况。

模型输出的参数有如下

| 参数名| 类型| 参数解释  
---|---|---|---  
0| format| str| 导出模型的格式  
1| keras| bool| 表示是否使用Keras  
2| optimize| bool| 用于在导出TorchScript模型时进行优化，以便在移动设备上获得更好的性能  
3| int8| bool| 用于在导出CoreML或TensorFlow模型时进行INT8量化  
4| dynamic| bool| 用于在导出CoreML或TensorFlow模型时进行INT8量化  
5| simplify| bool| 用于在导出ONNX模型时进行模型简化  
6| opset| int/optional| 用于指定导出ONNX模型时的opset版本  
7| workspace| int| 用于指定TensorRT模型的工作空间大小，以GB为单位  
8| nms| bool| 用于在导出CoreML模型时添加非极大值抑制（NMS）  
  
**命令行命令如下:**
    
    
    yolo task=detect mode=export model=best.pt format=onnx  
    

> 到此为止本文的讲解就结束了,希望对大家对于YOLOv11模型理解有帮助，希望本文能够帮助到大家。
