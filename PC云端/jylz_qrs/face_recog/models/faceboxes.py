import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
#所有的网络层都继承于nn.module
#in_channels输入数据的通道数，一张图片的特征图个数一般为RGB三个，但由于为了获得更多的特征，常常做一些处理，使得输入的特征图更多
#outchannels输出层特征图的个数，也可以认为是卷积核个数
#stride：卷积核的步长，可以是一个数字（表示水平和垂直方向的步长相同）或者一个元组（表示水平和垂直方向的步长分别设置）。
#padding：填充的大小，可以是一个数字（表示在水平和垂直方向上填充相同的大小）或者一个元组（表示在水平和垂直方向上填充不同的大小）。
#dilation：卷积核的膨胀率，可以是一个数字（表示水平和垂直方向上的膨胀率相同）或者一个元组（表示水平和垂直方向上的膨胀率不同）。
#groups：输入和输出之间的连接方式，可以是一个数字（表示使用分组卷积）。
#bias：是否使用偏置项，默认为True
#padding_mode：填充的方式，可以是'zeros'、'reflect'、'replicate'或'circular'。
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):

    def __init__(self):
        super(Inception, self).__init__()
        self.branch1x1 = BasicConv2d(128, 32, kernel_size=1, padding=0)
        #进去128张出来32张，卷积核大小1*1
        self.branch1x1_2 = BasicConv2d(128, 32, kernel_size=1, padding=0)
        #进去
        self.branch3x3_reduce = BasicConv2d(128, 24, kernel_size=1, padding=0)

        self.branch3x3 = BasicConv2d(24, 32, kernel_size=3, padding=1)
        self.branch3x3_reduce_2 = BasicConv2d(128, 24, kernel_size=1, padding=0)
        self.branch3x3_2 = BasicConv2d(24, 32, kernel_size=3, padding=1)
        self.branch3x3_3 = BasicConv2d(32, 32, kernel_size=3, padding=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        #在inception层中是几个网络分别对x进行卷进归一化再relu然后将最后的结果拼接
        #branch1*1:1,32,15,20
        branch1x1_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        # F.avg_pool2d函数对输入的张量 x进行了平均池化操作，得到了一个新的张量branch1x1_pool作为输出。具体来说，F.avg_pool2d
        # 函数对输入的张量在每个通道上进行了二维平均池化操作，即将每个通道上的二维矩阵按照给定的kernel_size（池化核大小）、stride（步长）和
        # padding（填充）参数进行划分，并在每个划分区域内计算平均值，最终得到一个输出张量。其中，kernel_size 表示池化核的大小，stride 表示步长，padding
        # 表示填充的大小。这里的 kernel_size = 3、stride = 1、padding = 1 表示使用一个 $3 \times3$ 的池化核，在每个 $3 \times
        # 3$ 的区域内计算平均值，同时使用 $1$ 个像素的填充来保证输出张量的大小与输入张量相同。因此，如果输入张量 x 的形状为[N, C, H, W]，那么输出张量
        # branch1x1_pool 的形状也为[N, C, H, W]，即与输入张量相同。
        branch1x1_2 = self.branch1x1_2(branch1x1_pool)

        branch3x3_reduce = self.branch3x3_reduce(x)
        #输出24张，
        branch3x3 = self.branch3x3(branch3x3_reduce)
        #将24张转换为32张
        branch3x3_reduce_2 = self.branch3x3_reduce_2(x)
        #输出24张
        branch3x3_2 = self.branch3x3_2(branch3x3_reduce_2)
        #将24张转换为32张
        branch3x3_3 = self.branch3x3_3(branch3x3_2)
        #将三十二张转换为32张

        outputs = [branch1x1, branch1x1_2, branch3x3, branch3x3_3]
        #将4个1*32*15*20的装在一个list中
        return torch.cat(outputs, 1)
    #将四个拼接起来变为1*128*15*20


class CRelu(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(CRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        # 这段代码定义了一个二维批量归一化（BatchNorm2d）层，用于对卷积层的输出进行标准化。具体来说，该批量归一化层的输入通道数为
        # out_channels，即卷积层的输出通道数；eps参数用于控制归一化时除数的平滑项，防止分母为0的情况发生。
        # 批量归一化是一种常用的正则化方法，用于加速模型训练并提高模型的泛化能力。在卷积层的输出中，不同通道的数值范围可能存在差异，
        # 导致模型难以收敛。批量归一化通过对每个通道的数值进行标准化，使得不同通道的数值范围相近，从而加速模型训练。此外，批量归一化
        # 还可以一定程度上缓解梯度消失和梯度爆炸等问题，提高模型的稳定性。因此，在深度学习中，批量归一化被广泛应用于卷积神经网络、循环神经网络等模型中。
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.cat([x, -x], 1)
        x = F.relu(x, inplace=True)
        # 这两行代码首先将输入的张量x沿着通道维度（即维度1）进行拼接，拼接后的张量是在通道维度上将原始张量
        # x和 - x拼在了一起，形成了一个通道数为原来的2倍的张量。具体来说，如果张量 x的形状为[N, C, H, W]，那么拼接后的张量的形状为[N, 2C, H, W]，
        # 其中N表示批次大小，C表示通道数，H和W分别表示空间维度的高和宽。

        # 对拼接后的张量 x进行了ReLU（整流线性单元）激活函数操作。ReLU
        # 激活函数的作用是将张量中所有小于0的元素都置为0，大于等于0的元素不变。这里的inplace = True
        # 表示对原始的张量进行就地修改，即将修改的结果直接保存在原始张量中，而不是新创建一个张量
        # 来保存结果。这样做的好处是可以节省内存空间，但是也会覆盖掉原始的张量数据，因此需要根据实际情况来决定是否使用
        # inplace。整个操作的结果是将输入的张量x沿着通道维度进行拼接，并对拼接后的张量进行了ReLU激活函数操作，得到了一个新的张量作为输出。
        return x


class FaceBoxes(nn.Module):

    def __init__(self, phase, size, num_classes):
        super(FaceBoxes, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size
        #创建了这么多网络层，名字与weights中对应
        self.conv1 = CRelu(3, 24, kernel_size=7, stride=4, padding=3)
        #每一个CRELU包含两层一个是卷积一个是归一化

        self.conv2 = CRelu(48, 64, kernel_size=5, stride=2, padding=2)
        # 进去是1*128*？

        self.inception1 = Inception()
        self.inception2 = Inception()
        self.inception3 = Inception()

        self.conv3_1 = BasicConv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.conv3_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv4_1 = BasicConv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.conv4_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.loc, self.conf = self.multibox(self.num_classes)
        #测试时
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)#在最后一个维度进行归一化，计算每个类别对应的分数

        if self.phase == 'train':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.xavier_normal_(m.weight.data)
                        m.bias.data.fill_(0.02)
                    else:
                        m.weight.data.normal_(0, 0.01)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        #如果为训练，需要将每一个层的参数初始化
        # Xavier
        # 初始化是一种常用的权重初始化方法，它的目的是使得网络在前向传播和反向传播时的梯度具有相同的
        # 数量级，从而加快训练速度。具体来说，Xavier
        # 初始化会根据神经元的输入和输出的维度计算权重的标准差，然后从一个均值为
        # 0、标准差为该值的正态分布中随机采样来初始化权重。

    def multibox(self, num_classes):
        loc_layers = []
        conf_layers = []

        loc_layers += [nn.Conv2d(128, 21 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(128, 21 * num_classes, kernel_size=3, padding=1)]
        loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
        loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)
        # nn.Sequential是PyTorch中的一个模块，可以将多个模块按照顺序组合在一起，形成一个大的模型。nn.Sequential
        # 接受一个或多个模块作为输入，然后将这些模块按照输入的顺序进行组合，形成一个新的模块。在组合的过程中，
        # 每个模块的输出都会作为下一个模块的输入，因此模块之间的连接是顺序连接。这个新的模块可以像单个模块一样使用，
        # 其输入输出与最后一个模块的输入输出相同。

    def forward(self, x):

        detection_sources = list()
        loc = list()
        conf = list()

        x = self.conv1(x)#RGB三通道注入后编成24个特征图，每个卷积核7*7，在进行前向计算时，每个函数的网络层的forwar函数都会被执行
        #然后再创立了一个
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        #在二维上最大池化
        # 用于对输入张量x进行下采样。具体来说，该最大池化操作的核大小为3×3，步幅为2，填充为1，即在输入
        # 张量的边缘填充一个像素，以便于核在边缘处进行卷积。在卷积神经网络中，最大池化操作通常用于减小特征图的大小，
        # 并帮助网络学习到更加鲁棒的特征。最大池化操作可以将每个池化窗口内的最大值作为该窗口的输出，从而减小特征图的大小，
        # 提高特征的局部不变性和平移不变性。在深度学习中，最大池化操作被广泛应用于卷积神经网络等模型中，可以有效地提高模型的性能和表达能力。
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.inception1(x)#注意是每一层推演都要执行一下当前模块的forward函数，inception第一层前向传播后完后，返回forward函数下一行
        x = self.inception2(x)
        x = self.inception3(x)
        #incepetion层进去和出来大小相同1*128*15*20
        # Inception层是Google在2014年的ImageNet大赛中提出的一种卷积网络结构，其结构包含了多个分支，并行处理输入的数据，
        # 最后将多个分支的输出进行拼接，
        # 形成一个更加丰富、多样化的特征表示。Inception层的好处主要包括以下几点：
        # 提高特征的多样性：Inception层使用了多个不同的卷积核，不同大小的池化层，同时还进行了1x1卷积，
        # 这些操作可以形成不同的卷积分支，提取出更加丰富多样的特征，有利于提高模型的泛化能力。
        #
        # 提高计算效率：使用1x1卷积来减少通道数，使得后面的卷积层的计算量大幅降低，同时也减少了参数数量，
        # 提高了模型的计算效率，有利于模型的训练和推理。
        # 提高模型的准确率：Inception层在ImageNet数据集上的表现非常优秀，相比于传统的卷积神经网络，
        # Inception网络可以在减少参数和计算量的情况下，提高模型的准确率，使得Inception网络成为了一种非常流行的网络结构，
        # 被广泛应用于计算机视觉领域的各种任务中。
        detection_sources.append(x)
        #在列表中增加第一个特征提取结果

        x = self.conv3_1(x)
        #进去128出来128
        x = self.conv3_2(x)
        #进去128出来256
        detection_sources.append(x)
        #增加第二个特征提取结果
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        # 进去128出来258
        detection_sources.append(x)
        #1，128，15，20
        #1，256，8，10
        #1，256，4，5

        #x是输入的特征量，l是loc——layer层网络，conf是conf_layer层网络，loc和conf都是基本的二维卷积神经网络
        for (x, l, c) in zip(detection_sources, self.loc, self.conf):
            # zip是python内置函数之一，它可以将多个可迭代对象中对应位置的元素打包成一个个元组，然后返回由这些元组组成的迭代器。具体来说，zip
            # 函数会取出每个可迭代对象中的第 $i$ 个元素，组成一个元组 $(
            # a_i, b_i, c_i, ...)$，然后返回这些元组组成的迭代器。如果各个可迭代对象的长度不同，则zip
            # 函数会以最短的可迭代对象为准，多余的元素会被忽略。
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
#             tensor.permute(*dims)方法可以用来对tensor
#             进行转置操作，其中 * dims表示要交换的维度的顺序，可以是一个tuple
#             或多个int值。转置之后，tensor 的维度顺序会发生改变，但tensor
#             的值不会发生改变。tensor.contiguous()方法可以用来返回一个与原 tensor具有相同值但存储方式不同的tensor。如果原
#             tensor在内存中是连续存储的，则tensor.contiguous()返回的tensor
#             与原tensor相同；否则，返回的tensor会先将原tensor拷贝到一个连续的内存区域中，然后再返回。在这里，l
#             是一个表示预测定位偏移量的网络层，x是输入给定图像的特征图。l(x)的返回值是一个四维的
# tensor，其维度顺序为 $(N, C, H, W)$，其中 $N$ 表示batch
# size，$C$ 表示特征通道数，$H$ 和 $W$ 分别表示特征图的高度和宽度。.permute(0, 2, 3, 1)将tensor
# 的维度顺序从 $(N, C, H, W)$ 转换为 $(N, H, W, C)$，即将特征通道数从第二个维度移动到最后一个维度，这样可以方便后续的处理。.contiguous()
# 方法可以保证tensor在内存中是连续存储的，这对后续的操作也是有帮助的，因为在PyTorch中有一些操作要求
# tensor必须是连续存储的。因此，l(x).permute(0, 2, 3, 1).contiguous()返回的是一个与l(x)
# 具有相同值但存储方式不同的tensor，其维度顺序为 $(N, H, W, C)$。
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        #conf:1,15,20,42 1,8,10,2 1,4,5,2

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        # 将多个Tensor拼接成一个大的
        # Tensor。具体来说，loc是一个列表，其中包含了多个shape相同的Tensor。o.view(o.size(0), -1)表示将oTensor
        # 的形状变为(o.size(0), -1)，展开为二维矩阵，按列进行拼接其中 - 1表示自动计算维度大小，以保证
        # Tensor的元素总数不变。因此，o.view(o.size(0), -1)
        # 的结果是一个二维的Tensor，其行数为o.size(0)，列数为。由于loc中的所有
        # Tensor的shape都相同，因此可以对loc
        # 中的每个Tensor都执行相同的操作，并将结果拼接成一个大的Tensor。torch.cat()
        # 函数的第一个参数是一个包含多个Tensor 的列表，第二个参数是要拼接的维度（默认为
        # 0）。因为o.view(o.size(0), -1)的结果是一个二维的Tensor，因此在这里，第二个参数为
        # -1，表示对每个二维Tensor的列进行拼接，最终得到一个二维Tensor。最后，将拼接后的
        # Tensor赋值给变量loc。这个操作的目的可能是将多个Tensor合并成一个，方便后续的处理。
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            # PyTorch中的view()函数，将形状为(batch_size, num_anchors * num_classes, H, W)
            # 的locTensor转换为形状为(batch_size, num_anchors * H * W, 4)的
            # Tensor，其中 - 1表示自动计算维度大小，以保证Tensor的元素总数不变。view()函数的作用是改变Tensor的形状，返回一个新的
            # Tensor，不改变原Tensor的值。这里的目的是将locTensor
            # 中每个像素点生成的anchor的坐标偏移量展开成一个长向量，方便后续的处理。具体来说，loc
            # Tensor的第1维是batch维，第维是anchor维，第3维和第4
            # 维是featuremap的高和宽。loc.view(loc.size(0), -1, 4)将locTensor的第2维和第3
            # 维展开成一个维度，即将每个像素点生成的anchor的坐标偏移量展开成一个长向量，形状为(batch_size, num_anchors * H * W, 4)。
            # 其中，第2维的大小为
            # num_anchors * H * W，表示每个像素点生成的anchor数量，第3维的大小为4，表示每个anchor对应的坐标偏移量。
            output = (loc.view(loc.size(0), -1, 4),
                      self.softmax(conf.view(conf.size(0), -1, self.num_classes)))
            #6400*4 6400*2
        else:
            output = (loc.view(loc.size(0), -1, 4),
                      #loc是边框偏移信息
                      #conf是置信度
                      conf.view(conf.size(0), -1, self.num_classes))

        return output
